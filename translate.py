'''
Translates a source file using a translation model.
'''
from __future__ import print_function

import argparse
import signal
import sys
import traceback
import numpy
import io
import json
from multiprocessing import Process, Queue, Event

from six.moves import xrange

from data_iterator import (load_dict, EOS_TOKEN, UNK_TOKEN)
from nmt_base import (build_sampler, gen_sample, init_params)
from utils import (load_params, init_tparams)


# utility function
def _send_jobs(fname, queue, word_dict, n_words_src):
    with io.open(fname, 'r') as f:
        for idx, line in enumerate(f):
            words = line.strip().split()
            words += [EOS_TOKEN]
            x = map(lambda w: word_dict[w] if w in word_dict else 1, words)
            x = map(lambda ii: ii if ii < n_words_src else 1, x)
            queue.put((idx, x, words))
    return idx+1


def _retrieve_jobs(rqueue, n_samples):
    trans = [None] * n_samples
    for idx in xrange(n_samples):
        resp = rqueue.get()
        trans[resp[0]] = resp[1]
    return trans


def translate_model(exit_event, queue, rqueue, pid, i2w_trg,
                    model, options, k, normalize, unk_replace):

    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng = RandomStreams(1234)
    # use_noise = theano.shared(numpy.float32(0.))

    # allocate model parameters
    params = init_params(options)

    # load model parameters and set theano shared variables
    params = load_params(model, params)
    tparams = init_tparams(params)

    # word index
    f_init, f_next = build_sampler(tparams, options, trng)

    def _seq2words(seq, word_idict_trg):
        words = []
        for w in seq:
            if w == 0:
                break
            words.append(word_idict_trg[w])
        return words

    def _translate(seq):
        seq = numpy.array(seq).reshape([len(seq), 1])

        # sample given an input sequence and obtain scores
        samples, alignments, scores = gen_sample(tparams, f_init, f_next,
                                                 seq, options, trng=trng,
                                                 k=k, maxlen=100,
                                                 stochastic=False,
                                                 argmax=False)

        # normalize scores according to sequence lengths
        if normalize:
            lengths = numpy.array([len(s) for s in samples])
            scores = scores / lengths
        sidx = numpy.argmin(scores)
        return samples[sidx], alignments[sidx]

    def _replace_unk(trans_words, src_words, alignment):
        for idx, word in enumerate(trans_words):
            if word == UNK_TOKEN:
                # pick a source word
                # with which the target word is strongly aligned
                # except for the EOS token
                trans_words[idx] = src_words[alignment[idx][:-1].argmax()]

        return trans_words

    while (not exit_event.is_set()) and (not queue.empty()):
        req = queue.get()
        if req is None:
            break

        # idx: sentence index
        # x: a sequence of word indices
        # src_words: original source sentence
        idx, x, src_words = req[0], req[1], req[2]
        seq, alignment = _translate(x)

        assert len(seq) == len(alignment)

        # indices to tokens
        trans_words = _seq2words(seq, i2w_trg)

        if unk_replace:
            # unknown word replacement
            trans_words = _replace_unk(trans_words, src_words, alignment)

        # list of words to a single string
        trans_words = ' '.join(trans_words)

        # put the result
        rqueue.put((idx, trans_words))

    # write to release resources if needed

    return


def main(model_path, option_path, dictionary_source, dictionary_target,
         source_file, saveto, k=5, normalize=False, unk_replace=False,
         n_process=5):

    # load model_options
    with io.open(option_path) as f:
        config = json.load(f)

    options = config['model']

    # load source dictionary and invert
    word_dict = load_dict(dictionary_source, n_words=options['n_words_src'])
    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk

    # load target dictionary and invert
    word_dict_trg = load_dict(dictionary_target, n_words=options['n_words'])
    word_idict_trg = dict()
    for kk, vv in word_dict_trg.iteritems():
        word_idict_trg[vv] = kk

    default_sigint_handler = signal.getsignal(signal.SIGINT)

    # make child processes ignore interrupt signals
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # create input and output queues for processes
    queue = Queue()     # producer
    rqueue = Queue()    # consumer
    exit_event = Event()

    # put all of the sentences into the producer queue
    n_samples = _send_jobs(source_file,
                           queue,
                           word_dict,
                           options['n_words_src'])

    # add sentinel values into the producer queue
    for midx in xrange(n_process):
        queue.put(None)

    # spawning workers
    processes = [None] * n_process
    for midx in xrange(n_process):
        processes[midx] = Process(
            target=translate_model,
            args=(exit_event, queue, rqueue, midx, word_idict_trg,
                  model_path, options, k, normalize, unk_replace))

    for proc in processes:
        proc.start()

    signal.signal(signal.SIGINT, default_sigint_handler)

    def _stop_child_processes():
        exit_event.set()

    def _clear_resources():

        for proc in processes:
            if proc.is_alive():
                proc.terminate()

        while not queue.empty():
            queue.get()

        while not rqueue.empty():
            rqueue.get()

    def _signal_handler(signum, frame):
        # print('Received an interrpt signal.')
        # print('Please wait for releasing resources...')

        _stop_child_processes()
        _clear_resources()

        sys.exit(130)

    signal.signal(signal.SIGINT, _signal_handler)

    # collecting translated sentences from the return queue
    trans = _retrieve_jobs(rqueue, n_samples)

    try:
        with io.open(saveto, 'w') as f:
            print('\n'.join(trans), file=f)
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)

        _stop_child_processes()
        _clear_resources()

        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=5)
    parser.add_argument('-p', type=int, default=5)
    parser.add_argument('-n', action="store_true", default=False)
    parser.add_argument('-u', action="store_true", default=False)
    parser.add_argument('model_path', type=str)
    parser.add_argument('option_path', type=str)
    parser.add_argument('dictionary_source', type=str)
    parser.add_argument('dictionary_target', type=str)
    parser.add_argument('source', type=str)
    parser.add_argument('saveto', type=str)

    args = parser.parse_args()

    main(args.model_path, args.option_path,
         args.dictionary_source, args.dictionary_target,
         args.source, args.saveto, k=args.k, normalize=args.n,
         unk_replace=args.u, n_process=args.p)
