'''
Translates a source file using a translation model.
'''
from __future__ import print_function
import argparse

import numpy
from data_iterator import load_dict

import io
import json
import signal
import sys

from nmt_base import (build_sampler, gen_sample, init_params)

from utils import (load_params, init_tparams)

from multiprocessing import Process, Queue, Event


# utility function
def _seqs2words(caps, word_idict_trg):
    capsw = []
    for cc in caps:
        ww = []
        for w in cc:
            if w == 0:
                break
            ww.append(word_idict_trg[w])
        capsw.append(' '.join(ww))
    return capsw


def _send_jobs(fname, queue, word_dict, n_words_src):
    with open(fname, 'r') as f:
        for idx, line in enumerate(f):
            words = line.strip().split()
            x = map(lambda w: word_dict[w] if w in word_dict else 1, words)
            x = map(lambda ii: ii if ii < n_words_src else 1, x)
            x += [0]
            queue.put((idx, x))
    return idx+1


def _finish_processes(queue, n_process):
    for midx in xrange(n_process):
        queue.put(None)


def _retrieve_jobs(rqueue, n_samples):
    trans = [None] * n_samples
    for idx in xrange(n_samples):
        resp = rqueue.get()
        trans[resp[0]] = resp[1]
    return trans


def translate_model(exit_event, queue, rqueue, pid,
                    model, options, k, normalize):

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

    def _translate(seq):
        # sample given an input sequence and obtain scores
        sample, score = gen_sample(tparams, f_init, f_next,
                                   numpy.array(seq).reshape([len(seq), 1]),
                                   options, trng=trng, k=k, maxlen=200,
                                   stochastic=False, argmax=False)

        # normalize scores according to sequence lengths
        if normalize:
            lengths = numpy.array([len(s) for s in sample])
            score = score / lengths
        sidx = numpy.argmin(score)
        return sample[sidx]

    while not exit_event.is_set():
        req = queue.get()
        if req is None:
            break

        idx, x = req[0], req[1]
        seq = _translate(x)

        rqueue.put((idx, seq))

    # write to release resources if needed

    return


def main(model_path, option_path, dictionary_source, dictionary_target,
         source_file, saveto, k=5, normalize=False, n_process=5):

    # load model_options
    with io.open(option_path) as f:
        config = json.load(f)

    options = config['model']

    # load source dictionary and invert
    word_dict = load_dict(dictionary_source, n_words=options['n_words_src']+2)
    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk

    # load target dictionary and invert
    word_dict_trg = load_dict(dictionary_target,
                              n_words=options['n_words']+2)
    word_idict_trg = dict()
    for kk, vv in word_dict_trg.iteritems():
        word_idict_trg[vv] = kk

    default_sigint_handler = signal.getsignal(signal.SIGINT)

    # make child processes ignore interrupt signals
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # create input and output queues for processes
    queue = Queue()
    rqueue = Queue()
    exit_event = Event()
    processes = [None] * n_process
    for midx in xrange(n_process):
        processes[midx] = Process(
            target=translate_model,
            args=(exit_event, queue, rqueue, midx,
                  model_path, options, k, normalize))
        processes[midx].start()

    signal.signal(signal.SIGINT, default_sigint_handler)

    def _signal_handler(signum, frame):
        print('Received an interrpt signal.')
        print('Please wait for releasing resources...')

        exit_event.set()

        for proc in processes:
            proc.join()

        while not queue.empty():
            queue.get()

        while not rqueue.empty():
            rqueue.get()

        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)

    n_samples = _send_jobs(source_file,
                           queue,
                           word_dict,
                           options['n_words_src'])

    # wait until all child processes finish the translation job
    for proc in processes:
        proc.join()

    # collecting translated sentences from the return queue
    trans = _seqs2words(_retrieve_jobs(rqueue, n_samples), word_idict_trg)
    _finish_processes(queue, n_process)

    with open(saveto, 'w') as f:
        print('\n'.join(trans), file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=5)
    parser.add_argument('-p', type=int, default=5)
    parser.add_argument('-n', action="store_true", default=False)
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
         n_process=args.p)
