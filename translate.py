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

from data_iterator import (load_dict, EOW_TOKEN, EOS_TOKEN, UNK_TOKEN)
from nmt_base import (build_sampler, gen_sample, init_params)
from utils import (load_params, init_tparams, prepare_character_tensor)


# utility function
def _send_jobs(fname, queue, char_dict_src, word_dict_src,
               n_chars_src, n_words_src):
    with io.open(fname, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            words = line.split()
            words += [EOS_TOKEN]
            line += EOW_TOKEN
            x = map(lambda w: word_dict_src[w] if w in word_dict_src else 1,
                    words)
            x = map(lambda ii: ii if ii < n_words_src else 1, x)

            xc = map(lambda c: char_dict_src[c] if c in char_dict_src else 1,
                     line)
            xc = [map(lambda ii: ii if ii < n_chars_src else 1, xc)]

            queue.put((idx, xc, x, words))
    return idx+1


def _retrieve_jobs(rqueue, n_samples):
    trans = [None] * n_samples
    for idx in xrange(n_samples):
        resp = rqueue.get()
        trans[resp[0]] = resp[1]
    return trans


def translate_model(exit_event, queue, rqueue, pid,
                    i2w_trg, i2c_trg, model,
                    options, k, normalize, unk_replace, use_character):

    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng = RandomStreams(1234)
    # use_noise = theano.shared(numpy.float32(0.))

    # allocate model parameters
    params = init_params(options)

    # load model parameters and set theano shared variables
    params = load_params(model, params)
    tparams = init_tparams(params)

    # word index
    f_inits, f_nexts = build_sampler(tparams, options, trng)

    def _seq2words(seq, word_idict_trg):
        words = []
        for w in seq:
            if w == 0:
                break
            words.append(word_idict_trg[w])
        return words

    def _charseq2words(char_seq, char_idict_trg):
        words = []
        for word in char_seq:
            token = ''
            for character in word:
                if character == 0:
                    break
                if character in char_idict_trg:
                    token += char_idict_trg[character]
                else:
                    token += UNK_TOKEN

            words.append(token)

        return words

    def _translate(x, x_mask, xc=None, xc_mask=None):
        assert x.ndim == 2
        if xc is not None and xc_mask is not None:
            assert use_character
            assert x.shape[0] == xc.shape[1]
            assert xc.ndim == 3
            assert xc_mask.ndim == 3

        inps = [x, x_mask]
        if use_character:
            inps += [xc, xc_mask]

        # sample given an input sequence and obtain scores
        word_solutions = gen_sample(tparams, f_inits, f_nexts,
                                    inps,
                                    options, trng=trng,
                                    k=k, max_sent_len=200,
                                    max_word_len=60,
                                    argmax=False)

        samples = word_solutions['samples']
        alignments = word_solutions['alignments']
        scores = word_solutions['scores']

        # normalize scores according to sequence lengths
        if normalize:
            lengths = numpy.array([len(s) for s in samples])
            scores = scores / lengths
        sidx = numpy.argmin(scores)
        samples = samples[sidx]
        alignments = alignments[sidx]

        translation_outputs = [samples, alignments]

        if use_character:
            word_characters = word_solutions['character_samples']
            word_characters = word_characters[sidx]

            translation_outputs += [word_characters]

        return translation_outputs

    def _replace_unk(trans_words, src_words,
                     alignment, trans_words_char):
        if use_character:
            assert len(trans_words) == len(trans_words_char)

        for idx, word in enumerate(trans_words):
            if word == UNK_TOKEN:
                if use_character:
                    trans_words[idx] = trans_words_char[idx]
                else:
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
        # xc: a sequence of character sequences
        # x: a sequence of word indices
        # src_words: original source sentence
        idx, xc, x, src_words = req

        x = numpy.array(x).reshape([len(x), 1])
        x_mask = numpy.ones_like(x).astype('float32')

        inps = [x, x_mask]

        if use_character:
            xc, xc_mask = prepare_character_tensor(xc)
            inps += [xc, xc_mask]

        # seq, alignment = _translate(*inps)
        trans_outs = _translate(*inps)

        seq = trans_outs[0]
        alignment = trans_outs[1]

        assert len(seq) == len(alignment)

        if len(trans_outs) >= 3:
            char_seq = trans_outs[2]

        # indices to word tokens
        trans_words = _seq2words(seq, i2w_trg)
        replace_inps = [trans_words, src_words, alignment]

        if use_character:
            # indices to characters (word)
            trans_words_char = _charseq2words(char_seq, i2c_trg)

            assert len(trans_words) == len(trans_words_char)

            replace_inps += [trans_words_char]

        if unk_replace:
            # unknown word replacement
            trans_words = _replace_unk(*replace_inps)

        # list of words to a single string
        trans_words = ' '.join(trans_words)

        # put the result
        rqueue.put((idx, trans_words))

    # write to release resources if needed

    return


def main(model_path, option_path,
         char_vocab_src, word_vocab_src,
         char_vocab_trg, word_vocab_trg,
         source_file, saveto, k=5, normalize=False, unk_replace=False,
         n_process=5):

    # load model_options
    with io.open(option_path) as f:
        config = json.load(f)

    options = config['model']
    use_character = options['use_character']

    # load source dictionary and invert
    char_dict_src = load_dict(char_vocab_src, dict_size=options['n_chars_src'])
    word_dict_src = load_dict(word_vocab_src, dict_size=options['n_words_src'])
    # word_idict_src = dict([(vv, kk) for kk, vv in word_dict.iteritems()])

    # load target dictionary and invert
    char_dict_trg = load_dict(char_vocab_trg, dict_size=options['n_chars_trg'])
    char_idict_trg = dict([(vv, kk) for kk, vv in char_dict_trg.iteritems()])
    word_dict_trg = load_dict(word_vocab_trg, dict_size=options['n_words_trg'])
    word_idict_trg = dict([(vv, kk) for kk, vv in word_dict_trg.iteritems()])

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
                           char_dict_src,
                           word_dict_src,
                           options['n_chars_src'],
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
                  char_idict_trg, model_path, options, k, normalize,
                  unk_replace, use_character))
        processes[midx].start()

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

        sys.exit(130)

    def _signal_handler(signum, frame):
        _stop_child_processes()
        _clear_resources()

    signal.signal(signal.SIGINT, _signal_handler)

    try:
        # collecting translated sentences from the return queue
        trans = _retrieve_jobs(rqueue, n_samples)

        with io.open(saveto, 'w', encoding='utf-8') as f:
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
    parser.add_argument('char_vocab_src', type=str)
    parser.add_argument('word_vocab_src', type=str)
    parser.add_argument('char_vocab_trg', type=str)
    parser.add_argument('word_vocab_trg', type=str)
    parser.add_argument('source', type=str)
    parser.add_argument('saveto', type=str)

    args = parser.parse_args()

    main(args.model_path, args.option_path,
         args.char_vocab_src, args.word_vocab_src,
         args.char_vocab_trg, args.word_vocab_trg,
         args.source, args.saveto, k=args.k, normalize=args.n,
         unk_replace=args.u, n_process=args.p)
