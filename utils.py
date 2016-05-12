from __future__ import print_function

import theano
from theano import tensor
import warnings
import six
from six.moves import xrange
import itertools
import copy

import numpy
from threading import Timer
from collections import OrderedDict


# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in six.iteritems(params):
        tparams[kk].set_value(vv)


# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in six.iteritems(zipped):
        new_params[kk] = vv.get_value()
    return new_params


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in six.iteritems(tparams)]


# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         state_before *
                         trng.binomial(state_before.shape,
                                       p=0.5,
                                       n=1,
                                       dtype=state_before.dtype),
                         state_before * 0.5)
    return proj


# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in six.iteritems(params):
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in six.iteritems(params):
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]

    return params


# some utilities
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')


def uniform_weight(nin, nout, scale=None):
    if scale is None:
        scale = numpy.sqrt(6. / (nin + nout))

    W = numpy.random.uniform(low=-scale, high=scale, size=(nin, nout))
    return W.astype('float32')


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k], )
    output_shape += (concat_size, )
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k], )

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None), )
        indices += (slice(offset, offset + tt.shape[axis]), )
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None), )

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out


class RepeatedTimer(object):
    def __init__(self, interval, function, return_queue,
                 *args, **kwargs):
        self._timer = None
        self._interval = interval
        self.function = function  # function bound to the timer
        # put return values of the function
        self._ret_queue = return_queue
        self.args = args
        self.kwargs = kwargs
        self._is_running = False    # Is the timer running?
        self._is_func_running = False

    def _run(self):
        self._is_running = False
        self.start()    # set a new Timer with pre-specified interval

        # check if the function is running
        if not self._is_func_running:
            self._is_func_running = True
            try:
                ret = self.function(*self.args, **self.kwargs)
            except RuntimeError as err:
                # stop the timer
                self.stop()
                ret = [Exception(err)]
            finally:
                self._ret_queue.put(ret)
                self._is_func_running = False

    def start(self):
        if not self._is_running:
            self._timer = Timer(self._interval, self._run)
            self._timer.start()
            self._is_running = True  # timer is running

    def stop(self):
        self._timer.cancel()
        self._is_running = False
        self._is_func_running = False


def prepare_character_tensor(cx):

    def isplit(iterable, splitters):
        return [list(g) for k, g in itertools.groupby(iterable,
                lambda x:x in splitters) if not k]

    # index of 'white space' is 2
    # sents = [isplit(sent, (2,)) + [[0]] for sent in cx]
    sents = [isplit(sent, (2,)) for sent in cx]
    num_sents = len(cx)
    num_words = numpy.max([len(sent) + 1 for sent in sents])

    # word lengths in a batch of sentences
    word_lengths = \
        [
            # assume the end of word token
            [len(word)+1 for word in sent]
            for sent in sents
        ]

    max_word_len = numpy.max(
        [
            w_len for w_lengths in word_lengths
            for w_len in w_lengths
        ])

    chars = numpy.zeros(
        [
            max_word_len,
            num_words,
            num_sents
        ], dtype='int64')

    chars_mask = numpy.zeros(
        [
            max_word_len,
            num_words,
            num_sents
        ], dtype='float32')

    for sent_idx, sent in enumerate(sents):
        for word_idx, word in enumerate(sent):
            chars[:word_lengths[sent_idx][word_idx]-1,
                  word_idx,
                  sent_idx] = sents[sent_idx][word_idx]

            chars_mask[:word_lengths[sent_idx][word_idx],
                       word_idx,
                       sent_idx] = 1.

    return chars, chars_mask


def beam_search(solutions, hypotheses,
                bs_state, k=1, decode_char=False, level='word'):
    """Performs beam search.

    Parameters:
    ----------
        solutions : dict
            See

        hypotheses : dict
            See

        bs_state : list
            State of beam search

        k : int
            Size of beam

        decode_char : boolean
            Character generation

    Returns:
    -------
        updated_solutions : dict

        updated_hypotheses : dict
    """

    assert len(bs_state) >= 2

    next_state, next_p = bs_state[0], bs_state[1]

    if level == 'word':
        next_alphas, next_trg_gates = bs_state[2], bs_state[3]

        if decode_char:
            next_word_ctxs, prev_word_inps = \
                bs_state[4], bs_state[5]

    # NLL: the lower, the better
    cand_scores = hypotheses['scores'][:, None] - numpy.log(next_p)
    cand_flat = cand_scores.flatten()
    # select (k - dead_k) best words or characters
    # argsort's default order: ascending
    ranks_flat = cand_flat.argsort()[:(k - solutions['num_samples'])]
    costs = cand_flat[ranks_flat]

    voc_size = next_p.shape[1]
    # translation candidate indices
    trans_indices = (ranks_flat / voc_size).astype('int64')
    word_indices = ranks_flat % voc_size

    new_hyp_samples = []
    new_hyp_scores = numpy.zeros(
        k - solutions['num_samples']).astype('float32')
    new_hyp_states = []

    if level == 'word':
        new_hyp_word_trg_gate = []
        new_hyp_alignment = []
        new_hyp_char_samples = []
        new_hyp_prev_word_inps = []
        new_hyp_word_ctxs = []

    for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
        new_hyp_samples.append(hypotheses['samples'][ti] + [wi])
        new_hyp_scores[idx] = copy.copy(costs[idx])
        new_hyp_states.append(copy.copy(next_state[ti]))

        if level == 'word':
            new_hyp_word_trg_gate.append(
                hypotheses['word_trg_gates'][ti] +
                [copy.copy(next_trg_gates[ti])]
            )
            new_hyp_alignment.append(
                hypotheses['alignments'][ti] +
                [copy.copy(next_alphas[ti])]
            )
            if decode_char:
                # NOTE just copy of character sequences generated previously
                new_hyp_char_samples.append(
                    copy.copy(hypotheses['character_samples'][ti]))
                new_hyp_prev_word_inps.append(copy.copy(prev_word_inps[ti]))
                new_hyp_word_ctxs.append(copy.copy(next_word_ctxs[ti]))

    # check the finished samples
    updated_hypotheses = OrderedDict([
        ('num_samples', 0),
        ('samples', []),
        ('scores', []),
        ('states', []),
    ])

    if level == 'word':
        updated_hypotheses['word_trg_gates'] = []
        updated_hypotheses['alignments'] = []

        if decode_char:
            updated_hypotheses['character_samples'] = []
            updated_hypotheses['prev_word_inps'] = []
            updated_hypotheses['word_ctxs'] = []

    for idx in xrange(len(new_hyp_samples)):
        if new_hyp_samples[idx][-1] == 0:
            # if the last word is the EOS token
            solutions['num_samples'] += 1

            solutions['samples'].append(new_hyp_samples[idx])
            solutions['scores'].append(new_hyp_scores[idx])

            if level == 'word':
                solutions['word_trg_gates'].append(new_hyp_word_trg_gate[idx])
                solutions['alignments'].append(new_hyp_alignment[idx])

                if decode_char:
                    solutions['character_samples'].append(
                        new_hyp_char_samples[idx])
        else:
            updated_hypotheses['num_samples'] += 1

            updated_hypotheses['samples'].append(new_hyp_samples[idx])
            updated_hypotheses['scores'].append(new_hyp_scores[idx])
            updated_hypotheses['states'].append(new_hyp_states[idx])

            if level == 'word':
                updated_hypotheses['word_trg_gates'].append(
                    new_hyp_word_trg_gate[idx])
                updated_hypotheses['alignments'].append(new_hyp_alignment[idx])
                if decode_char:
                    updated_hypotheses['character_samples'].append(
                        new_hyp_char_samples[idx])
                    updated_hypotheses['prev_word_inps'].append(
                        new_hyp_prev_word_inps[idx])
                    updated_hypotheses['word_ctxs'].append(
                        new_hyp_word_ctxs[idx])

    assert updated_hypotheses['num_samples'] + solutions['num_samples'] == k

    updated_hypotheses['scores'] = numpy.array(updated_hypotheses['scores'])

    return solutions, updated_hypotheses
