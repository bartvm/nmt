from __future__ import print_function

import theano
from theano import tensor
import warnings
import six
from six.moves import xrange
import pickle
import sys
import itertools
import copy

import numpy
import inspect
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


class Parameters():
    def __init__(self):
        # self.__dict__['tparams'] = dict()
        self.__dict__['tparams'] = OrderedDict()

    def __setattr__(self, name, array):
        tparams = self.__dict__['tparams']
        # if name not in tparams:
        tparams[name] = array

    def __setitem__(self, name, array):
        self.__setattr__(name, array)

    def __getitem__(self, name):
        return self.__getattr__(name)

    def __getattr__(self, name):
        tparams = self.__dict__['tparams']
        return tparams[name]

    # def __getattr__(self):
    # return self.get()

    def remove(self, name):
        del self.__dict__['tparams'][name]

    def get(self):
        return self.__dict__['tparams']

    def values(self):
        tparams = self.__dict__['tparams']
        return tparams.values()

    def save(self, filename):
        tparams = self.__dict__['tparams']
        pickle.dump({p: tparams[p] for p in tparams}, open(filename, 'wb'), 2)

    def load(self, filename):
        tparams = self.__dict__['tparams']
        loaded = pickle.load(open(filename, 'rb'), encoding='latin1')
        for k in loaded:
            tparams[k] = loaded[k]

    def setvalues(self, values):
        tparams = self.__dict__['tparams']
        for p, v in zip(tparams, values):
            tparams[p] = v

    def __enter__(self):
        _, _, _, env_locals = inspect.getargvalues(inspect.currentframe(
        ).f_back)
        self.__dict__['_env_locals'] = env_locals.keys()

    def __exit__(self, type, value, traceback):
        _, _, _, env_locals = inspect.getargvalues(inspect.currentframe(
        ).f_back)
        prev_env_locals = self.__dict__['_env_locals']
        del self.__dict__['_env_locals']
        for k in env_locals.keys():
            if k not in prev_env_locals:
                self.__setattr__(k, env_locals[k])
                env_locals[k] = self.__getattr__(k)
        return True


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
                self._ret_queue.put(ret)
            except RuntimeError as err:
                print(err, file=sys.stderr)

                # stop the timer
                self.stop()

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
        next_alphas = bs_state[2]

        if decode_char:
            next_char_state, cproj = bs_state[3], bs_state[4]

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
        new_hyp_alignment = []
        new_hyp_char_samples = []
        new_hyp_char_state = []
        new_hyp_cproj = []

    for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
        new_hyp_samples.append(hypotheses['samples'][ti] + [wi])
        new_hyp_scores[idx] = copy.copy(costs[idx])
        new_hyp_states.append(copy.copy(next_state[ti]))

        if level == 'word':
            new_hyp_alignment.append(
                hypotheses['alignments'][ti] +
                [copy.copy(next_alphas[ti])]
            )
            if decode_char:
                # NOTE just copy of character sequences generated previously
                new_hyp_char_samples.append(
                    copy.copy(hypotheses['character_samples'][ti]))
                new_hyp_char_state.append(copy.copy(next_char_state[ti]))
                new_hyp_cproj.append(copy.copy(cproj[ti]))

    # check the finished samples
    updated_hypotheses = OrderedDict([
        ('num_samples', 0),
        ('samples', []),
        ('scores', []),
        ('states', []),
    ])

    if level == 'word':
        updated_hypotheses['alignments'] = []

        if decode_char:
            updated_hypotheses['character_samples'] = []
            updated_hypotheses['char_state'] = []
            updated_hypotheses['cproj'] = []

    for idx in xrange(len(new_hyp_samples)):
        if new_hyp_samples[idx][-1] == 0:
            # if the last word is the EOS token
            solutions['num_samples'] += 1

            solutions['samples'].append(new_hyp_samples[idx])
            solutions['scores'].append(new_hyp_scores[idx])

            if level == 'word':
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
                updated_hypotheses['alignments'].append(new_hyp_alignment[idx])
                if decode_char:
                    updated_hypotheses['character_samples'].append(
                        new_hyp_char_samples[idx])
                    updated_hypotheses['char_state'].append(
                        new_hyp_char_state[idx])
                    updated_hypotheses['cproj'].append(new_hyp_cproj[idx])

    assert updated_hypotheses['num_samples'] + solutions['num_samples'] == k

    updated_hypotheses['scores'] = numpy.array(updated_hypotheses['scores'])

    return solutions, updated_hypotheses
