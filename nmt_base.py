'''
Build a neural machine translation model with soft attention
'''
from __future__ import print_function
import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams

import six
from six.moves import xrange
import ipdb
import numpy
import copy

import sys

from collections import OrderedDict
from utils import dropout_layer, norm_weight, concatenate
from layers import get_layer
from data_iterator import get_stream, load_dict


def load_data(src, trg,
              valid_src, valid_trg,
              src_vocab, trg_vocab,
              n_words, n_words_src,
              batch_size, valid_batch_size):
    print('Loading data')

    dictionaries = [src_vocab, trg_vocab]
    datasets = [src, trg]
    valid_datasets = [valid_src, valid_trg]

    # load dictionaries and invert them
    worddicts = [None] * len(dictionaries)
    worddicts_r = [None] * len(dictionaries)
    for ii, dd in enumerate(dictionaries):
        worddicts[ii] = load_dict(dd)
        worddicts_r[ii] = dict()
        for kk, vv in six.iteritems(worddicts[ii]):
            worddicts_r[ii][vv] = kk

    train_stream = get_stream([datasets[0]],
                              [datasets[1]],
                              dictionaries[0],
                              dictionaries[1],
                              n_words_source=n_words_src,
                              n_words_target=n_words,
                              batch_size=batch_size)
    valid_stream = get_stream([valid_datasets[0]],
                              [valid_datasets[1]],
                              dictionaries[0],
                              dictionaries[1],
                              n_words_source=n_words_src,
                              n_words_target=n_words,
                              batch_size=valid_batch_size)

    return worddicts_r, train_stream, valid_stream


# initialize all parameters
def init_params(options):
    params = OrderedDict()

    # embedding
    params['Wemb'] = norm_weight(options['n_words_src'],
                                 options['dim_word_src'])
    params['Wemb_dec'] = norm_weight(options['n_words'],
                                     options['dim_word_trg'])

    # encoder: bidirectional RNN
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix='encoder',
                                              nin=options['dim_word_src'],
                                              dim=options['dim'])
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix='encoder_r',
                                              nin=options['dim_word_src'],
                                              dim=options['dim'])
    ctxdim = 2 * options['dim']

    # init_state, init_cell
    params = get_layer('ff')[0](options,
                                params,
                                prefix='ff_state',
                                nin=ctxdim,
                                nout=options['dim'])
    # decoder
    params = get_layer(options['decoder'])[0](options,
                                              params,
                                              prefix='decoder',
                                              nin=options['dim_word_trg'],
                                              dim=options['dim'],
                                              dimctx=ctxdim)
    # readout
    params = get_layer('ff')[0](options,
                                params,
                                prefix='ff_logit_lstm',
                                nin=options['dim'],
                                nout=options['dim_word_trg'],
                                ortho=False)
    params = get_layer('ff')[0](options,
                                params,
                                prefix='ff_logit_prev',
                                nin=options['dim_word_trg'],
                                nout=options['dim_word_trg'],
                                ortho=False)
    params = get_layer('ff')[0](options,
                                params,
                                prefix='ff_logit_ctx',
                                nin=ctxdim,
                                nout=options['dim_word_trg'],
                                ortho=False)
    params = get_layer('ff')[0](options,
                                params,
                                prefix='ff_logit',
                                nin=options['dim_word_trg'],
                                nout=options['n_words'])

    return params


# build a training model
def build_model(tparams, options):
    opt_ret = dict()

    trng = MRG_RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')
    y = tensor.matrix('y', dtype='int64')
    y_mask = tensor.matrix('y_mask', dtype='float32')

    # for the backward rnn, we just need to invert x and x_mask
    xr = x[::-1]
    xr_mask = x_mask[::-1]

    n_timesteps = x.shape[0]
    n_timesteps_trg = y.shape[0]
    n_samples = x.shape[1]

    # word embedding for forward rnn (source)
    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word_src']])
    proj = get_layer(options['encoder'])[1](tparams,
                                            emb,
                                            options,
                                            prefix='encoder',
                                            mask=x_mask)
    # word embedding for backward rnn (source)
    embr = tparams['Wemb'][xr.flatten()]
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word_src']])
    projr = get_layer(options['encoder'])[1](tparams,
                                             embr,
                                             options,
                                             prefix='encoder_r',
                                             mask=xr_mask)

    # context will be the concatenation of forward and backward rnns
    ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim - 1)

    # mean of the context (across time) will be used to initialize decoder rnn
    ctx_mean = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]

    # or you can use the last state of forward + backward encoder rnns
    # ctx_mean = concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)

    # initial decoder state
    init_state = get_layer('ff')[1](tparams,
                                    ctx_mean,
                                    options,
                                    prefix='ff_state',
                                    activ=tensor.tanh)

    # word embedding (target), we will shift the target sequence one time step
    # to the right. This is done because of the bi-gram connections in the
    # readout and decoder rnn. The first target will be all zeros and we will
    # not condition on the last output.
    emb = tparams['Wemb_dec'][y.flatten()]
    emb = emb.reshape([n_timesteps_trg, n_samples, options['dim_word_trg']])
    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted

    # decoder - pass through the decoder conditional gru with attention
    proj = get_layer(options['decoder'])[1](tparams,
                                            emb,
                                            options,
                                            prefix='decoder',
                                            mask=y_mask,
                                            context=ctx,
                                            context_mask=x_mask,
                                            one_step=False,
                                            init_state=init_state)
    # hidden states of the decoder gru
    proj_h = proj[0]

    # weighted averages of context, generated by attention module
    ctxs = proj[1]

    # weights (alignment matrix)
    opt_ret['dec_alphas'] = proj[2]

    # compute word probabilities
    logit_lstm = get_layer('ff')[1](tparams,
                                    proj_h,
                                    options,
                                    prefix='ff_logit_lstm',
                                    activ=None)
    logit_prev = get_layer('ff')[1](tparams,
                                    emb,
                                    options,
                                    prefix='ff_logit_prev',
                                    activ=None)
    logit_ctx = get_layer('ff')[1](tparams,
                                   ctxs,
                                   options,
                                   prefix='ff_logit_ctx',
                                   activ=None)
    logit = tensor.tanh(logit_lstm + logit_prev + logit_ctx)
    if options['use_dropout']:
        logit = dropout_layer(logit, use_noise, trng)
    logit = get_layer('ff')[1](tparams,
                               logit,
                               options,
                               prefix='ff_logit',
                               activ=None)
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0] * logit_shp[1],
                                               logit_shp[2]]))

    # cost
    y_flat = y.flatten()
    y_flat_idx = tensor.arange(y_flat.shape[0]) * options['n_words'] + y_flat
    cost = -tensor.log(probs.flatten()[y_flat_idx])
    cost = cost.reshape([y.shape[0], y.shape[1]])
    cost = (cost * y_mask).sum(0)

    return trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost


# build a sampler
def build_sampler(tparams, options, trng):
    x = tensor.matrix('x', dtype='int64')
    xr = x[::-1]
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # word embedding (source), forward and backward
    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word_src']])
    embr = tparams['Wemb'][xr.flatten()]
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word_src']])

    # encoder
    proj = get_layer(options['encoder'])[1](tparams,
                                            emb,
                                            options,
                                            prefix='encoder')
    projr = get_layer(options['encoder'])[1](tparams,
                                             embr,
                                             options,
                                             prefix='encoder_r')

    # concatenate forward and backward rnn hidden states
    ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim - 1)

    # get the input for decoder rnn initializer mlp
    ctx_mean = ctx.mean(0)
    # ctx_mean = concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)
    init_state = get_layer('ff')[1](tparams,
                                    ctx_mean,
                                    options,
                                    prefix='ff_state',
                                    activ=tensor.tanh)

    print('Building f_init...', end=' ')
    outs = [init_state, ctx]
    f_init = theano.function([x], outs, name='f_init', profile=False)
    print('Done')

    # x: 1 x 1
    y = tensor.vector('y_sampler', dtype='int64')
    init_state = tensor.matrix('init_state', dtype='float32')

    # if it's the first word, emb should be all zero and it is indicated by -1
    emb = tensor.switch(y[:, None] < 0,
                        tensor.alloc(0., 1, tparams['Wemb_dec'].shape[1]),
                        tparams['Wemb_dec'][y])

    # apply one step of conditional gru with attention
    proj = get_layer(options['decoder'])[1](tparams,
                                            emb,
                                            options,
                                            prefix='decoder',
                                            mask=None,
                                            context=ctx,
                                            one_step=True,
                                            init_state=init_state)
    # get the next hidden state
    next_state = proj[0]

    # get the weighted averages of context for this target word y
    ctxs = proj[1]

    logit_lstm = get_layer('ff')[1](tparams,
                                    next_state,
                                    options,
                                    prefix='ff_logit_lstm',
                                    activ=None)
    logit_prev = get_layer('ff')[1](tparams,
                                    emb,
                                    options,
                                    prefix='ff_logit_prev',
                                    activ=None)
    logit_ctx = get_layer('ff')[1](tparams,
                                   ctxs,
                                   options,
                                   prefix='ff_logit_ctx',
                                   activ=None)
    logit = tensor.tanh(logit_lstm + logit_prev + logit_ctx)
    logit = get_layer('ff')[1](tparams,
                               logit,
                               options,
                               prefix='ff_logit',
                               activ=None)

    # compute the softmax probability
    next_probs = tensor.nnet.softmax(logit)

    # sample from softmax distribution to get the sample
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # compile a function to do the whole thing above, next word probability,
    # sampled word for the next target, next hidden state to be used
    print('Building f_next..', end=' ')
    inps = [y, ctx, init_state]
    outs = [next_probs, next_sample, next_state]
    f_next = theano.function(inps, outs, name='f_next', profile=False)
    print('Done')

    return f_init, f_next


# generate sample, either with stochastic sampling or beam search. Note that,
# this function iteratively calls f_init and f_next functions.
def gen_sample(tparams,
               f_init,
               f_next,
               x,
               options,
               trng=None,
               k=1,
               maxlen=30,
               stochastic=True,
               argmax=False):

    # k is the beam size we have
    if k > 1:
        assert not stochastic, \
            'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []

    # get initial state of decoder rnn and encoder context
    ret = f_init(x)
    next_state, ctx0 = ret[0], ret[1]
    next_w = -1 * numpy.ones((1, )).astype('int64')  # bos indicator

    for ii in xrange(maxlen):
        ctx = numpy.tile(ctx0, [live_k, 1])
        inps = [next_w, ctx, next_state]
        ret = f_next(*inps)
        next_p, next_w, next_state = ret[0], ret[1], ret[2]

        if stochastic:
            if argmax:
                nw = next_p[0].argmax()
            else:
                nw = next_w[0]
            sample.append(nw)
            sample_score += next_p[0, nw]
            if nw == 0:
                break
        else:
            cand_scores = hyp_scores[:, None] - numpy.log(next_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k - dead_k)]

            voc_size = next_p.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k - dead_k).astype('float32')
            new_hyp_states = []

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti] + [wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append(copy.copy(next_state[ti]))

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = numpy.array(hyp_states)

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

    return sample, sample_score


# calculate the log probablities on a given corpus using translation model
def pred_probs(f_log_probs, options, stream, verbose=True):
    probs = []

    n_done = 0

    for x, x_mask, y, y_mask in stream.get_epoch_iterator():
        n_done += len(x)

        x, x_mask, y, y_mask = x.T, x_mask.T, y.T, y_mask.T

        pprobs = f_log_probs(x, x_mask, y, y_mask)
        for pp in pprobs:
            probs.append(pp)

        if numpy.isnan(numpy.mean(probs)):
            ipdb.set_trace()

        if verbose:
            print('%d samples computed' % (n_done), file=sys.stderr)

    return numpy.array(probs)
