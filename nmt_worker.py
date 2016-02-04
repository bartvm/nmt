'''
Build a neural machine translation model with soft attention
'''

import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import config

import six
from six.moves import cPickle as pkl
from six.moves import xrange

import cPickle as pkl
import ipdb
import numpy
import copy

import os
import warnings
import sys
import argparse
import time

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'platoon'))

from platoon.channel import Worker
from platoon.param_sync import EASGD


from collections import OrderedDict
from data_iterator import TextIterator
from utils import *
from optimizers import *
from layers import *

import settings
profile = settings.profile

def prepare_data(seqs_x, seqs_y, maxlen=None):
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]
        
    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y
	
        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_y = numpy.max(lengths_y) + 1

    x = numpy.zeros((maxlen_x, n_samples)).astype('int64')
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx]+1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.
   
    return x, x_mask, y, y_mask

# initialize all parameters
def init_params(options):
    params = OrderedDict()

    # embedding
    params['Wemb'] = norm_weight(options['n_words_src'],
                                 options['dim_word_src'])
    params['Wemb_dec'] = norm_weight(options['n_words'],
                                     options['dim_word_trg'])


    # MLP for unknown words embedding! Takes 
    # and outputs embedding for unknown words!
    params = get_layer('ff')[0](options, 
                                params,
                                prefix='ff_embedding',
                                nin=options['ctx_len_emb'] * options['dim_word_src'],
                                nout=options['dim_word_src'])

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

def build_model(tparams, options):
    opt_ret = dict()

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')
    y = tensor.matrix('y', dtype='int64')
    y_mask = tensor.matrix('y_mask', dtype='float32')

    # description num_of_unk_words * context_size
    unk_ctx = tensor.matrix('unk_ctx', dtype='int64') 

    #number of unknown words in a batch
    n_ctx = unk_ctx.shape[0]
    # get word embedding for the context
    unk_ctx_wrd_emb = tparams['Wemb'][unk_ctx.flatten()]
    # reshape the word embedding.
    unk_ctx_wrd_emb = unk_ctx_wrd_emb.reshape([n_ctx, 
                                               options['ctx_len_emb']*options['dim_word_src']])
    # given the context embedding, 
    # output the embeddings for 
    # unknown words.
    unk_ctx_emb = get_layer('ff')[1](tparams, 
                                     unk_ctx_wrd_emb, 
                                     options,
				     prefix='ff_embedding', 
                                     activ='tanh')
    

    # for the backward rnn, we just need to invert x and x_mask
    xr = x[::-1]
    xr_mask = x_mask[::-1]
    n_timesteps = x.shape[0]
    n_timesteps_trg = y.shape[0]
    n_samples = x.shape[1]

    # word embedding for forward rnn (source) 
    x_temp = x.flatten();    
    ones_vec = T.switch(T.eq(x_temp, 1), 1, 0)
    cum_sum = T.extra_ops.cumsum(ones_vec)
    cmb_x_vector = T.switch(T.eq(x_temp, 1), cum_sum + options['n_words_src'] - 1,  x_temp)
    c_word_embedding = concatenate([tparams['Wemb'], unk_ctx_emb], axis=0) 
    emb = c_word_embedding[cmb_x_vector[:]];
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word_src']])
    proj = get_layer(options['encoder'])[1](tparams,
                                            emb,
                                            options,
                                            prefix='encoder',
                                            mask=x_mask)
    # word embedding for backward rnn (source)
    cmb_x_vectorr = cmb_x_vector[::-1]
    embr = c_word_embedding[cmb_x_vectorr[:]];
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
                                    activ='tanh')

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
                                    activ='linear')
    logit_prev = get_layer('ff')[1](tparams,
                                    emb,
                                    options,
                                    prefix='ff_logit_prev',
                                    activ='linear')
    logit_ctx = get_layer('ff')[1](tparams,
                                   ctxs,
                                   options,
                                   prefix='ff_logit_ctx',
                                   activ='linear')
    logit = tensor.tanh(logit_lstm + logit_prev + logit_ctx)
    if options['use_dropout']:
        logit = dropout_layer(logit, use_noise, trng)
    logit = get_layer('ff')[1](tparams,
                               logit,
                               options,
                               prefix='ff_logit',
                               activ='linear')
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0] * logit_shp[1],
                                               logit_shp[2]]))

    # cost
    y_flat = y.flatten()
    y_flat_idx = tensor.arange(y_flat.shape[0]) * options['n_words'] + y_flat
    cost = -tensor.log(probs.flatten()[y_flat_idx])
    cost = cost.reshape([y.shape[0], y.shape[1]])
    cost = (cost * y_mask).sum(0)

    return trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost, unk_ctx

# build a sampler
def build_sampler(tparams, options, trng):
    x = tensor.matrix('x', dtype='int64')
    unk_ctx = tensor.matrix('unk_ctx', dtype='int64') 

    xr = x[::-1]
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]
    n_ctx = unk_ctx.shape[0] 

    unk_ctx_wrd_emb = tparams['Wemb'][unk_ctx.flatten()]
    unk_ctx_wrd_emb = unk_ctx_wrd_emb.reshape([n_ctx, 
                                               options['ctx_len_emb']*options['dim_word_src']])
    unk_ctx_emb = get_layer('ff')[1](tparams, 
                                     unk_ctx_wrd_emb, 
                                     options,
				     prefix='ff_embedding',
			             activ='tanh')
	
    # word embedding (source), forward and backward
    x_temp = x.flatten();    
    ones_vec = T.switch(T.eq(x_temp, 1), 1, 0)
    cum_sum = T.extra_ops.cumsum(ones_vec)
    cmb_x_vector = T.switch(T.eq(x_temp, 1), cum_sum + options['n_words_src'] - 1,  x_temp)
    c_word_embedding = concatenate([tparams['Wemb'], unk_ctx_emb], axis=0) 
    emb = c_word_embedding[cmb_x_vector[:]];
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word_src']])
    
    cmb_x_vectorr = cmb_x_vector[::-1]
    embr = c_word_embedding[cmb_x_vectorr[:]];
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
                                    activ='tanh')

    print 'Building f_init...'
    outs = [init_state, ctx]
    f_init = theano.function([x, unk_ctx], outs, name='f_init', profile=profile)
    print 'Done'

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
                                    activ='linear')
    logit_prev = get_layer('ff')[1](tparams,
                                    emb,
                                    options,
                                    prefix='ff_logit_prev',
                                    activ='linear')
    logit_ctx = get_layer('ff')[1](tparams,
                                   ctxs,
                                   options,
                                   prefix='ff_logit_ctx',
                                   activ='linear')
    logit = tensor.tanh(logit_lstm + logit_prev + logit_ctx)
    logit = get_layer('ff')[1](tparams,
                               logit,
                               options,
                               prefix='ff_logit',
                               activ='linear')

    # compute the softmax probability
    next_probs = tensor.nnet.softmax(logit)

    # sample from softmax distribution to get the sample
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # compile a function to do the whole thing above, next word probability,
    # sampled word for the next target, next hidden state to be used
    print 'Building f_next..' 
    inps = [y, ctx, init_state]
    outs = [next_probs, next_sample, next_state]
    f_next = theano.function(inps, outs, name='f_next', profile=profile)
    print 'Done'

    return f_init, f_next

    
def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

# generate sample, either with stochastic sampling or beam search. Note that,
# this function iteratively calls f_init and f_next functions.
def gen_sample(tparams,
               ctx_len_emb=5,
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
    unk_ctx = get_ctx_matrix(x, ctx_len_emb)
    ret = f_init(x, unk_ctx)
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
def pred_probs(f_log_probs, prepare_data, ctx_len_emb ,options, iterator, verbose=True):
    probs = []
    n_done = 0

    for x, y in iterator:
        n_done += len(x)
        x, x_mask, y, y_mask = prepare_data(x, y)
        unk_ctx = get_ctx_matrix(x, ctx_len_emb)
        pprobs = f_log_probs(x, x_mask, y, y_mask, unk_ctx)
        for pp in pprobs:
            probs.append(pp)

        if numpy.isnan(numpy.mean(probs)):
            ipdb.set_trace()

        if verbose:
            print >> sys.stderr, '%d samples computed' % (n_done)

    return numpy.array(probs)

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def train(dim_word_src=100,  # source word vector dimensionality
          dim_word_trg=100,  # target word vector dimensionality
          dim=1000,  # the number of LSTM units
          encoder='gru',
          decoder='gru_cond',
          patience=10,  # early stopping patience
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          decay_c=0.,  # L2 regularization penalty
          alpha_c=0.,  # alignment regularization
          clip_c=-1.,  # gradient clipping threshold
          lrate=0.01,  # learning rate
          n_words_src=3000,  # source vocabulary size
          n_words=3000,  # target vocabulary size
          maxlen=15,  # maximum length of the description
          optimizer='rmsprop',
          batch_size=64,
          valid_batch_size=64,
	  ctx_len_emb = 5,
          train_len = 1,
          saveto='model.npz',
          validFreq=1000,
          saveFreq=1000,   # save the parameters after every saveFreq updates
          sampleFreq=100,   # generate some samples after every sampleFreq
          datasets=[
              '/u/goyalani/dl4mt-material/europarl-v7.fr-en.en.tok',
              '/u/goyalani/dl4mt-material/europarl-v7.fr-en.fr.tok'],
          valid_datasets=['/u/goyalani/dl4mt-material/newstest2011.en.tok',
                          '/u/goyalani/dl4mt-material/newstest2011.fr.tok'],
          dictionaries=[
              '/u/goyalani/dl4mt-material/europarl-v7.fr-en.en.tok.pkl',
              '/u/goyalani/dl4mt-material/europarl-v7.fr-en.fr.tok.pkl'],  
	
          use_dropout=False,
	  overwrite=False,
	  valid_sync=False,
          reload_=False):

    worker = Worker(control_port=5567)
    

    # Model options
    model_options = locals().copy()

    # load dictionaries and invert them
    worddicts = [None] * len(dictionaries)
    worddicts_r = [None] * len(dictionaries)
    for ii, dd in enumerate(dictionaries):
        with open(dd, 'rb') as f:
            worddicts[ii] = pkl.load(f)
        worddicts_r[ii] = dict()
        for kk, vv in worddicts[ii].iteritems():
            worddicts_r[ii][vv] = kk

    # reload options
    if reload_ and os.path.exists(saveto):
        with open('%s.pkl' % saveto, 'rb') as f:
            models_options = pkl.load(f)

    print 'Loading data'
    train = TextIterator(datasets[0],
                             datasets[1],
                             dictionaries[0],
                             dictionaries[1],
                             n_words_source=n_words_src,
                             n_words_target=n_words,
                             batch_size=batch_size,
                             maxlen=maxlen)
    valid = TextIterator(valid_datasets[0],
                             valid_datasets[1],
                             dictionaries[0],
                             dictionaries[1],
                             n_words_source=n_words_src,
                             n_words_target=n_words,
                             batch_size=valid_batch_size,
                             maxlen=maxlen)

    print 'Building model'
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        params = load_params(saveto, params)

    tparams = init_tparams(params)
    worker.init_shared_params(tparams.values(), param_sync_rule=EASGD(0.5))
    print 'Params init done'

    # use_noise is for dropout
    trng, use_noise, \
        x, x_mask, y, y_mask, \
        opt_ret, \
	cost, \
        unk_ctx = \
        build_model(tparams, model_options)
    inps = [x, x_mask, y, y_mask, unk_ctx]

    print 'Buliding sampler'
    f_init, f_next  = build_sampler(tparams, model_options, trng)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile)
    print 'Done'

    cost = cost.mean()

    # apply L2 regularization on weights
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # regularize the alpha weights
    if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * (
            (tensor.cast(y_mask.sum(0)//x_mask.sum(0), 'float32')[:, None] -
             opt_ret['dec_alphas'].sum(0))**2).sum(1).mean()
        cost += alpha_reg

    # after all regularizers - compile the computational graph for cost
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=profile)
    print 'Done'

    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print 'Done'

    # apply gradient clipping here
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        grads = new_grads

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)
    print 'Done'

    print 'Optimization'
    best_p = None

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size
    if sampleFreq == -1:
        sampleFreq = len(train[0])/batch_size


    # Training data iterator!
    def train_iter():
        while True:
	    try:
 		if train.end_of_data == True:
                     train.reset();
                     train.end_of_data = False
	    	[x_data, y_data] =  train.next()
            	kf = get_minibatches_idx(len(x_data), batch_size, shuffle=True)
            	for _, train_index in kf:
			y = [y_data[t] for t in train_index]
			x = [x_data[t] for t in train_index]
                	x, x_mask, y, y_mask = prepare_data(x, y, maxlen=maxlen)
                	yield x, x_mask, y, y_mask
	    except:
			train.end_of_data = False
            		train.reset()
			break;
			
				
    train_it = train_iter()

    # Making sure that the worker start training with the most recent params
    worker.copy_to_local()	

    while True:
        n_samples = 0
        step = worker.send_req('next')
        print step
	if step == 'train':
            	use_noise.set_value(1.)
		for i in xrange(train_len):
			x, x_mask, y, y_mask = next(train_it)
			if x is None:
                		print 'Minibatch with zero sample under length ', maxlen
                		continue
			if len(x) == 0:
                		print 'No input is present ', x
                		continue
		
                        # For this particular, mini-batch, it 
                        # outputs the context matrix for all the 
                        # unknown words.	
                        unk_ctx = get_ctx_matrix(x, ctx_len_emb)
			
                        if unk_ctx is None:
        			print 'No unknown word is present ', unk_ctx
                		continue
			if len(unk_ctx) == 0:
                		print 'No unknown word is present ', unk_ctx
                		continue
		
			# compute cost, grads and copy grads to shared variables    
			cost = f_grad_shared(x, x_mask, y, y_mask, unk_ctx)
	        
			# do the update on parameters
                	f_update(lrate)
		
		print 'Train cost:', cost
   
		step = worker.send_req(dict(done=train_len))
            	print "Syncing with global params"
            	worker.sync_params(synchronous=True) 
		
	if step == 'valid':
            if valid_sync:
                worker.copy_to_local()
            use_noise.set_value(0.)
	    valid_errs = pred_probs(f_log_probs, 
                                    prepare_data, 
                                    ctx_len_emb,
                                    model_options, 
                                    valid)
	    valid_err = valid_errs.mean()
            res = worker.send_req(dict(test_err=float(valid_err),
                                       valid_err=float(valid_err)))

            if res == 'best':
                best_p = unzip(tparams)

            print ('Valid ', valid_err,
                   'Test ', valid_err)
            if valid_sync:
                worker.copy_to_local()

        if step == 'stop':
            break

    # Release all shared ressources.
    worker.close()
	
    print 'Saving...'

    if best_p is not None:
    	params = best_p
    else:
        params = unzip(tparams)
	
    use_noise.set_value(0.)

    if saveto:
	numpy.savez(saveto, **best_p)
	print 'model saved'	

    params = copy.copy(best_p)
    numpy.savez(saveto, zipped_params=best_p, **params)

def main(job_id, params):
    print params
    basedir = params['basedir'][0];
    valid_err = train(saveto=params['model'][0],
          reload_= params['reload'][0],
          dim = params['dim'][0],
          n_words = params['n-words'][0],
          n_words_src = params['n-words'][0],
          decay_c = params['decay-c'][0],
          clip_c = params['clip-c'][0],
          lrate = params['learning-rate'][0],
          optimizer = params['optimizer'][0],
          maxlen = 15,
          batch_size = 64,
          valid_batch_size = 64,
          datasets = ['%s/europarl-v7.fr-en.fr.tok'%basedir,
          '%s/europarl-v7.fr-en.en.tok'%basedir],
          valid_datasets=['%s/newstest2011.fr.tok'%basedir,
          '%s/newstest2011.en.tok'%basedir],
          dictionaries=['%s/europarl-v7.fr-en.fr.tok.pkl'%basedir,
          '%s/europarl-v7.fr-en.en.tok.pkl'%basedir],
          validFreq=500000,
          dispFreq=1,
          saveFreq=100,
          sampleFreq=50,
          use_dropout=params['use-dropout'][0])
    return


if __name__ == '__main__':
    basedir = '/data/lisatmp3/nmt/data'
    mode=sys.argv[1]
    main(0, {
        'mode' : [mode],
        'basedir' : ['/data/lisatmp3/nmt/data'],
        'model': ['%s/models/model_attention.npz'%basedir],
        'dim': [124],
        'n-words': [3000],
        'optimizer': ['adadelta'],
        'decay-c': [0.],
        'clip-c': [1.],
        'use-dropout': [False],
        'learning-rate': [0.001],
        'reload': [False]})

