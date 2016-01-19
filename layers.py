import theano.tensor as T
import numpy
from utils import *

import settings
profile = settings.profile

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'gru_cond': ('param_init_gru_cond', 'gru_cond_layer'),
          }


def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))

def zero_vector(length):
    return numpy.zeros((length,)).astype('float32')

def tanh(x):
    return tensor.tanh(x)

def linear(x):
    return x

# utility function to slice a tensor
def _slice(_x, n, dim):
    if _x.ndim == 3:
        return _x[:, :, n*dim:(n+1)*dim]
    return _x[:, n*dim:(n+1)*dim]

def _gru(m_, x_, xx_, h_, U, Ux, bg=None, bi=None):

    dim = U.shape[0]

    preact = x_ + T.dot(h_, U)
    if not (bg is None):
        preact += bg

    # reset and update gates
    r = T.nnet.sigmoid(_slice(preact, 0, dim))
    u = T.nnet.sigmoid(_slice(preact, 1, dim))

    # compute the hidden state proposal
    preact2 = xx_ + r * T.dot(h_, Ux)
    if not (bi is None):
        preact2 += bi

    h = T.tanh(preact2)

    # leaky integrate and obtain next hidden state
    h = u * h_ + (1. - u) * h
    h = m_[:, None] * h + (1. - m_)[:, None] * h_

    return h

# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, prm, pre='ff', nin=None, nout=None,
                       ortho=True):

    prm[pre+'_W'] = uniform_weight(nin, nout)
    prm[pre+'_b'] = zero_vector(nout)

    return prm


def fflayer(tp, state_below, options, pre='rconv',
            activ='lambda x: T.tanh(x)', **kwargs):
    return eval(activ)(
        T.dot(state_below, tp[pre+'_W']) + tp[pre+'_b'])


# GRU layer
def param_init_gru(options, prm, pre='gru', nin=None, dim=None):

    prm[pre+'_W'] = numpy.concatenate([
                    uniform_weight(nin, dim),
                    uniform_weight(nin, dim)
                    ], axis=1)
    prm[pre+'_U'] = numpy.concatenate([
                    ortho_weight(dim),
                    ortho_weight(dim)
                    ], axis=1)
    prm[pre+'_b'] = zero_vector(2*dim)


    prm[pre+'_Wx'] = uniform_weight(nin, dim)
    prm[pre+'_Ux'] = ortho_weight(dim)
    prm[pre+'_bx'] = zero_vector(dim)

    return prm


def gru_layer(tp, state_below, options, pre='gru', mask=None,
              **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 4:
        n_samples = state_below.shape[2]
    elif state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tp[pre+'_Ux'].shape[1]

    if mask is None:
        mask = T.alloc(1., state_below.shape[0], 1)

    # state_below is the input word embeddings
    state_below_ = T.dot(state_below, tp[pre+'_W']) + tp[pre+'_b']
    state_belowx = T.dot(state_below, tp[pre+'_Wx']) + tp[pre+'_bx']

    # prepare scan arguments
    seqs = [mask, state_below_, state_belowx]
    
    if state_below.ndim == 4:
        init_states = [T.alloc(0., state_below.shape[1], n_samples, dim)]
    else:
        init_states = [T.alloc(0., n_samples, dim)]

    _step = _gru
    shared_vars = [tp[pre+'_U'],tp[pre+'_Ux']]

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=init_states,
                                non_sequences=shared_vars,
                                name=pre+'_layers',
                                n_steps=nsteps,
                                profile=profile,
                                strict=True)
    rval = [rval]
    return rval


# Conditional GRU layer with Attention
def param_init_gru_cond(options, prm, pre='gru_cond',
                        nin=None, dim=None, dimctx=None,
                        nin_nonlin=None, dim_nonlin=None):
    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim


    prm = param_init_gru(options, prm, pre=pre, nin=nin, dim=dim)

    prm[pre+'_U_nl'] = numpy.concatenate([
                        ortho_weight(dim_nonlin),
                        ortho_weight(dim_nonlin)
                        ], axis=1)
    prm[pre+'_b_nl'] = zero_vector(2 * dim_nonlin)

    prm[pre+'_Ux_nl'] = ortho_weight(dim_nonlin)
    prm[pre+'_bx_nl'] = zero_vector(dim_nonlin)

    # context to LSTM
    prm[pre+'_Wc'] = uniform_weight(dimctx, dim*2)
    prm[pre+'_Wcx'] = uniform_weight(dimctx, dim)

    # attention: combined -> hidden
    prm[pre+'_W_comb_att'] = uniform_weight(dim, dimctx)

    # attention: context -> hidden
    prm[pre+'_Wc_att'] = uniform_weight(dimctx,dimctx)

    # attention: hidden bias
    prm[pre+'_b_att'] = zero_vector(dimctx)

    # attention:
    prm[pre+'_U_att'] = uniform_weight(dimctx, 1)
    prm[pre+'_c_att'] = zero_vector(1)

    return prm


def gru_cond_layer(tp, state_below, options, pre='gru',
                   mask=None, context=None, one_step=False,
                   init_memory=None, init_state=None,
                   context_mask=None,
                   **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 4:
        n_samples = state_below.shape[2]
    elif state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = T.alloc(1., state_below.shape[0], 1)

    dim = tp[pre+'_Wcx'].shape[1]

    # initial/previous state
    if init_state is None:
        if state_below.ndim == 4:
            init_state = [T.alloc(0., state_below.shape[1], n_samples, dim)]
        else:
            init_state = [T.alloc(0., n_samples, dim)]

    # projected context
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = T.dot(context, tp[pre+'_Wc_att']) + tp[pre+'_b_att']

    # projected x
    state_belowx = T.dot(state_below, tp[pre+'_Wx']) + tp[pre+'_bx']
    state_below_ = T.dot(state_below, tp[pre+'_W']) + tp[pre+'_b']

    def _step_slice(m_, x_, xx_, h_, ctx_, alpha_, pctx_, cc_,
                    U, Wc, W_comb_att, U_att, c_att, Ux, Wcx,
                    U_nl, Ux_nl, b_nl, bx_nl):

        h1 = _gru(m_, x_, xx_, h_, U, Ux)

        # attention
        pstate_ = T.dot(h1, W_comb_att)
        pctx__ = pctx_ + pstate_[None, :, :]
        #pctx__ += xc_
        pctx__ = T.tanh(pctx__)
        alpha = T.dot(pctx__, U_att)+c_att
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = T.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context

        new_x_ = T.dot(ctx_, Wc)
        new_xx_ = T.dot(ctx_, Wcx)

        h2 = _gru(m_, new_x_, new_xx_, h1, U_nl, Ux_nl, bg=b_nl, bi=bx_nl)

        return h2, ctx_, alpha.T  # pstate_, preact, preactx, r, u

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    shared_vars = [tp[pre+'_U'],
                   tp[pre+'_Wc'],
                   tp[pre+'_W_comb_att'],
                   tp[pre+'_U_att'],
                   tp[pre+'_c_att'],
                   tp[pre+'_Ux'],
                   tp[pre+'_Wcx'],
                   tp[pre+'_U_nl'],
                   tp[pre+'_Ux_nl'],
                   tp[pre+'_b_nl'],
                   tp[pre+'_bx_nl']]

    if one_step:
        rval = _step(*(seqs + [init_state, None, None, pctx_, context] + shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  T.alloc(0., n_samples, context.shape[2]),
                                                  T.alloc(0., n_samples, context.shape[0])],
                                    non_sequences=[pctx_, context]+shared_vars,
                                    name=pre+'_layers',
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    return rval
