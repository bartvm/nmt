import theano
from theano import tensor
import numpy
from utils import uniform_weight, ortho_weight

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'gru_cond': ('param_init_gru_cond', 'gru_cond_layer'), }


def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


def zero_vector(length):
    return numpy.zeros((length, )).astype('float32')


def tanh(x):
    return tensor.tanh(x)


def sigmoid(x):
    return tensor.nnet.sigmoid(x)


def linear(x):
    return x


# utility function to slice a tensor
def _slice(_x, n, dim):
    if _x.ndim == 3:
        return _x[:, :, n * dim:(n + 1) * dim]
    return _x[:, n * dim:(n + 1) * dim]


def _gru(mask, x_t2gates, x_t2prpsl, h_tm1, U, Ux, activ='tanh'):

    dim = U.shape[0]    # dimension of hidden states

    # concatenated activations of the gates in a GRU
    activ_gates = sigmoid(x_t2gates + tensor.dot(h_tm1, U))

    # reset and update gates
    reset_gate = _slice(activ_gates, 0, dim)
    update_gate = _slice(activ_gates, 1, dim)

    # compute the hidden state proposal
    h_prpsl = eval(activ)(x_t2prpsl + reset_gate * tensor.dot(h_tm1, Ux))

    # leaky integrate and obtain next hidden state
    h_t = update_gate * h_tm1 + (1. - update_gate) * h_prpsl

    # if this time step is not valid, discard the current hidden states
    # obtained above and copy the previous hidden states to the current ones.
    h_t = mask[:, None] * h_t + (1. - mask)[:, None] * h_tm1

    return h_t


def _compute_alignment(h_tm1,       # s_{i-1}
                       prj_annot,   # proj annotations: U_a * h_j for all j
                       Wd_att, U_att, c_att,
                       context_mask=None):

    # W_a * s_{i-1}
    prj_h_tm1 = tensor.dot(h_tm1, Wd_att)

    # tanh(W_a * s_{i-1} + U_a * h_j) for all j
    nonlin_proj = tensor.tanh(prj_h_tm1[None, :, :] + prj_annot)

    # v_a^{T} * tanh(.) + bias
    alpha = tensor.dot(nonlin_proj, U_att) + c_att
    alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
    alpha = tensor.exp(alpha)
    if context_mask:
        alpha = alpha * context_mask
    alpha = alpha / alpha.sum(0, keepdims=True)

    return alpha


# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options,
                       param,
                       prefix='ff',
                       nin=None,
                       nout=None,
                       ortho=True):

    param[prefix + '_W'] = uniform_weight(nin, nout)
    param[prefix + '_b'] = zero_vector(nout)

    return param


def fflayer(tparams,
            state_below,
            options,
            prefix='rconv',
            activ='lambda x: tensor.tanh(x)',
            **kwargs):
    return eval(activ)(tensor.dot(state_below, tparams[prefix + '_W']) +
                       tparams[prefix + '_b'])


# GRU layer
def param_init_gru(options, param, prefix='gru', nin=None, dim=None):

    param[prefix + '_W'] = numpy.concatenate(
        [
            uniform_weight(nin, dim), uniform_weight(nin, dim)
        ],
        axis=1)
    param[prefix + '_U'] = numpy.concatenate(
        [
            ortho_weight(dim), ortho_weight(dim)
        ],
        axis=1)
    param[prefix + '_b'] = zero_vector(2 * dim)

    param[prefix + '_Wx'] = uniform_weight(nin, dim)
    param[prefix + '_Ux'] = ortho_weight(dim)
    param[prefix + '_bx'] = zero_vector(dim)

    return param


def gru_layer(tparams,
              state_below,
              options,
              prefix='gru',
              mask=None,
              **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 4:
        n_samples = state_below.shape[2]
    elif state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[prefix + '_Ux'].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # state_below is the input word embeddings
    state_below_ = (tensor.dot(state_below, tparams[prefix + '_W']) +
                    tparams[prefix + '_b'])
    state_belowx = (tensor.dot(state_below, tparams[prefix + '_Wx']) +
                    tparams[prefix + '_bx'])

    # prepare scan arguments
    seqs = [mask, state_below_, state_belowx]

    if state_below.ndim == 4:
        init_states = [tensor.alloc(0., state_below.shape[1], n_samples, dim)]
    else:
        init_states = [tensor.alloc(0., n_samples, dim)]

    _step = _gru
    shared_vars = [tparams[prefix + '_U'], tparams[prefix + '_Ux']]

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=init_states,
                                non_sequences=shared_vars,
                                name=prefix + '_layers',
                                n_steps=nsteps,
                                profile=False,
                                strict=True)
    rval = [rval]
    return rval


# Conditional GRU layer with Attention
def param_init_gru_cond(options,
                        param,
                        prefix='gru_cond',
                        nin=None,
                        dim=None,
                        dimctx=None,
                        nin_nonlin=None,
                        dim_nonlin=None):
    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim

    param = param_init_gru(options, param, prefix=prefix, nin=nin, dim=dim)

    param[prefix + '_U_nl'] = numpy.concatenate(
        [
            ortho_weight(dim_nonlin), ortho_weight(dim_nonlin)
        ],
        axis=1)
    param[prefix + '_b_nl'] = zero_vector(2 * dim_nonlin)

    param[prefix + '_Ux_nl'] = ortho_weight(dim_nonlin)
    param[prefix + '_bx_nl'] = zero_vector(dim_nonlin)

    # context to LSTM
    param[prefix + '_Wc'] = uniform_weight(dimctx, dim * 2)
    param[prefix + '_Wcx'] = uniform_weight(dimctx, dim)

    # attention: combined -> hidden
    param[prefix + '_W_comb_att'] = uniform_weight(dim, dimctx)

    # attention: context -> hidden
    param[prefix + '_Wc_att'] = uniform_weight(dimctx, dimctx)

    # attention: hidden bias
    param[prefix + '_b_att'] = zero_vector(dimctx)

    # attention:
    param[prefix + '_U_att'] = uniform_weight(dimctx, 1)
    param[prefix + '_c_att'] = zero_vector(1)

    return param


def gru_cond_layer(tparams,
                   state_below,
                   options,
                   prefix='gru',
                   mask=None,
                   context=None,
                   one_step=False,
                   init_memory=None,
                   init_state=None,
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
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[prefix + '_Wcx'].shape[1]

    # initial/previous state
    if init_state is None:
        if state_below.ndim == 4:
            init_state = [tensor.alloc(0., state_below.shape[1],
                                       n_samples, dim)]
        else:
            init_state = [tensor.alloc(0., n_samples, dim)]

    # projected context
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = (tensor.dot(context, tparams[prefix + '_Wc_att']) +
             tparams[prefix + '_b_att'])

    # projected x
    state_belowx = (tensor.dot(state_below, tparams[prefix + '_Wx']) +
                    tparams[prefix + '_bx'])
    state_below_ = (tensor.dot(state_below, tparams[prefix + '_W']) +
                    tparams[prefix + '_b'])

    def _step_slice(m_, x_, xx_, h_, ctx_, alpha_, pctx_, cc_, U, Wc,
                    W_comb_att, U_att, c_att, Ux, Wcx, U_nl, Ux_nl, b_nl,
                    bx_nl):

        h1 = _gru(m_, x_, xx_, h_, U, Ux)

        # attention
        alpha = _compute_alignment(h1, pctx_,
                                   W_comb_att, U_att, c_att,
                                   context_mask=context_mask)

        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context

        new_x_ = tensor.dot(ctx_, Wc) + b_nl
        new_xx_ = tensor.dot(ctx_, Wcx) + bx_nl

        h2 = _gru(m_, new_x_, new_xx_, h1, U_nl, Ux_nl)

        return h2, ctx_, alpha.T  # pstate_, preact, preactx, r, u

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    shared_vars = [tparams[prefix + '_U'], tparams[prefix + '_Wc'],
                   tparams[prefix + '_W_comb_att'], tparams[prefix + '_U_att'],
                   tparams[prefix + '_c_att'], tparams[prefix + '_Ux'],
                   tparams[prefix + '_Wcx'], tparams[prefix + '_U_nl'],
                   tparams[prefix + '_Ux_nl'], tparams[prefix + '_b_nl'],
                   tparams[prefix + '_bx_nl']]

    if one_step:
        rval = _step(*(
            seqs + [init_state, None, None, pctx_, context] + shared_vars))
    else:
        rval, updates = theano.scan(
            _step,
            sequences=seqs,
            outputs_info=[
                init_state,
                tensor.alloc(0., n_samples, context.shape[2]),
                tensor.alloc(0., n_samples, context.shape[0])
            ],
            non_sequences=[pctx_, context] + shared_vars,
            name=prefix + '_layers',
            n_steps=nsteps,
            profile=False,
            strict=True)
    return rval
