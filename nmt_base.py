'''
Build a neural machine translation model with soft attention
'''
from __future__ import division

import io
import subprocess
import logging
import os
from collections import OrderedDict

import numpy
import theano
from six.moves import xrange
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams

from utils import (unzip, RepeatedTimer,
                   dropout_layer, norm_weight, concatenate,
                   prepare_character_tensor, beam_search)
from layers import get_layer

LOGGER = logging.getLogger(__name__)


def validation(tparams, process_queue, translator_cmd, evaluator_cmd,
               model_filename, trans_valid_src):

    # We need to make sure that the model remains unchanged during evaluation
    model = unzip(tparams)
    save_params(model, model_filename)

    # Translation runs on CPUs with BLAS
    env_THEANO_FLAGS = 'device=cpu,floatX=%s,optimizer=%s' % (
        theano.config.floatX,
        theano.config.optimizer)

    env = dict(os.environ, **{'OMP_NUM_THREADS': '1',
                              'THEANO_FLAGS': env_THEANO_FLAGS})

    trans_proc = subprocess.Popen(translator_cmd, env=env,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)

    # put the process into the queue for signal handling
    process_queue.put(trans_proc)

    output_msg, error_msg = trans_proc.communicate()

    process_queue.get()

    if trans_proc.returncode == 1:
        raise RuntimeError("%s\nFailed to translate sentences" % error_msg)

    try:
        with io.open(trans_valid_src, 'r') as trans_result_f:
            eval_proc = subprocess.Popen(evaluator_cmd,
                                         stdin=trans_result_f,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)

            process_queue.put(eval_proc)

            # sample output:
            # BLEU = 100.00, 100.0/100.0/100.0/100.0
            # extra part of the output is ignored.

            out_msg, err_msg = eval_proc.communicate()

            process_queue.get()

            if eval_proc.returncode == 1:

                os.remove(trans_valid_src)

                raise RuntimeError("%s\nFailed to evaluate translation" %
                                   error_msg)

            overall_bleu_score = float(out_msg.split(',')[0].split('=')[1])
            bleu_n = [float(score) for score in out_msg.split()[3].split('/')]
            bleu_1, bleu_2, bleu_3, bleu_4 = bleu_n

            evaluation_score = (overall_bleu_score,
                                bleu_1, bleu_2, bleu_3, bleu_4)

            os.remove(trans_valid_src)

    except IOError:
        # LOGGER.error('Translation cannot be found, so that BLEU set to 0')
        evaluation_score = (0., 0., 0., 0., 0.)

    return (model, evaluation_score)


def prepare_validation_timer(tparams,
                             process_queue,
                             model_filename,
                             model_option_filename,
                             eval_intv,
                             valid_ret_queue,
                             translator,
                             evaluator,
                             nproc,
                             beam_size,
                             src_char_vocab,
                             src_word_vocab,
                             trg_char_vocab,
                             trg_word_vocab,
                             valid_src,
                             valid_trg,
                             trans_valid_src):

    translator_cmd = [
        "python",
        translator,
        "-p", str(nproc),
        "-k", str(beam_size),
        "-n",
        "-u",
        model_filename,
        model_option_filename,
        src_char_vocab,
        src_word_vocab,
        trg_char_vocab,
        trg_word_vocab,
        valid_src,
        trans_valid_src,
    ]

    evaluator_cmd = [
        "perl",
        evaluator,
        valid_trg,
    ]

    args = (tparams, process_queue)
    kwargs = {'translator_cmd': translator_cmd,
              'evaluator_cmd': evaluator_cmd,
              'model_filename': model_filename,
              'trans_valid_src': trans_valid_src}

    return RepeatedTimer(eval_intv*60, validation, valid_ret_queue,
                         *args, **kwargs)


# initialize all parameters
def init_params(options):
    params = OrderedDict()

    src_inp_dim = options['dim_word_src']
    trg_inp_dim = options['dim_word_trg']
    trg_out_dim = options['dim_word_trg']
    if options['use_character']:
        src_inp_dim += options['char_hid']*2
        trg_inp_dim += options['char_hid']*2
        trg_out_dim += options['char_hid']

    if options['use_character']:
        """
            Define parameters for a character-level model

        """
        # character embedding
        params['Cemb'] = norm_weight(
            options['n_chars_src'],
            options['dim_char_src'],
            scale=1/numpy.sqrt(options['n_chars_src'])
        )
        params['Cemb_dec'] = norm_weight(
            options['n_chars_trg'],
            options['dim_char_trg'],
            scale=1/numpy.sqrt(options['n_chars_trg'])
        )

        # bidirectional RNN encoder for characters in a SOURCE word
        params = get_layer(options['encoder'])[0](
            options,
            params,
            prefix='char_enc_src',
            nin=options['dim_char_src'],
            dim=options['char_hid']
        )
        params = get_layer(options['encoder'])[0](
            options,
            params,
            prefix='char_enc_src_r',
            nin=options['dim_char_src'],
            dim=options['char_hid']
        )

        # bidirectional RNN encoder for characters in a TARGET word
        params = get_layer(options['encoder'])[0](
            options,
            params,
            prefix='char_enc_trg',
            nin=options['dim_char_trg'],
            dim=options['char_hid']
        )
        params = get_layer(options['encoder'])[0](
            options,
            params,
            prefix='char_enc_trg_r',
            nin=options['dim_char_trg'],
            dim=options['char_hid']
        )

        '''
        # non-linear mapping of character embeddings to word embeddings
        params = get_layer('ff')[0](
            options, params, prefix='char2word_src',
            nin=options['char_hid']*2,
            nout=options['dim_word_src']
        )
        params = get_layer('ff')[0](
            options, params, prefix='char2word_src_r',
            nin=options['char_hid']*2,
            nout=options['dim_word_src']
        )
        params = get_layer('ff')[0](
            options, params, prefix='char2word_trg',
            nin=options['char_hid']*2,
            nout=options['dim_word_trg']
        )

        # (soft) gate to select embeddings
        params = get_layer('ff')[0](
            options, params, prefix='word_gate_src',
            nin=options['dim_word_src'],
            nout=options['dim_word_src']
        )
        params = get_layer('ff')[0](
            options, params, prefix='word_gate_trg',
            nin=options['dim_word_trg'],
            nout=options['dim_word_trg']
        )
        '''
        params = get_layer('ff')[0](options,
                                    params,
                                    prefix='ff_char_state',
                                    nin=options['dim'],
                                    nout=options['char_hid'])

        # decoder for target characters
        # NOTE character decoder doesn't need alignment
        #  when computing hidden states in the current implementation
        params = get_layer('gru')[0](
            options,
            params,
            prefix='char_decoder',
            nin=options['dim_char_trg'],
            dim=options['char_hid']
        )
        params = get_layer('ff')[0](options,
                                    params,
                                    prefix='ff_char_logit_lstm',
                                    nin=options['char_hid'],
                                    nout=options['dim_char_trg'],
                                    ortho=False)
        params = get_layer('ff')[0](options,
                                    params,
                                    prefix='ff_char_logit_prev_c',
                                    nin=options['dim_char_trg'],
                                    nout=options['dim_char_trg'],
                                    ortho=False)
        params = get_layer('ff')[0](options,
                                    params,
                                    prefix='ff_char_logit_word',
                                    nin=src_inp_dim,
                                    nout=options['dim_char_trg'],
                                    ortho=False)
        params = get_layer('ff')[0](options,
                                    params,
                                    prefix='ff_char_logit_word_state',
                                    nin=options['dim'],
                                    nout=options['dim_char_trg'],
                                    ortho=False)
        params = get_layer('ff')[0](options,
                                    params,
                                    prefix='ff_char_logit',
                                    # nin=int(options['dim_char_trg']/2),
                                    nin=options['dim_char_trg'],
                                    nout=options['n_chars_trg'])

    """
        Define parameters for a word-level model

    """
    # word embedding
    params['Wemb'] = norm_weight(options['n_words_src'],
                                 options['dim_word_src'],
                                 scale=1/numpy.sqrt(options['n_words_src']))
    params['Wemb_dec'] = norm_weight(options['n_words_trg'],
                                     options['dim_word_trg'],
                                     scale=1/numpy.sqrt(
                                         options['n_words_trg']))

    # encoder: bidirectional RNN for source words
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix='word_encoder',
                                              nin=src_inp_dim,
                                              dim=options['dim'])
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix='word_encoder_r',
                                              nin=src_inp_dim,
                                              dim=options['dim'])
    ctxdim = 2 * options['dim']

    if options['init_decoder'] == 'adaptive':
        # init state weighting
        params = get_layer('ff')[0](options,
                                    params,
                                    prefix='ff_state_proj',
                                    nin=ctxdim,
                                    nout=ctxdim)
        params = get_layer('ff')[0](options,
                                    params,
                                    prefix='ff_context_proj',
                                    nin=ctxdim,
                                    nout=ctxdim)
        params['word_weight_score'] = norm_weight(ctxdim, 1)

    # init_state
    params = get_layer('ff')[0](options,
                                params,
                                prefix='ff_word_state',
                                nin=ctxdim,
                                nout=options['dim'])

    # decoder for target words
    params = get_layer(options['decoder'])[0](options,
                                              params,
                                              prefix='word_decoder',
                                              nin=trg_inp_dim,
                                              dim=options['dim'],
                                              dimctx=ctxdim)
    # readout
    params = get_layer('ff')[0](options,
                                params,
                                prefix='ff_word_logit_lstm',
                                nin=options['dim'],
                                nout=options['dim_word_trg'],
                                ortho=False)
    params = get_layer('ff')[0](options,
                                params,
                                prefix='ff_word_logit_prev',
                                nin=trg_inp_dim,
                                nout=options['dim_word_trg'],
                                ortho=False)
    params = get_layer('ff')[0](options,
                                params,
                                prefix='ff_word_logit_ctx',
                                nin=ctxdim,
                                nout=options['dim_word_trg'],
                                ortho=False)
    params = get_layer('ff')[0](options,
                                params,
                                prefix='ff_word_logit',
                                nin=options['dim_word_trg'],
                                nout=options['n_words_trg'])

    return params


# build a training model
def build_model(tparams, options):
    """ Build a computational graph for model training

    Parameters:
    -------
        tparams : dict
            Model parameters
        options : dict
            Model configurations

    Returns:
    -------
        trng : Randomstream in Theano

        use_noise : TheanoSharedVariable

        encoder_vars : list
            This return value contains TheanoVariables used to construct
            part of the computational graph, especially used in the `encoder`.

        decoder_vars : list
            This return value contains TheanoVariables used to construct
            part of the computational graph, especially used in the `decoder`.

        opt_ret : dict

        costs : list
            A list of costs at word-level or
            both word-level and character-level costs

    """
    opt_ret = dict()

    trng = MRG_RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')
    y = tensor.matrix('y', dtype='int64')
    y_mask = tensor.matrix('y_mask', dtype='float32')

    encoder_vars = [x, x_mask]  # collection of varaibles used in the encoder
    decoder_vars = [y, y_mask]  # used in the decoder

    # for the backward rnn, we just need to invert x and x_mask
    xr = x[::-1]    # reverse words
    xr_mask = x_mask[::-1]

    n_words_src = x.shape[0]
    n_words_trg = y.shape[0]
    n_samples = x.shape[1]

    # word embedding for forward rnn (source)
    wemb_src = tparams['Wemb'][x.flatten()]
    wemb_src = wemb_src.reshape([n_words_src, n_samples,
                                 options['dim_word_src']])

    # word embedding for backward rnn (source)
    wembr_src = tparams['Wemb'][xr.flatten()]
    wembr_src = wembr_src.reshape([n_words_src, n_samples,
                                   options['dim_word_src']])

    if options['use_character']:
        xc = tensor.matrix('xc', dtype='int64')
        xc_mask = tensor.matrix('xc_mask', dtype='float32')
        yc_in = tensor.matrix('yc_in', dtype='int64')
        yc_in_mask = tensor.matrix('yc_in_mask', dtype='float32')
        yc = tensor.tensor3('yc', dtype='int64')
        yc_mask = tensor.tensor3('yc_mask', dtype='float32')

        n_nz_words_src = xc.shape[1]
        n_nz_words_trg = yc_in.shape[1]

        encoder_vars += [xc, xc_mask]
        decoder_vars += [yc_in, yc_in_mask, yc, yc_mask]

        xcr = xc[::-1]  # reverse characters; word order is intact
        xcr_mask = xc_mask[::-1]
        ycr_in = yc_in[::-1]
        ycr_in_mask = yc_in_mask[::-1]

        n_chars_src = xc.shape[0]
        n_chars_trg = yc_in.shape[0]

        # extract character embeddings
        cemb_src = tparams['Cemb'][xc.flatten()]
        cemb_src = cemb_src.reshape(
            [
                n_chars_src,
                n_nz_words_src,
                options['dim_char_src']
            ]
        )

        # compute hidden states of character embeddings in the source language
        cproj_src = get_layer(options['encoder'])[1](tparams,
                                                     cemb_src,
                                                     options,
                                                     prefix='char_enc_src',
                                                     mask=xc_mask)
        # repeate the above steps for reverse characters
        # NOTE word order does NOT change
        cembr_src = tparams['Cemb'][xcr.flatten()]
        cembr_src = cembr_src.reshape(
            [
                n_chars_src,
                n_nz_words_src,
                options['dim_char_src']
            ]
        )
        cprojr_src = get_layer(options['encoder'])[1](tparams,
                                                      cembr_src,
                                                      options,
                                                      prefix='char_enc_src_r',
                                                      mask=xcr_mask)

        cproj_comb_src = concatenate([cproj_src[0][-1], cprojr_src[0][-1]],
                                     axis=cproj_src[0].ndim-2)
        # word representations from characters in a reverse order
        cprojr_comb_src = cproj_comb_src[::-1]

        '''
        cproj_comb_src = get_layer('ff')[1](
            tparams, cproj_comb_src, options,
            prefix='char2word_src', activ=None
        )
        cprojr_comb_src = get_layer('ff')[1](
            tparams, cprojr_comb_src, options,
            prefix='char2word_src_r', activ=None
        )

        word_gate_src = get_layer('ff')[1](tparams, wemb_src, options,
                                           prefix='word_gate_src',
                                           activ=tensor.nnet.sigmoid)
        '''

        # fill the reduced set of word embeddings into
        # a 3D tensor of the same size with word embeddings
        nz_word_src_inds = x_mask.flatten().nonzero()
        tmp_cproj_comb_src = tensor.alloc(
            0.,
            n_words_src * n_samples, options['dim_word_src'])

        cproj_comb_src = tensor.set_subtensor(
            tmp_cproj_comb_src[nz_word_src_inds],
            cproj_comb_src)
        cproj_comb_src = cproj_comb_src.reshape(
            [
                n_words_src,
                n_samples,
                options['dim_word_src']
            ]
        )

        '''
        src_inp = word_gate_src * wemb_src + \
            (1 - word_gate_src) * cproj_comb_src
        '''

        src_inp = concatenate([wemb_src, cproj_comb_src], axis=wemb_src.ndim-1)

        '''
        word_gate_src_r = get_layer('ff')[1](tparams, wembr_src, options,
                                             prefix='word_gate_src',
                                             activ=tensor.nnet.sigmoid)
        '''

        nz_wordr_src_inds = xr_mask.flatten().nonzero()
        tmp_cprojr_comb_src = tensor.alloc(
            0.,
            n_words_src * n_samples, options['dim_word_src'])

        cprojr_comb_src = tensor.set_subtensor(
            tmp_cprojr_comb_src[nz_wordr_src_inds],
            cprojr_comb_src)
        cprojr_comb_src = cprojr_comb_src.reshape(
            [
                n_words_src,
                n_samples,
                options['dim_word_src']
            ]
        )

        '''
        src_inpr = word_gate_src_r * wembr_src + \
            (1 - word_gate_src_r) * cprojr_comb_src
        '''

        src_inpr = concatenate([wembr_src, cprojr_comb_src],
                               axis=wembr_src.ndim-1)
    else:
        src_inp, src_inpr = wemb_src, wemb_src

    # hidden states for new word embeddings for the forward rnn
    src_proj = get_layer(options['encoder'])[1](tparams,
                                                src_inp,
                                                options,
                                                prefix='word_encoder',
                                                mask=x_mask)
    # hidden states for new word embeddings for the backward rnn
    src_projr = get_layer(options['encoder'])[1](tparams,
                                                 src_inpr,
                                                 options,
                                                 prefix='word_encoder_r',
                                                 mask=xr_mask)

    # context will be the concatenation of forward and backward rnns
    ctx = concatenate([src_proj[0], src_projr[0][::-1]],
                      axis=src_proj[0].ndim - 1)

    if options['init_decoder'] == 'last':
        ctx_mean = concatenate([src_proj[0][-1], src_projr[0][-1]],
                               axis=src_proj[0].ndim-2)
    elif options['init_decoder'] == 'average':
        ctx_mean = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]
    elif options['init_decoder'] == 'adaptive':
        ctx_mean = (ctx * x_mask[:, :, None]).max(0)

        # NOTE compute importance of words for the given context vector
        proj_states = get_layer('ff')[1](tparams,
                                         ctx,
                                         options,
                                         prefix='ff_state_proj',
                                         activ=None)
        proj_mean = get_layer('ff')[1](tparams,
                                       ctx_mean,
                                       options,
                                       prefix='ff_context_proj',
                                       activ=None)

        # # src words x # batches x ctxdim
        word_weights = tensor.tanh(proj_states + proj_mean[None, :, :])
        # # src words x # batches x 1
        word_weights = tensor.dot(word_weights, tparams['word_weight_score'])
        word_weights = word_weights.reshape([word_weights.shape[0],
                                             word_weights.shape[1]])
        # # src words x # batches
        word_weights = tensor.exp(
            word_weights - word_weights.max(0, keepdims=True))
        word_weights = word_weights * x_mask
        # # src words x # batches
        word_weights = word_weights / word_weights.sum(0, keepdims=True)

        # update the context vector
        ctx_mean = (ctx * word_weights[:, :, None]).sum(0)

    # initial decoder state
    init_state = get_layer('ff')[1](tparams,
                                    ctx_mean,
                                    options,
                                    prefix='ff_word_state',
                                    activ=tensor.tanh)

    # word embedding (target)
    wemb_trg = tparams['Wemb_dec'][y.flatten()]
    wemb_trg = wemb_trg.reshape([n_words_trg, n_samples,
                                options['dim_word_trg']])

    if options['use_character']:
        # character embedding in the target language
        cemb_trg = tparams['Cemb_dec'][yc_in.flatten()]
        cemb_trg = cemb_trg.reshape(
            [
                n_chars_trg,
                n_nz_words_trg,
                options['dim_char_trg']
            ]
        )

        # hidden states of character embeddings in the target language
        cproj_trg = get_layer(options['encoder'])[1](tparams,
                                                     cemb_trg,
                                                     options,
                                                     prefix='char_enc_trg',
                                                     mask=yc_in_mask)

        # repeat for the reverse characters
        cembr_trg = tparams['Cemb_dec'][ycr_in.flatten()]
        cembr_trg = cembr_trg.reshape(
            [
                n_chars_trg,
                n_nz_words_trg,
                options['dim_char_trg']
            ]
        )

        # hidden states of character embeddings in the target language
        cprojr_trg = get_layer(options['encoder'])[1](tparams,
                                                      cembr_trg,
                                                      options,
                                                      prefix='char_enc_trg_r',
                                                      mask=ycr_in_mask)

        # pick the last state to represent words in the forward chain of words
        cproj_comb_trg = concatenate([cproj_trg[0][-1], cprojr_trg[0][-1]],
                                     axis=cproj_trg[0].ndim-2)

        '''
        cproj_comb_trg = get_layer('ff')[1](tparams, cproj_comb_trg, options,
                                            prefix='char2word_trg',
                                            activ=None)

        word_gate_trg = get_layer('ff')[1](tparams, wemb_trg, options,
                                           prefix='word_gate_trg',
                                           activ=tensor.nnet.sigmoid)
        '''

        nz_word_trg_inds = y_mask.flatten().nonzero()
        new_cproj_comb_trg = tensor.alloc(
            0.,
            n_words_trg * n_samples, options['dim_word_trg'])

        new_cproj_comb_trg = tensor.set_subtensor(
            new_cproj_comb_trg[nz_word_trg_inds],
            cproj_comb_trg)
        cproj_comb_trg = new_cproj_comb_trg.reshape(
            [
                n_words_trg,
                n_samples,
                options['dim_word_trg']
            ]
        )

        '''
        trg_inp = word_gate_trg * wemb_trg + \
            (1 - word_gate_trg) * cproj_comb_trg
        '''
        trg_inp = concatenate([wemb_trg, cproj_comb_trg], axis=wemb_trg.ndim-1)
    else:
        trg_inp = wemb_trg

    # We will shift the target sequence one time step
    # to the right. This is done because of the bi-gram connections in the
    # readout and decoder rnn. The first target will be all zeros and we will
    # not condition on the last output.
    trg_inp_shifted = tensor.zeros_like(trg_inp)
    trg_inp_shifted = tensor.set_subtensor(trg_inp_shifted[1:], trg_inp[:-1])
    trg_inp = trg_inp_shifted

    # decoder - pass through the decoder conditional gru with attention
    trg_proj = get_layer(options['decoder'])[1](tparams,
                                                trg_inp,
                                                options,
                                                prefix='word_decoder',
                                                mask=y_mask,
                                                context=ctx,
                                                context_mask=x_mask,
                                                one_step=False,
                                                init_state=init_state)
    # hidden states of the decoder gru
    proj_h = trg_proj[0]

    # weighted averages of context, generated by attention module
    ctxs = trg_proj[1]

    # weights (alignment matrix)
    # num trg words x batch size x num src words
    opt_ret['dec_alphas'] = trg_proj[2]

    # compute word probabilities
    # hidden at t to word at t
    word_logit_lstm = get_layer('ff')[1](tparams,
                                         proj_h,
                                         options,
                                         prefix='ff_word_logit_lstm',
                                         activ=None)
    # combined representation of word at t-1 to word at t
    word_logit_prev = get_layer('ff')[1](tparams,
                                         trg_inp,
                                         options,
                                         prefix='ff_word_logit_prev',
                                         activ=None)
    # context at t to word at t
    word_logit_ctx = get_layer('ff')[1](tparams,
                                        ctxs,
                                        options,
                                        prefix='ff_word_logit_ctx',
                                        activ=None)

    word_logit = tensor.tanh(
        word_logit_lstm +
        word_logit_prev +
        word_logit_ctx
    )

    if options['use_dropout']:
        word_logit = dropout_layer(word_logit, use_noise, trng)
    word_logit = get_layer('ff')[1](tparams,
                                    word_logit,
                                    options,
                                    prefix='ff_word_logit',
                                    activ=None)
    word_logit_shp = word_logit.shape
    word_probs = tensor.nnet.softmax(word_logit.reshape(
        [
            word_logit_shp[0] * word_logit_shp[1],
            word_logit_shp[2]
        ]
    ))

    # compute cost for words
    y_flat = y.flatten()
    y_flat_idx = tensor.arange(y_flat.shape[0]) * options['n_words_trg'] + \
        y_flat

    word_cost = -tensor.log(word_probs.flatten()[y_flat_idx])
    word_cost = word_cost.reshape([y.shape[0], y.shape[1]])
    word_cost = (word_cost * y_mask).sum(0)
    word_cost.name = 'word_cost'

    costs = [word_cost]

    # compute character probabilities
    if options['use_character']:

        """ Character decoder in the target side

        """
        # shift precomputed word reprs from characters to the right
        cproj_comb_trg_shifted = tensor.zeros_like(cproj_comb_trg)
        cproj_comb_trg_shifted = tensor.set_subtensor(
            cproj_comb_trg_shifted[1:],
            cproj_comb_trg[:-1]
        )
        cproj_comb_trg = cproj_comb_trg_shifted

        # initial character decoder state
        init_char_state = get_layer('ff')[1](tparams,
                                             proj_h,
                                             options,
                                             prefix='ff_char_state',
                                             activ=tensor.tanh)

        char_dec_emb = tparams['Cemb_dec'][yc.flatten()]
        char_dec_emb = char_dec_emb.reshape(
            [
                n_chars_trg,
                n_words_trg,
                n_samples,
                options['dim_char_trg']
            ]
        )

        # shift character indices over the axis of character
        char_dec_emb_shifted = tensor.zeros_like(char_dec_emb)
        char_dec_emb_shifted = tensor.set_subtensor(char_dec_emb_shifted[1:],
                                                    char_dec_emb[:-1])
        char_dec_emb = char_dec_emb_shifted

        proj_char_h = get_layer('gru')[1](tparams,
                                          char_dec_emb,
                                          options,
                                          prefix='char_decoder',
                                          mask=yc_mask,
                                          init_state=init_char_state)
        # proj_char_h: (# chars x # words x # samples x char hid dim)
        proj_char_h = proj_char_h[0]

        wemb_trg = tparams['Wemb_dec'][y.flatten()]
        wemb_trg = wemb_trg.reshape([n_words_trg, n_samples,
                                    options['dim_word_trg']])

        '''
        word_gate_trg = get_layer('ff')[1](tparams, wemb_trg, options,
                                           prefix='word_gate_trg',
                                           activ=tensor.nnet.sigmoid)
        trg_inp = word_gate_trg * wemb_trg + \
            (1 - word_gate_trg) * cproj_comb_trg
        '''
        trg_inp = concatenate([wemb_trg, cproj_comb_trg], axis=wemb_trg.ndim-1)

        # from hidden of character at i of word at t to character t,i
        # char_logit_lstm: (# chars x # words x # samples x trg char dim)
        char_logit_lstm = get_layer('ff')[1](tparams,
                                             proj_char_h,
                                             options,
                                             prefix='ff_char_logit_lstm',
                                             activ=None)
        # from character t,(i-1) to character t,i
        # char_logit_prev: (# chars x # words x # samples x trg char dim)
        char_logit_prev = get_layer('ff')[1](tparams,
                                             char_dec_emb,
                                             options,
                                             prefix='ff_char_logit_prev_c',
                                             activ=None)
        # from character embedding t-1 to character t,i
        # char_logit_prev_cemb: (# words x # samples x trg char dim)
        # from word at t to chracter t,i
        # char_logit_cur_w: (# words x # samples x trg char dim)
        char_logit_word = get_layer('ff')[1](
            tparams,
            trg_inp,
            options,
            prefix='ff_char_logit_word',
            activ=None
        )
        char_logit_word_state = get_layer('ff')[1](
            tparams,
            proj_h,
            options,
            prefix='ff_char_logit_word_state',
            activ=None
        )

        char_logit = tensor.tanh(
            char_logit_lstm +
            char_logit_prev +
            char_logit_word +
            char_logit_word_state
        )

        if options['use_dropout']:
            char_logit = dropout_layer(char_logit, use_noise, trng)
        char_logit = get_layer('ff')[1](tparams,
                                        char_logit,
                                        options,
                                        prefix='ff_char_logit',
                                        activ=None)
        char_logit_shp = char_logit.shape
        char_probs = tensor.nnet.softmax(char_logit.reshape(
            [
                char_logit_shp[0] * char_logit_shp[1] * char_logit_shp[2],
                char_logit_shp[3]
            ]
        ))

        # compute character cost
        yc_flat = yc.flatten()
        yc_shp = yc.shape
        yc_flat_idx = tensor.arange(yc_flat.shape[0]) * options['n_chars_trg'] + \
            yc_flat
        char_cost = -tensor.log(char_probs.flatten()[yc_flat_idx])
        char_cost = char_cost.reshape([yc_shp[0], yc_shp[1], yc_shp[2]])
        # sum of losses over characters in a word
        char_cost = (char_cost * yc_mask).sum(0)
        # summing losses over all words in a sentence
        char_cost = char_cost.sum(0)
        char_cost.name = 'char_cost'

        costs += [char_cost]

    return trng, use_noise, encoder_vars, decoder_vars, opt_ret, costs


# build a sampler
def build_sampler(tparams, options, trng):
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')
    xr = x[::-1]
    xr_mask = x_mask[::-1]

    encoder_vars = [x, x_mask]

    n_words_src = x.shape[0]
    n_samples = x.shape[1]

    # word embedding (source), forward and backward
    wemb_src = tparams['Wemb'][x.flatten()]
    wemb_src = wemb_src.reshape([n_words_src, n_samples,
                                options['dim_word_src']])

    # word embedding for backward rnn (source)
    wembr_src = tparams['Wemb'][xr.flatten()]
    wembr_src = wembr_src.reshape([n_words_src, n_samples,
                                  options['dim_word_src']])
    if options['use_character']:
        xc = tensor.tensor3('xc', dtype='int64')
        xc_mask = tensor.tensor3('xc_mask', dtype='float32')
        xcr = xc[::-1]
        xcr_mask = xc_mask[::-1]

        encoder_vars += [xc, xc_mask]

        n_chars_src = xc.shape[0]

        # extract character embeddings
        cemb_src = tparams['Cemb'][xc.flatten()]
        cemb_src = cemb_src.reshape(
            [
                n_chars_src,
                n_words_src,
                n_samples,
                options['dim_char_src']
            ]
        )

        # compute hidden states of character embeddings
        cproj_src = get_layer(options['encoder'])[1](tparams,
                                                     cemb_src,
                                                     options,
                                                     prefix='char_enc_src',
                                                     mask=xc_mask)

        cembr_src = tparams['Cemb'][xcr.flatten()]
        cembr_src = cembr_src.reshape(
            [
                n_chars_src,
                n_words_src,
                n_samples,
                options['dim_char_src']
            ]
        )
        cprojr_src = get_layer(options['encoder'])[1](tparams,
                                                      cembr_src,
                                                      options,
                                                      prefix='char_enc_src_r',
                                                      mask=xcr_mask)

        cproj_comb_src = concatenate([cproj_src[0][-1], cprojr_src[0][-1]],
                                     axis=cproj_src[0].ndim-2)
        # word representations from characters in a reverse order
        cprojr_comb_src = cproj_comb_src[::-1]

        '''
        cproj_comb_src = get_layer('ff')[1](
            tparams, cproj_comb_src, options,
            prefix='char2word_src', activ=None
        )
        cprojr_comb_src = get_layer('ff')[1](
            tparams, cprojr_comb_src, options,
            prefix='char2word_src_r', activ=None
        )

        word_gate_src = get_layer('ff')[1](tparams, wemb_src, options,
                                           prefix='word_gate_src',
                                           activ=tensor.nnet.sigmoid)

        src_inp = word_gate_src * wemb_src + \
            (1 - word_gate_src) * cproj_comb_src

        word_gate_src_r = get_layer('ff')[1](tparams, wembr_src, options,
                                             prefix='word_gate_src',
                                             activ=tensor.nnet.sigmoid)

        src_inpr = word_gate_src_r * wembr_src + \
            (1 - word_gate_src_r) * cprojr_comb_src

        gates = concatenate(
            [
                (word_gate_src >= 0.8).sum(2),
                (1-word_gate_src >= 0.5).sum(2)
            ], axis=1) / word_gate_src.shape[2].astype('float32')

        '''

        src_inp = concatenate([wemb_src, cproj_comb_src], axis=wemb_src.ndim-1)
        src_inpr = concatenate([wembr_src, cprojr_comb_src],
                               axis=wembr_src.ndim-1)

        gates = tensor.alloc(1., src_inp.shape[0], 1)
    else:
        src_inp = wemb_src
        src_inpr = wembr_src
        gates = tensor.alloc(1., src_inp.shape[0], 1)

    # encoder
    src_proj = get_layer(options['encoder'])[1](tparams,
                                                src_inp,
                                                options,
                                                prefix='word_encoder',
                                                mask=x_mask)
    # hidden states for new word embeddings for the backward rnn
    src_projr = get_layer(options['encoder'])[1](tparams,
                                                 src_inpr,
                                                 options,
                                                 prefix='word_encoder_r',
                                                 mask=xr_mask)

    # concatenate forward and backward rnn hidden states
    ctx = concatenate([src_proj[0], src_projr[0][::-1]],
                      axis=src_proj[0].ndim - 1)

    if options['init_decoder'] == 'last':
        ctx_mean = concatenate([src_proj[0][-1], src_projr[0][-1]],
                               axis=src_proj[0].ndim-2)
    elif options['init_decoder'] == 'average':
        ctx_mean = ctx.mean(0)
    elif options['init_decoder'] == 'adaptive':
        ctx_mean = (ctx * x_mask[:, :, None]).max(0)

        proj_states = get_layer('ff')[1](tparams,
                                         ctx,
                                         options,
                                         prefix='ff_state_proj',
                                         activ=None)
        proj_mean = get_layer('ff')[1](tparams,
                                       ctx_mean,
                                       options,
                                       prefix='ff_context_proj',
                                       activ=None)

        word_weights = tensor.tanh(proj_states + proj_mean[None, :, :])
        word_weights = tensor.dot(word_weights, tparams['word_weight_score'])
        word_weights = word_weights.reshape([word_weights.shape[0],
                                             word_weights.shape[1]])
        word_weights = tensor.exp(
            word_weights - word_weights.max(0, keepdims=True))
        word_weights = word_weights * x_mask
        word_weights = word_weights / word_weights.sum(0, keepdims=True)

        # update the context vector
        ctx_mean = (ctx * word_weights[:, :, None]).sum(0)

    init_word_state = get_layer('ff')[1](tparams,
                                         ctx_mean,
                                         options,
                                         prefix='ff_word_state',
                                         activ=tensor.tanh)

    LOGGER.info('Building f_init')
    encoder_outs = [init_word_state, ctx]
    if options['init_decoder'] == 'adaptive':
        encoder_outs.append(word_weights)
    if options['use_character']:
        encoder_outs.append(gates)

    f_init = theano.function(encoder_vars, encoder_outs,
                             name='f_init', profile=False)

    # y: 1 x 1
    y = tensor.vector('y_sampler', dtype='int64')
    init_word_state = tensor.matrix('init_word_state', dtype='float32')

    wemb_trg = tparams['Wemb_dec'][y]

    if options['use_character']:
        yc = tensor.matrix('yc_sampler', dtype='int64')
        yc_mask = tensor.matrix('yc_mask_sampler', dtype='float32')

        ycr = yc[::-1]
        ycr_mask = yc_mask[::-1]

        cemb_trg = tparams['Cemb_dec'][yc.flatten()]
        cemb_trg = cemb_trg.reshape([yc.shape[0], yc.shape[1],
                                     options['dim_char_trg']])

        cembr_trg = tparams['Cemb_dec'][ycr.flatten()]
        cembr_trg = cembr_trg.reshape([ycr.shape[0], ycr.shape[1],
                                       options['dim_char_trg']])

        # hidden states of forward character seuqences
        cproj_trg = get_layer(options['encoder'])[1](tparams,
                                                     cemb_trg,
                                                     options,
                                                     prefix='char_enc_trg',
                                                     mask=yc_mask)
        # hidden states of backward character sequences
        cprojr_trg = get_layer(options['encoder'])[1](tparams,
                                                      cembr_trg,
                                                      options,
                                                      prefix='char_enc_trg_r',
                                                      mask=ycr_mask)

        # combination of the last states of forward and backward RNNs
        # this corresponds to word representation
        cproj_comb_trg = concatenate([cproj_trg[0][-1], cprojr_trg[0][-1]],
                                     axis=cproj_trg[0].ndim-2)

        assert cproj_comb_trg.ndim == 2

        '''
        cproj_comb_trg = get_layer('ff')[1](tparams, cproj_comb_trg, options,
                                            prefix='char2word_trg',
                                            activ=None)

        word_gate_trg = get_layer('ff')[1](tparams, wemb_trg, options,
                                           prefix='word_gate_trg',
                                           activ=tensor.nnet.sigmoid)

        trg_inp = word_gate_trg * wemb_trg + \
            (1 - word_gate_trg) * cproj_comb_trg

        trg_gates = concatenate(
            [
                (word_gate_trg >= 0.8).sum(1, keepdims=True),
                (1-word_gate_trg >= 0.5).sum(1, keepdims=True)
            ], axis=1) / word_gate_trg.shape[1].astype('float32')
        '''
        trg_inp = concatenate([wemb_trg, cproj_comb_trg], axis=wemb_trg.ndim-1)
        trg_gates = tensor.alloc(1., 1, 1)

        # if the variables are for the first word,
        # they  should be all zero.
        cproj_comb_trg = cproj_comb_trg * (y[:, None] >= 0)
    else:
        trg_inp = wemb_trg
        trg_gates = tensor.alloc(1., 1, 1)

    # if it's the first word, emb should be all zero and it is indicated by -1
    trg_inp = trg_inp * (y[:, None] >= 0)
    wemb_trg = wemb_trg * (y[:, None] >= 0)

    # apply one step of conditional gru with attention
    trg_proj = get_layer(options['decoder'])[1](tparams,
                                                trg_inp,
                                                options,
                                                prefix='word_decoder',
                                                mask=None,
                                                context=ctx,
                                                context_mask=x_mask,
                                                one_step=True,
                                                init_state=init_word_state)
    # get the next hidden state
    next_word_state = trg_proj[0]

    # get the weighted averages of context for this target word y
    ctxs = trg_proj[1]

    dec_alphas = trg_proj[2]

    word_logit_lstm = get_layer('ff')[1](tparams,
                                         next_word_state,
                                         options,
                                         prefix='ff_word_logit_lstm',
                                         activ=None)
    # characters in a word at t-1 to word at t
    word_logit_prev = get_layer('ff')[1](tparams,
                                         trg_inp,
                                         options,
                                         prefix='ff_word_logit_prev',
                                         activ=None)
    word_logit_ctx = get_layer('ff')[1](tparams,
                                        ctxs,
                                        options,
                                        prefix='ff_word_logit_ctx',
                                        activ=None)
    word_logit = tensor.tanh(word_logit_lstm +
                             word_logit_prev +
                             word_logit_ctx)
    word_logit = get_layer('ff')[1](tparams,
                                    word_logit,
                                    options,
                                    prefix='ff_word_logit',
                                    activ=None)

    # compute the softmax probability
    next_word_probs = tensor.nnet.softmax(word_logit)

    # sample from softmax distribution to get the sample
    next_word_sample = trng.multinomial(pvals=next_word_probs).argmax(1)

    # compile a function to do the whole thing above, next word probability,
    # sampled word for the next target, next hidden state to be used
    LOGGER.info('Building f_word_next')
    f_wsamp_inps = [x_mask, y, ctx, init_word_state]
    f_wsamp_outs = [next_word_probs, next_word_sample, next_word_state,
                    dec_alphas, trg_gates]

    if options['use_character']:
        next_char_state = get_layer('ff')[1](tparams,
                                             next_word_state,
                                             options,
                                             prefix='ff_char_state',
                                             activ=tensor.tanh)

        f_wsamp_inps += [yc, yc_mask]
        f_wsamp_outs += [next_char_state, cproj_comb_trg]

    f_word_next = theano.function(f_wsamp_inps, f_wsamp_outs,
                                  name='f_word_next', profile=False)

    f_nexts = [f_word_next]

    if options['use_character']:
        # NOTE character generator
        # yc: 1 x # characters
        y = tensor.vector('y_sampler', dtype='int64')
        yc = tensor.vector('yc_sampler', dtype='int64')
        # init_char_state: # characters x char hid dim
        init_char_state = tensor.matrix('init_char_state', dtype='float32')

        wemb_trg = tparams['Wemb_dec'][y]
        '''
        word_gate_trg = get_layer('ff')[1](tparams, wemb_trg, options,
                                           prefix='word_gate_trg',
                                           activ=tensor.nnet.sigmoid)

        trg_dec_inp = word_gate_trg * wemb_trg + \
            (1 - word_gate_trg) * cproj_comb_trg
        '''

        trg_dec_inp = concatenate([wemb_trg, cproj_comb_trg],
                                  axis=wemb_trg.ndim-1)
        # char_emb: # chracters x char dim
        char_emb = tensor.switch(
            yc[:, None] < 0,
            tensor.alloc(0., 1, tparams['Cemb_dec'].shape[1]),
            tparams['Cemb_dec'][yc]
        )

        # char_emb: 1 x # characters x char dim
        char_emb = char_emb[None, :, :]
        # apply one step of conditional gru with attention
        char_proj = get_layer(options['encoder'])[1](
            tparams,
            char_emb,
            options,
            prefix='char_decoder',
            mask=None,
            one_step=True,
            init_state=init_char_state
        )
        # get the next hidden state
        # next_char_state: # characters x char hid dim
        next_char_state = char_proj[0][0]
        # char_emb = # characters x char dim
        char_emb = char_emb[0]

        # char_logit_lstm: # characters x dim_char_trg
        char_logit_lstm = get_layer('ff')[1](tparams,
                                             next_char_state,
                                             options,
                                             prefix='ff_char_logit_lstm',
                                             activ=None)
        # char_logit_lstm: # characters x dim_char_trg
        char_logit_prev = get_layer('ff')[1](tparams,
                                             char_emb,
                                             options,
                                             prefix='ff_char_logit_prev_c',
                                             activ=None)
        char_logit_word = get_layer('ff')[1](
            tparams,
            trg_dec_inp,
            options,
            prefix='ff_char_logit_word',
            activ=None
        )
        char_logit_word_state = get_layer('ff')[1](
            tparams,
            next_word_state,
            options,
            prefix='ff_char_logit_word_state',
            activ=None
        )

        char_logit = tensor.tanh(
            char_logit_lstm +
            char_logit_prev +
            char_logit_word +
            char_logit_word_state
        )

        char_logit = get_layer('ff')[1](tparams,
                                        char_logit,
                                        options,
                                        prefix='ff_char_logit',
                                        activ=None)

        # compute the softmax probability
        next_char_probs = tensor.nnet.softmax(char_logit)

        # sample from softmax distribution to get the sample
        next_char_sample = trng.multinomial(pvals=next_char_probs).argmax(1)

        # sampled char for the next target, next hidden state to be used
        LOGGER.info('Building f_char_next')
        inps = [y, yc, init_char_state, cproj_comb_trg, next_word_state]
        outs = [next_char_probs, next_char_sample, next_char_state]
        f_char_next = theano.function(inps, outs, name='f_char_next',
                                      profile=False)

        f_nexts.append(f_char_next)

    return f_init, f_nexts


# generate sample, either with stochastic sampling or beam search. Note that,
# this function iteratively calls f_init and f_next functions.
def gen_sample(tparams,
               f_init,
               f_nexts,     # list of functions to generate outputs
               inps,
               options,
               trng=None,
               k=1,
               max_sent_len=30,
               max_word_len=10,
               stochastic=True,
               argmax=False):

    if len(inps) == 2 and len(f_nexts) == 1:
        assert not options['use_character']

        x, x_mask = inps
        f_word_next = f_nexts[0]
    elif len(inps) == 4 and len(f_nexts) == 2:
        assert options['use_character']

        x, x_mask, xc, xc_mask = inps
        f_word_next, f_char_next = f_nexts
    else:
        raise ValueError('The number of input variables should be equal to '
                         'the number of items in `f_nexts` multiplied by 2')

    assert max_sent_len > 0 and max_word_len > 0

    # k is the beam size we have
    assert k >= 1

    word_live_k = 1

    word_solutions_ds = [('num_samples', 0), ('samples', []),
                         ('alignments', []), ('scores', []),
                         ('word_src_gates', []), ('word_trg_gates', [])]

    word_hypotheses_ds = [
        ('num_samples', word_live_k),
        ('samples', [[]] * word_live_k),
        ('word_trg_gates', [[]] * word_live_k),
        ('alignments', [[]] * word_live_k),
        ('scores', numpy.zeros(word_live_k).astype('float32')),
    ]

    if options['use_character']:
        # For the character decoder, we define a few more variables
        word_solutions_ds += [('character_samples', [])]
        word_hypotheses_ds += [
            ('character_samples', [[]] * word_live_k),
            ('char_states', []),
            ('cproj', []),
        ]

    word_solutions = OrderedDict(word_solutions_ds)
    word_hypotheses = OrderedDict(word_hypotheses_ds)

    def _check_stop_condition(solutions, hypotheses, k):
        return solutions['num_samples'] >= k or hypotheses['num_samples'] < 1

    # get initial state of decoder rnn and encoder context
    # ctx0 is 3d tensor of hidden states for the input sentence
    # next_state is a summary of hidden states for the input setence
    # ctx0: (# src words x # sentence (i.e., 1) x # hid dim)
    # next_state: (# sentences (i.e., 1) x # hid dim of the target setence)
    encoder_outs = f_init(*inps)
    next_word_state, ctx0 = encoder_outs[0], encoder_outs[1]
    if len(encoder_outs) == 3:
        assert (options['init_decoder'] == 'adaptive') ^ \
            options['use_character']
        if options['init_decoder'] == 'adaptive':
            word_solutions['word_weights'] = encoder_outs[2]
        if options['use_character']:
            word_solutions['word_src_gates'] = encoder_outs[2]
    elif len(encoder_outs) == 4:
        assert options['init_decoder'] == 'adaptive'
        assert options['use_character']

        word_solutions['word_weights'] = encoder_outs[2]
        word_solutions['word_src_gates'] = encoder_outs[3]

    next_w = -1 * numpy.ones((1, )).astype('int64')  # bos indicator

    if options['use_character']:
        next_chars = -1 * numpy.ones((1, word_live_k)).astype('int64')
        next_chars_mask = numpy.zeros_like(next_chars).astype('float32')

    for ii in xrange(max_sent_len):
        word_live_k = word_hypotheses['num_samples']

        # NOTE `hyp_samples` is initailized by a list with a single empty list
        # repeat the contexts the number of hypotheses
        # (corresponding to the number of setences)
        # (# src words x 1 x hid dim) -> (# src words x # next_hyp x hid dim)
        ctx = numpy.tile(ctx0, [1, word_live_k, 1])
        x_mask_ = numpy.tile(x_mask, [1, word_live_k])

        # inputs to sample word candidates
        wsamp_inps = [x_mask_, next_w, ctx, next_word_state]
        if options['use_character']:
            wsamp_inps += [next_chars, next_chars_mask]

        # generate a word for the given last hidden states
        # and previously generated words
        wsamp_outs = f_word_next(*wsamp_inps)

        next_p, next_word_state, next_alphas, next_trg_gates = \
            wsamp_outs[0], wsamp_outs[2], wsamp_outs[3], wsamp_outs[4]

        if options['use_character']:
            next_char_state, cproj_comb_trg = wsamp_outs[5], wsamp_outs[6]

        # preparation of inputs to beam search
        beam_state = [next_word_state, next_p, next_alphas, next_trg_gates]

        if options['use_character']:
            beam_state += [next_char_state, cproj_comb_trg]

        # perform beam search to generate most probable word sequence
        # with limited budget.
        word_solutions, word_hypotheses = \
            beam_search(word_solutions, word_hypotheses, beam_state,
                        decode_char=options['use_character'], k=k)

        if _check_stop_condition(word_solutions, word_hypotheses, k):
            break

        # get the last single word for each hypothesis
        next_w = numpy.array([w[-1] for w in word_hypotheses['samples']])
        next_word_state = numpy.array(word_hypotheses['states'])

        # Perform the nested beam search if the model can handle characters
        # Otherwise, repeat the beam search procedure above.
        if options['use_character']:
            cproj_comb_trg = numpy.array(word_hypotheses['cproj'])
            init_char_state = numpy.array(word_hypotheses['char_state'])

            word_live_k = word_hypotheses['num_samples']
            next_chars = [None] * word_live_k

            # perform nested beam search for character sequences
            for k_idx in xrange(word_live_k):
                char_live_k = 1

                char_solutions = OrderedDict([
                    ('num_samples', 0),
                    ('samples', []),
                    ('scores', []),
                ])

                char_hypotheses = OrderedDict([
                    ('num_samples', char_live_k),
                    ('samples', [[]] * char_live_k),
                    ('scores', numpy.zeros(char_live_k).astype('float32')),
                    ('states', []),
                ])

                next_w_k = next_w[k_idx]
                next_c = -1 * numpy.ones((1, )).astype('int64')
                # hidden state of the rnn decoder for characters
                next_char_state = numpy.tile(init_char_state[k_idx][None, :],
                                             [char_live_k, 1])
                cproj_comb_trg_k = cproj_comb_trg[k_idx][None, :]
                next_word_state_k = next_word_state[k_idx][None, :]

                for jj in xrange(max_word_len):
                    char_live_k = char_hypotheses['num_samples']
                    cproj_comb_trg_ = numpy.tile(cproj_comb_trg_k,
                                                 [char_live_k, 1])
                    next_word_state_ = numpy.tile(next_word_state_k,
                                                  [char_live_k, 1])
                    next_w_ = numpy.tile(next_w_k, char_live_k)

                    inps = [next_w_, next_c, next_char_state,
                            cproj_comb_trg_, next_word_state_]
                    next_pc, next_c, next_char_state = f_char_next(*inps)

                    # perform beam search to generate
                    # the most probable char sequences with limited budget.
                    beam_state = [next_char_state, next_pc]
                    char_solutions, char_hypotheses \
                        = beam_search(char_solutions, char_hypotheses,
                                      beam_state, k=k, level='char')

                    if _check_stop_condition(char_solutions,
                                             char_hypotheses, k):
                        break

                    # get the last single character for each hypothesis
                    # we keep track of a word of generated characters so far
                    next_c = numpy.array(
                        [c[-1] for c in char_hypotheses['samples']])
                    next_char_state = numpy.array(char_hypotheses['states'])

                # dump remaining hypotheses
                if char_hypotheses['num_samples'] > 0:
                    for idx in xrange(char_hypotheses['num_samples']):
                        char_solutions['samples'].append(
                            char_hypotheses['samples'][idx])
                        char_solutions['scores'].append(
                            char_hypotheses['scores'][idx])

                # NOTE select the most probable character sequence
                # for the current word sequences (beam)
                char_sample_scores = char_solutions['scores'] /\
                    numpy.array([len(s) for s in char_solutions['samples']])
                cc = char_solutions['samples'][
                    char_sample_scores[1:].argmin()+1]
                next_chars[k_idx] = cc

            # NOTE adding chosen character seuqneces into word hypotheses
            assert len(next_chars) == word_hypotheses['num_samples']
            for idx, char_seq in enumerate(next_chars):
                word_hypotheses['character_samples'][idx].append(char_seq)

            # NOTE character sequences into matrix
            max_char_len = numpy.max(
                [
                    len(char_seq) for char_seq in next_chars
                ]
            )
            new_next_chars = numpy.zeros(
                (max_char_len, word_live_k)).astype('int64')
            next_chars_mask = numpy.zeros(
                (max_char_len, word_live_k)).astype('float32')

            for word_hyp_idx, char_seq in enumerate(next_chars):
                new_next_chars[:len(char_seq), word_hyp_idx] = \
                    numpy.array(char_seq)
                next_chars_mask[:len(char_seq), word_hyp_idx] = 1.

            next_chars = new_next_chars

    # dump every remaining one
    if word_hypotheses['num_samples'] > 0:
        for idx in xrange(word_hypotheses['num_samples']):
            word_solutions['samples'].append(
                word_hypotheses['samples'][idx])
            word_solutions['scores'].append(
                word_hypotheses['scores'][idx])
            # word_solutions['word_trg_gates'].append(
            #     word_hypotheses['word_trg_gates'][idx])
            word_solutions['alignments'].append(
                word_hypotheses['alignments'][idx])
            if options['use_character']:
                word_solutions['character_samples'].append(
                    word_hypotheses['character_samples'][idx])

    return word_solutions


# calculate the log probablities on a given corpus using translation model
def pred_probs(f_log_probs, options, stream):
    probs = []

    n_done = 0

    for xc, x, x_mask, yc, y, y_mask in stream.get_epoch_iterator():
        n_done += len(x)

        x, x_mask, y, y_mask = x.T, x_mask.T, y.T, y_mask.T

        encoder_inps = [x, x_mask]
        decoder_inps = [y, y_mask]

        if options['use_character']:
            xc, xc_mask = prepare_character_tensor(xc)
            yc, yc_mask = prepare_character_tensor(yc)

            xc_in = xc.reshape([xc.shape[0], -1])
            xc_in_mask = xc_mask.reshape([xc_mask.shape[0], -1])

            xc_in = xc_in[:, x_mask.flatten() > 0]
            xc_in_mask = xc_in_mask[:, x_mask.flatten() > 0]

            yc_in = yc.reshape([yc.shape[0], -1])
            yc_in_mask = yc_mask.reshape([yc_mask.shape[0], -1])

            yc_in = yc_in[:, y_mask.flatten() > 0]
            yc_in_mask = yc_in_mask[:, y_mask.flatten() > 0]

            encoder_inps += [xc_in, xc_in_mask]
            decoder_inps += [yc_in, yc_in_mask, yc, yc_mask]

        inps = encoder_inps + decoder_inps

        pprobs = f_log_probs(*inps)

        for pp in pprobs:
            probs.append(pp)

        if not numpy.isfinite(numpy.mean(probs)):
            raise RuntimeError('non-finite probabilities')

    return numpy.array(probs)


def save_params(params, filename, symlink=None):
    """Save the parameters.

    Saves the parameters as an ``.npz`` file. It optionally also creates a
    symlink to this archive.

    """
    numpy.savez(filename, **params)
    if symlink:
        if os.path.lexists(symlink):
            os.remove(symlink)
        os.symlink(filename, symlink)
