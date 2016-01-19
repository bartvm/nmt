#!/usr/bin/env python
import theano
import theano.tensor as T
import numpy as np

from utils import *
from layers import *

if __name__ == '__main__':

    op = Parameters()
    with op:
        batch_sz=2
        n_chars_src=10
        src_char_dim=3
        src_h_word_dim=10
        src_h_sent_dim=10
        n_chars_trg=6
        trg_char_dim=3
        trg_h_word_dim=10
        trg_h_sent_dim=10

    params = OrderedDict()
    params['Cemb_src'] = norm_weight(op['n_chars_src'], op['src_char_dim'])     
    params['Cemb_trg'] = norm_weight(op['n_chars_trg'], op['trg_char_dim'])     
    params = param_init_gru(op,
                            params,
                            pre='char_enc',
                            nin=op['src_char_dim'],
                            dim=op['src_h_word_dim'])
    params = param_init_gru(op,
                            params,
                            pre='word_enc',
                            nin=op['src_h_word_dim'],
                            dim=op['src_h_sent_dim'])

    ctxdim = 2 * op['src_h_sent_dim']

    params = param_init_gru_cond(op,
                                params,
                                pre='char_dec',
                                nin=op['trg_char_dim'],
                                dim=op['trg_h_word_dim'],
                                dimctx=ctxdim)
    params = param_init_gru_cond(op,
                                params,
                                pre='word_dec',
                                nin=op['trg_h_word_dim'],
                                dim=op['trg_h_sent_dim'],
                                dimctx=ctxdim)

    tp = init_tparams(params)

    # build a model
    x = tensor.tensor3('x', dtype='int64')
    y = tensor.tensor3('y', dtype='int64')

    n_chars_src = x.shape[0]
    n_words_src = x.shape[1]
    n_chars_trg = y.shape[0]
    n_words_trg = y.shape[1]
    n_sents = x.shape[2]

    emb_src = tp['Cemb_src'][x.flatten()]
    emb_src = emb_src.reshape([n_chars_src, n_words_src, n_sents, op['src_char_dim']])

    #f_emb = theano.function(inputs=[x], outputs=[emb_src])

    char_hidden_states = gru_layer(tp, emb_src, op, pre='char_enc')
    char_hidden_states = char_hidden_states[0]

    word_hidden_states = gru_layer(tp, char_hidden_states[-1], op, pre='word_enc')
    word_hidden_states = word_hidden_states[0]

    # compile theano functions
    f_encode_word = theano.function(inputs=[x],
                            outputs=[char_hidden_states])
    f_encode_sent = theano.function(inputs=[x],
                            outputs=[word_hidden_states])

    """

        Test drive

    """
    # define character sequences for both source and target languages

    n_chars_src = 4
    n_words_src = 3

    src_word_1 = [7,0,6,8]
    src_word_2 = [3,5,9,3]
    src_word_3 = [8,4,2,0]

    src_word_4 = [0,1,2,3]
    src_word_5 = [1,4,7,9]
    src_word_6 = [5,1,6,1]

    src_sentence_1 = np.transpose(np.array([src_word_1,src_word_2,src_word_3]))
    src_sentence_2 = np.transpose(np.array([src_word_4,src_word_5,src_word_6]))

    # sentences is a 3D tensor of (No. chars x No. words x No. sentences)
    src_sentences = np.concatenate([
                            src_sentence_1[:,:,None],
                            src_sentence_2[:,:,None]
                            ],axis=2)

    trg_word_1 = [7,0,6,8]
    trg_word_2 = [3,5,9,3]
    trg_word_3 = [8,4,2,0]

    trg_word_4 = [0,1,2,3]
    trg_word_5 = [1,4,7,9]
    trg_word_6 = [5,1,6,1]

    trg_sentence_1 = np.transpose(np.array([trg_word_1,trg_word_2,trg_word_3]))
    trg_sentence_2 = np.transpose(np.array([trg_word_4,trg_word_5,trg_word_6]))

    trg_sentences = np.concatenate([
                            trg_sentence_1[:,:,None],
                            trg_sentence_2[:,:,None]
                            ],axis=2)


    assert trg_sentences.shape[2] == src_sentences.shape[2]

    # feed the data to the encoder
    ret_char_hidden_states = f_encode_word(src_sentences)[0]

    assert len(ret_char_hidden_states) == n_chars_src, \
        'The resulting output should match the maximum number of characters \
            in a set of sentences: expected: %d, actual: %d' \
        % (n_chars_src, len(ret_char_hidden_states))

    print ret_char_hidden_states[0].shape

    ret_word_hidden_states = f_encode_sent(src_sentences)[0]

    assert len(ret_word_hidden_states) == n_words_src, \
        'The resulting output should match the maximum number of words \
            in a set of sentences: expected: %d, actual: %d' \
        % (n_words_src, len(ret_word_hidden_states))

    print ret_word_hidden_states[0].shape
    print len(ret_word_hidden_states)

    # decoding starts

