import numpy
import sys


def contextwin(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence

    l :: array containing the word indexes

    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >= 1
    l = list(l)

    lpadded = win // 2 * [0] + l + win // 2 * [0]
    out = [lpadded[i:(i + win)] for i in range(len(l)) if l[i]==1]

    #This condition not necessery, we worry only for UNKs
    #assert len(out) == len(l)
    return out



def getmaskwin(l, mask_element):
    '''
    mask_element :: element which you want to check
    l :: array containing the word indexes

    or can also use, np.where(list == mask_element)
    enumerate is probably efficient.

    '''
    assert (mask_element) == 1
    assert len(l) >= 1
    l = list(l)
    masked_list = numpy.zeros(len(l))
    indices = [i for i, x in enumerate(l) if x == mask_element]
    masked_list[indices[:]] = mask_element;
    assert len(masked_list) == len(l)
    return masked_list

def get_ctx_matrix(x_data, context_len):
        x_temp = x_data.transpose()
        unk_ctx = numpy.array([])
        count = 0
        for i in range(len(x_temp)):
                outa  = contextwin(x_temp[i], context_len);
                if outa:
                        if count == 0:
                                unk_ctx = outa;
                        else:
                                unk_ctx = numpy.append(unk_ctx, outa, axis = 0)
                        count = count + 1;

        return unk_ctx;



def prepare_data(seqs_x, seqs_y, maxlen=None, n_words_src=30000,
                 n_words=30000):

    # x: a list of sentences
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
            return None, None, None, None

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
  
    unk_ctx = get_ctx_matrix(x, 5)
    return x, x_mask, y, y_mask, unk_ctx


