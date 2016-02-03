#!/usr/bin/env python

from data_iterator import WCIterator
from nmt import prepare_data

if __name__ == '__main__':
    src_data = '/home/jsnam/data/europarl/europarl-v7.fr-en.en.tok'
    trg_data = '/home/jsnam/data/europarl/europarl-v7.fr-en.fr'
    src_dict = '/home/jsnam/data/europarl/europarl-v7.fr-en.en.tok.pkl'
    trg_dict = '/home/jsnam/data/europarl/europarl-v7.fr-en.fr.pkl'
    batch_size = 80
    maxlen = 50
    n_words_src, n_words_target = 30000, -1

    train = WCIterator(src_data,
                       trg_data,
                       src_dict,
                       trg_dict,
                       n_words_source=n_words_src,
                       n_words_target=n_words_target,
                       batch_size=batch_size,
                       maxlen=maxlen)

    print 'Ready'

    for x, y in train:
        print len(x[1])
        print len(y[1])

        x, x_mask, y, y_mask = prepare_data(x, y)

        print x.shape
        print y.shape
        break
