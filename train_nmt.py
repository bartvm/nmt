import numpy
import os

from nmt import train

def main(job_id, params):
    print params
    basedir = params['basedir'][0];
    print basedir

if __name__ == '__main__':
    basedir = '/data/lisatmp3/nmt/europarl'
    main(0, {
	'basedir' : ['/data/lisatmp3/nmt/europarl'],
        'model': ['%s/models/model_attention.npz'%basedir],
        'dim_word': [150],
        'dim': [124],
        'n-words': [3000],
        'optimizer': ['adadelta'],
        'decay-c': [0.],
        'clip-c': [1.],
        'use-dropout': [False],
        'learning-rate': [0.0001],
        'reload': [False]})


