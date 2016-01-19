import numpy
import os

from trg_char_nmt import train

def main(job_id, params):
    print params
    validerr = train(saveto=params['model'][0],
                                        reload_=params['reload'][0],
                                        dim_word_src=params['dim_word_src'][0],
                                        dim_word_trg=params['dim_word_trg'][0],
                                        dim=params['dim'][0],
                                        n_words=params['trg_vocab_size'][0],
                                        n_words_src=params['src_vocab_size'][0],
                                        decay_c=params['decay-c'][0],
                                        clip_c=params['clip-c'][0],
                                        lrate=params['learning-rate'][0],
                                        optimizer=params['optimizer'][0], 
                                        maxlen=50,
                                        batch_size=32,
                                        valid_batch_size=32,
					datasets=['/home/%s/data/mt/europarl-v7.fr-en.en.tok'%os.environ['USER'], 
					'/home/%s/data/mt/europarl-v7.fr-en.fr'%os.environ['USER']],
					valid_datasets=['/home/%s/data/mt/newstest2011.en.tok'%os.environ['USER'], 
					'/home/%s/data/mt/newstest2011.fr'%os.environ['USER']],
					dictionaries=['/home/%s/data/mt/europarl-v7.fr-en.en.tok.pkl'%os.environ['USER'], 
					'/home/%s/data/mt/europarl-v7.fr-en.fr.pkl'%os.environ['USER']],
                                        validFreq=5000,
                                        dispFreq=500,
                                        saveFreq=5000,
                                        sampleFreq=1000,
                                        use_dropout=params['use-dropout'][0])
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': ['model_hal.npz'],
        'dim_word_src': [512],
        'dim_word_trg': [100],
        'dim': [1024],
        'src_vocab_size': [300000], 
        'trg_vocab_size': [324], 
        'optimizer': ['adadelta'],
        'decay-c': [0.], 
        'clip-c': [1.], 
        'use-dropout': [False],
        'learning-rate': [0.0001],
        'reload': [False]})
