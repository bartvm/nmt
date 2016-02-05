import copy
import os
import numpy
import time
import theano
from theano import tensor
import six
from six.moves import xrange, cPickle

from nmt_base import (prepare_data, pred_probs, build_model,
                      build_sampler, init_params, gen_sample)
from utils import load_params, init_tparams, zipp, unzip, itemlist
from data_iterator import get_stream, load_dict
import optimizers


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
          n_words_src=100000,  # source vocabulary size
          n_words=-1,  # target vocabulary size
          maxlen=100,  # maximum length of the description
          optimizer='rmsprop',
          batch_size=16,
          valid_batch_size=16,
          saveto='model.npz',
          validFreq=1000,
          saveFreq=1000,   # save the parameters after every saveFreq updates
          sampleFreq=100,   # generate some samples after every sampleFreq
          datasets=[
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok',
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok'],
          valid_datasets=['../data/dev/newstest2011.en.tok',
                          '../data/dev/newstest2011.fr.tok'],
          dictionaries=[
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok.pkl',
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok.pkl'],
          use_dropout=False,
          reload_=False):

    # Model options
    model_options = locals().copy()

    # load dictionaries and invert them
    worddicts = [None] * len(dictionaries)
    worddicts_r = [None] * len(dictionaries)
    for ii, dd in enumerate(dictionaries):
        worddicts[ii] = load_dict(dd)
        worddicts_r[ii] = dict()
        for kk, vv in six.iteritems(worddicts[ii]):
            worddicts_r[ii][vv] = kk

    # reload options
    if reload_ and os.path.exists(saveto):
        with open('%s.pkl' % saveto, 'rb') as f:
            models_options = cPickle.load(f, encoding='latin')

    print('Loading data')
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

    print('Building model')
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        params = load_params(saveto, params)

    tparams = init_tparams(params)

    trng, use_noise, \
        x, x_mask, y, y_mask, \
        opt_ret, \
        cost = \
        build_model(tparams, model_options)
    inps = [x, x_mask, y, y_mask]

    print('Buliding sampler')
    f_init, f_next = build_sampler(tparams, model_options, trng)

    # before any regularizer
    print('Building f_log_probs...', end=' ')
    f_log_probs = theano.function(inps, cost, profile=False)
    print('Done')

    cost = cost.mean()

    # apply L2 regularization on weights
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in six.iteritems(tparams):
            weight_decay += (vv**2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # regularize the alpha weights
    if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * ((tensor.cast(
            y_mask.sum(0) // x_mask.sum(0), 'float32')[:, None] -
                                opt_ret['dec_alphas'].sum(0))**2).sum(1).mean()
        cost += alpha_reg

    # after all regularizers - compile the computational graph for cost
    print('Building f_cost...', end=' ')
    f_cost = theano.function(inps, cost, profile=False)
    print('Done')

    print('Computing gradient...', end=' ')
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print('Done')

    # apply gradient clipping here
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2), g / tensor.sqrt(
                g2) * clip_c, g))
        grads = new_grads

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print('Building optimizers...', end=' ')
    f_grad_shared, f_update = getattr(optimizers, optimizer)(lr, tparams,
                                                             grads, inps, cost)
    print('Done')

    print('Optimization')

    history_errs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        history_errs = list(numpy.load(saveto)['history_errs'])
    best_p = None
    bad_counter = 0

    uidx = 0
    estop = False
    for eidx in xrange(max_epochs):
        n_samples = 0

        for x, y in train_stream.get_epoch_iterator():
            n_samples += len(x)
            uidx += 1
            use_noise.set_value(1.)

            x, x_mask, y, y_mask = prepare_data(x, y, maxlen=maxlen)

            if x is None:
                # print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                continue

            ud_start = time.time()

            # compute cost, grads and copy grads to shared variables
            cost = f_grad_shared(x, x_mask, y, y_mask)

            # do the update on parameters
            f_update(lrate)

            ud = time.time() - ud_start

            # check for bad numbers, usually we remove non-finite elements
            # and continue training - but not done here
            if numpy.isnan(cost) or numpy.isinf(cost):
                print('NaN detected')
                return 1., 1., 1.

            # verbose
            if numpy.mod(uidx, dispFreq) == 0:
                print('Epoch ', eidx, 'Update ', uidx,
                      'Cost ', cost, 'UD ', ud)

            # save the best model so far
            if numpy.mod(uidx, saveFreq) == 0:
                print('Saving...', end=' ')

                if best_p is not None:
                    params = best_p
                else:
                    params = unzip(tparams)
                numpy.savez(saveto, history_errs=history_errs, **params)
                cPickle.dump(model_options, open('%s.pkl' % saveto, 'wb'))
                print('Done')

            # generate some samples with the model and display them
            if numpy.mod(uidx, sampleFreq) == 0:
                # FIXME: random selection?
                for jj in xrange(numpy.minimum(5, x.shape[1])):
                    stochastic = True
                    sample, score = gen_sample(tparams,
                                               f_init,
                                               f_next,
                                               x[:, jj][:, None],
                                               model_options,
                                               trng=trng,
                                               k=1,
                                               maxlen=30,
                                               stochastic=stochastic,
                                               argmax=False)
                    print('Source ', jj, ': ', end=' ')
                    for vv in x[:, jj]:
                        if vv == 0:
                            break
                        if vv in worddicts_r[0]:
                            print(worddicts_r[0][vv], end=' ')
                        else:
                            print('UNK', end=' ')
                    print()
                    print('Truth ', jj, ' : ', end=' ')
                    for vv in y[:, jj]:
                        if vv == 0:
                            break
                        if vv in worddicts_r[1]:
                            print(worddicts_r[1][vv], end=' ')
                        else:
                            print('UNK', end=' ')
                    print()
                    print('Sample ', jj, ': ', end=' ')
                    if stochastic:
                        ss = sample
                    else:
                        score = score / numpy.array([len(s) for s in sample])
                        ss = sample[score.argmin()]
                    for vv in ss:
                        if vv == 0:
                            break
                        if vv in worddicts_r[1]:
                            print(worddicts_r[1][vv], end=' ')
                        else:
                            print('UNK', end=' ')
                    print()

            # validate model on validation set and early stop if necessary
            if numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                valid_errs = pred_probs(f_log_probs, prepare_data,
                                        model_options, valid_stream)
                valid_err = valid_errs.mean()
                history_errs.append(valid_err)

                if uidx == 0 or valid_err <= numpy.array(history_errs).min():
                    best_p = unzip(tparams)
                    bad_counter = 0
                if len(history_errs) > patience and valid_err >= \
                        numpy.array(history_errs)[:-patience].min():
                    bad_counter += 1
                    if bad_counter > patience:
                        print('Early Stop!')
                        estop = True
                        break

                if numpy.isnan(valid_err):
                    raise RuntimeError('NaN detected in validation error')

                print('Valid ', valid_err)

            # finish after this many updates
            if uidx >= finish_after:
                print('Finishing after %d iterations!' % uidx)
                estop = True
                break

        print('Seen %d samples' % n_samples)

        if estop:
            break

    if best_p is not None:
        zipp(best_p, tparams)

    use_noise.set_value(0.)
    valid_err = pred_probs(f_log_probs, prepare_data, model_options,
                           valid_stream).mean()

    print('Valid ', valid_err)

    params = copy.copy(best_p)
    numpy.savez(saveto,
                zipped_params=best_p,
                history_errs=history_errs,
                **params)

    return valid_err
