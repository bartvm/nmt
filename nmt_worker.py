from __future__ import print_function

import copy
import os
import sys

import numpy
import six
import theano
import yaml
from platoon.channel import Worker
from platoon.param_sync import EASGD
from six.moves import xrange, cPickle
from theano import tensor
from toolz.dicttoolz import merge

import optimizers
from nmt_base import (init_params, build_model, build_sampler,
                      pred_probs, load_data)
from utils import unzip, init_tparams, load_params, itemlist


def train(model_options, data_options,
          patience,  # early stopping patience
          max_epochs,
          finish_after,  # finish after this many updates
          disp_freq,
          decay_c,  # L2 regularization penalty
          alpha_c,  # alignment regularization
          clip_c,  # gradient clipping threshold
          lrate,  # learning rate
          optimizer,
          saveto,
          valid_freq,
          train_len,
          valid_sync,
          save_freq,   # save the parameters after every saveFreq updates
          sample_freq,   # generate some samples after every sampleFreq
          reload_=False):

    worker = Worker(control_port=5567)

    # reload options
    if reload_ and os.path.exists(saveto):
        with open('%s.pkl' % saveto, 'rb') as f:
            model_options = cPickle.load(f, encoding='latin1')

    worddicts_r, train_stream, valid_stream = load_data(**data_options)

    print('Building model')
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        params = load_params(saveto, params)

    tparams = init_tparams(params)
    worker.init_shared_params(tparams.values(), param_sync_rule=EASGD(0.5))
    print('Params init done')

    # use_noise is for dropout
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
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # regularize the alpha weights
    if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * ((tensor.cast(
            y_mask.sum(0) // x_mask.sum(0), 'float32')[:, None] -
            opt_ret['dec_alphas'].sum(0)) ** 2).sum(1).mean()
        cost += alpha_reg

    # Not used?
    # after all regularizers - compile the computational graph for cost
    # print('Building f_cost...', end=' ')
    # f_cost = theano.function(inps, cost, profile=False)
    # print('Done')

    print('Computing gradient...', end=' ')
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print('Done')

    # apply gradient clipping here
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g ** 2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c ** 2), g / tensor.sqrt(
                g2) * clip_c, g))
        grads = new_grads

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print('Building optimizers...', end=' ')
    f_grad_shared, f_update = getattr(optimizers, optimizer)(lr, tparams,
                                                             grads, inps, cost)
    print('Done')

    print('Optimization')
    best_p = None

    # Training data iterator!
    def train_iter():
        while True:
            for x, x_mask, y, y_mask in train_stream.get_epoch_iterator():
                yield x.T, x_mask.T, y.T, y_mask.T

    train_it = train_iter()

    # Making sure that the worker start training with the most recent params
    worker.copy_to_local()

    while True:
        step = worker.send_req('next')
        print(step)
        if step == 'train':
            use_noise.set_value(1.)
            for i in xrange(train_len):
                x, x_mask, y, y_mask = next(train_it)

                cost = f_grad_shared(x, x_mask, y, y_mask)

                f_update(lrate)

            print('Train cost:', cost)

            step = worker.send_req(dict(done=train_len))
            print("Syncing with global params")
            worker.sync_params(synchronous=True)

        if step == 'valid':
            if valid_sync:
                worker.copy_to_local()
            use_noise.set_value(0.)
            valid_errs = pred_probs(f_log_probs,
                                    model_options,
                                    valid_stream)
            valid_err = valid_errs.mean()
            res = worker.send_req(dict(test_err=float(valid_err),
                                       valid_err=float(valid_err)))

            if res == 'best':
                best_p = unzip(tparams)

            print(('Valid ', valid_err,
                   'Test ', valid_err))
            if valid_sync:
                worker.copy_to_local()

        if step == 'stop':
            break

    # Release all shared ressources.
    worker.close()

    print('Saving...')

    if best_p is not None:
        params = best_p
    else:
        params = unzip(tparams)

    use_noise.set_value(0.)

    if saveto:
        numpy.savez(saveto, **best_p)
        print('model saved')

    params = copy.copy(best_p)
    numpy.savez(saveto, zipped_params=best_p, **params)


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        config = yaml.load(f)
    train(config['model'], config['data'],
          **merge(config['training'], config['management'], config['multi']))
