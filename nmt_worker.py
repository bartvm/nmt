from __future__ import print_function

import logging
import os
import time

import numpy
import six
import theano
from mimir import RemoteLogger
from platoon.channel import Worker
from platoon.param_sync import EASGD
from six.moves import xrange
from theano import tensor
from toolz.dicttoolz import merge

import optimizers
from nmt_base import (init_params, build_model, build_sampler, save_params,
                      pred_probs, load_data)
from utils import unzip, init_tparams, load_params, itemlist

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def train(worker, model_options, data_options,
          patience,  # early stopping patience
          max_epochs,
          finish_after,  # finish after this many updates
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
          control_port,
          batch_port,
          reload_):

    LOGGER.info('Connecting to data socket and loading validation data')
    worker.init_mb_sock(batch_port)
    _, _, valid_stream = load_data(**data_options)

    LOGGER.info('Building model')
    params = init_params(model_options)
    # reload parameters
    experiment_id = worker.send_req('experiment_id')
    model_filename = '{}.model.npz'.format(experiment_id)
    saveto_filename = '{}.npz'.format(saveto)
    if reload_ and os.path.exists(saveto_filename):
        LOGGER.info('Loading parameters from {}'.format(saveto_filename))
        params = load_params(saveto_filename, params)

    LOGGER.info('Initializing parameters')
    tparams = init_tparams(params)
    alpha = worker.send_req('alpha')
    worker.init_shared_params(tparams.values(), param_sync_rule=EASGD(alpha))

    # use_noise is for dropout
    trng, use_noise, \
        x, x_mask, y, y_mask, \
        opt_ret, \
        cost = \
        build_model(tparams, model_options)
    inps = [x, x_mask, y, y_mask]

    LOGGER.info('Building sampler')
    worker.start_compilation()
    f_init, f_next = build_sampler(tparams, model_options, trng)

    # before any regularizer
    LOGGER.info('Building f_log_probs')
    f_log_probs = theano.function(inps, cost, profile=False)

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
    # LOGGER.info('Building f_cost')
    # f_cost = theano.function(inps, cost, profile=False)

    LOGGER.info('Computing gradient')
    grads = tensor.grad(cost, wrt=itemlist(tparams))

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
    LOGGER.info('Building optimizers')
    f_grad_shared, f_update = getattr(optimizers, optimizer)(lr, tparams,
                                                             grads, inps, cost)
    worker.finish_compilation()

    LOGGER.info('Optimization')

    log = RemoteLogger()
    train_start = time.clock()
    best_p = None

    # Making sure that the worker start training with the most recent params
    worker.copy_to_local()

    uidx = 0
    while True:
        step = worker.send_req('next')
        LOGGER.info('Received command: {}'.format(step))
        if step == 'train':
            use_noise.set_value(1.)
            for i in xrange(train_len):
                x, x_mask, y, y_mask = worker.recv_mb()

                uidx += 1
                log_entry = {'iteration': uidx}

                # compute cost, grads and copy grads to shared variables
                update_start = time.clock()
                cost = f_grad_shared(x, x_mask, y, y_mask)
                f_update(lrate)

                log_entry['cost'] = float(cost)
                log_entry['average_source_length'] = \
                    float(x_mask.sum(0).mean())
                log_entry['average_target_length'] = \
                    float(y_mask.sum(0).mean())
                log_entry['update_time'] = time.clock() - update_start
                log_entry['train_time'] = time.clock() - train_start
                log_entry['time'] = time.time()
                log.log(log_entry)

            step = worker.send_req({'done': train_len})
            LOGGER.info("Syncing with global params")
            worker.sync_params(synchronous=True)

        if step == 'valid':
            if valid_sync:
                worker.copy_to_local()
            use_noise.set_value(0.)
            valid_errs = pred_probs(f_log_probs, model_options, valid_stream)
            valid_err = float(valid_errs.mean())
            res = worker.send_req({'valid_err': valid_err})
            log.log({'validation_cost': valid_err,
                     'train_time': time.clock() - train_start,
                     'time': time.time()})

            if res == 'best' and saveto:
                best_p = unzip(tparams)
                save_params(best_p, model_filename, saveto_filename)

            if valid_sync:
                worker.copy_to_local()

        if step == 'stop':
            break

    # Release all shared ressources.
    worker.close()


if __name__ == "__main__":
    LOGGER.info('Connecting to worker')
    worker = Worker(control_port=5567)
    LOGGER.info('Retrieving configuration')
    config = worker.send_req('config')
    train(worker, config['model'], config['data'],
          **merge(config['training'], config['management'], config['multi']))
