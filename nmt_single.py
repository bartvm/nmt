import binascii
import copy
import io
import json
import logging
import os
import shutil
import sys
import time
import signal
import Queue
import numpy

import six
import theano
from mimir import Logger
from theano import tensor
from six.moves import xrange
from toolz.dicttoolz import merge

from data_iterator import UNK_TOKEN, load_data
from nmt_base import (pred_probs, build_model, save_params,
                      build_sampler, init_params, gen_sample,
                      prepare_validation_timer)
from utils import (load_params, init_tparams, zipp,
                   unzip, itemlist)
import optimizers

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def train(experiment_id, model_options, data_options, validation_options,
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
          eval_intv,    # time interval for evaluation in minutes
          save_freq,   # save the parameters after every saveFreq updates
          sample_freq,   # generate some samples after every sampleFreq
          reload_=False):

    worddicts_r, train_stream, valid_stream = load_data(**data_options)

    LOGGER.info('Building model')
    params = init_params(model_options)
    # reload parameters
    model_filename = '{}.model.npz'.format(experiment_id)
    model_option_filename = '{}.config.json'.format(experiment_id)
    saveto_filename = '{}.npz'.format(saveto)
    if reload_ and os.path.exists(saveto_filename):
        LOGGER.info('Loading parameters from {}'.format(saveto_filename))
        params = load_params(saveto_filename, params)

    LOGGER.info('Initializing parameters')
    tparams = init_tparams(params)

    # use_noise is for dropout
    trng, use_noise, \
        x, x_mask, y, y_mask, \
        opt_ret, \
        cost = \
        build_model(tparams, model_options)
    inps = [x, x_mask, y, y_mask]

    LOGGER.info('Building sampler')
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

    LOGGER.info('Optimization')

    log = Logger(filename='{}.log.jsonl.gz'.format(experiment_id))

    # evaluation score will be stored into the following queue
    valid_ret_queue = Queue.Queue()
    process_queue = Queue.Queue()

    rt = prepare_validation_timer(tparams, process_queue, model_filename,
                                  model_option_filename,
                                  eval_intv, valid_ret_queue,
                                  **validation_options)
    rt.start()

    def _timer_signal_handler(signum, frame):
        LOGGER.info('Received SIGINT')
        LOGGER.info('Now attempting to stop the timer')
        rt.stop()

        LOGGER.info('Please wait for terminating all child processes')
        while not process_queue.empty():
            proc = process_queue.get()
            if proc.poll() is None:     # check if the process has terminated
                # child process is still working
                # LOGGER.info('Attempt to kill', proc.pid)
                # terminate it by sending an interrupt signal
                proc.send_signal(signal.SIGINT)

                # wait for child process while avoiding deadlock
                # ignore outputs
                proc.communicate()

        sys.exit(130)

    signal.signal(signal.SIGINT, _timer_signal_handler)

    train_start = time.clock()
    best_p = None
    best_score = 0
    bad_counter = 0

    uidx = 0
    estop = False

    for eidx in xrange(max_epochs):
        n_samples = 0

        for x, x_mask, y, y_mask in train_stream.get_epoch_iterator():
            n_samples += len(x)
            x, x_mask, y, y_mask = x.T, x_mask.T, y.T, y_mask.T

            use_noise.set_value(1.)

            uidx += 1
            log_entry = {'iteration': uidx, 'epoch': eidx}

            # compute cost, grads and copy grads to shared variables
            update_start = time.clock()
            cost = f_grad_shared(x, x_mask, y, y_mask)
            f_update(lrate)

            log_entry['cost'] = float(cost)
            log_entry['average_source_length'] = float(x_mask.sum(0).mean())
            log_entry['average_target_length'] = float(y_mask.sum(0).mean())
            log_entry['update_time'] = time.clock() - update_start
            log_entry['train_time'] = time.clock() - train_start

            # check for bad numbers, usually we remove non-finite elements
            # and continue training - but not done here
            if not numpy.isfinite(cost):
                LOGGER.error('NaN detected')
                return 1., 1., 1.

            # save the best model so far
            if numpy.mod(uidx, save_freq) == 0:
                LOGGER.info('Saving best model so far')

                if best_p is not None:
                    params = best_p
                else:
                    params = unzip(tparams)

                # save params to exp_id.npz and symlink model.npz to it
                save_params(params, model_filename, saveto_filename)

            # generate some samples with the model and display them
            if numpy.mod(uidx, sample_freq) == 0:
                # FIXME: random selection?
                log_entry['samples'] = []
                for jj in xrange(numpy.minimum(5, x.shape[1])):
                    log_entry['samples'].append({'source': '', 'truth': '',
                                                 'sample': ''})
                    stochastic = True
                    sample, _, score = gen_sample(tparams,
                                                  f_init,
                                                  f_next,
                                                  x[:, jj][:, None],
                                                  model_options,
                                                  trng=trng,
                                                  k=1,
                                                  maxlen=30,
                                                  stochastic=stochastic,
                                                  argmax=False)
                    for vv in x[:, jj]:
                        if vv == 0:
                            break
                        if vv in worddicts_r[0]:
                            token = worddicts_r[0][vv]
                        else:
                            token = UNK_TOKEN
                        log_entry['samples'][-1]['source'] += token + ' '
                    for vv in y[:, jj]:
                        if vv == 0:
                            break
                        if vv in worddicts_r[1]:
                            token = worddicts_r[1][vv]
                        else:
                            token = UNK_TOKEN
                        log_entry['samples'][-1]['truth'] += token + ' '
                    if stochastic:
                        ss = sample
                    else:
                        score = score / numpy.array([len(s) for s in sample])
                        ss = sample[score.argmin()]
                    for vv in ss:
                        if vv == 0:
                            break
                        if vv in worddicts_r[1]:
                            token = worddicts_r[1][vv]
                        else:
                            token = UNK_TOKEN
                        log_entry['samples'][-1]['sample'] += token + ' '

            # validate model on validation set and early stop if necessary
            if numpy.mod(uidx, valid_freq) == 0:
                use_noise.set_value(0.)
                valid_errs = pred_probs(f_log_probs,
                                        model_options, valid_stream)
                valid_err = valid_errs.mean()
                log_entry['validation_cost'] = float(valid_err)

                if not numpy.isfinite(valid_err):
                    raise RuntimeError('NaN detected in validation error')

            # collect validation scores (e.g., BLEU) from the child thread
            if not valid_ret_queue.empty():

                (ret_model, scores) = valid_ret_queue.get()

                valid_bleu = scores[0]
                # LOGGER.info('BLEU on the validation set: %.2f' % valid_bleu)
                log_entry['validation_bleu'] = valid_bleu

                if valid_bleu > best_score:
                    best_p = ret_model
                    best_score = valid_bleu
                    bad_counter = 0
                else:
                    bad_counter += 1
                    if bad_counter > patience:
                        estop = True
                        break

            # finish after this many updates
            if uidx >= finish_after:
                LOGGER.info('Finishing after {} iterations'.format(uidx))
                estop = True
                break

            log.log(log_entry)

        LOGGER.info('Completed epoch, seen {} samples'.format(n_samples))

        if estop:
            log.log(log_entry)
            break

    if best_p is not None:
        zipp(best_p, tparams)

    use_noise.set_value(0.)
    LOGGER.info('Calculating validation cost')
    valid_err = pred_probs(f_log_probs, model_options,
                           valid_stream).mean()

    if not best_p:
        best_p = unzip(tparams)

    params = copy.copy(best_p)
    save_params(params, model_filename, saveto_filename)

    rt.stop()

    return valid_err

if __name__ == "__main__":
    # Load the configuration file
    with io.open(sys.argv[1]) as f:
        config = json.load(f)
    # Create unique experiment ID and backup config file
    experiment_id = binascii.hexlify(os.urandom(3)).decode()
    shutil.copyfile(sys.argv[1], '{}.config.json'.format(experiment_id))
    train(experiment_id, config['model'], config['data'], config['validation'],
          **merge(config['training'], config['management']))
