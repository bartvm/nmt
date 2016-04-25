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
import Queue as queue
import numpy
from collections import OrderedDict

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
                   unzip, itemlist, prepare_character_tensor)
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
    trng, use_noise, encoder_vars, decoder_vars, \
        opt_ret, costs = build_model(tparams, model_options)

    inps = encoder_vars + decoder_vars

    LOGGER.info('Building sampler')
    f_enc_init, f_sample_nexts = build_sampler(tparams, model_options, trng)

    # before any regularizer
    LOGGER.info('Building functions to compute log prob')
    f_log_probs = [
        theano.function(inps, cost_, name='f_log_probs_%s' % cost_.name,
                        on_unused_input='ignore')
        for cost_ in costs
    ]

    assert len(costs) >= 1

    cost = costs[0]
    for cost_ in costs[1:]:
        cost += cost_

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
        x_mask = encoder_vars[1]
        y_mask = decoder_vars[1]

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
    valid_ret_queue = queue.Queue()
    process_queue = queue.Queue()

    if eval_intv > 0:
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

        for xc, x, x_mask, yc, y, y_mask in train_stream.get_epoch_iterator():
            n_samples += len(x)
            x, x_mask, y, y_mask = x.T, x_mask.T, y.T, y_mask.T

            encoder_inps = [x, x_mask]
            decoder_inps = [y, y_mask]
            if model_options['use_character']:
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

            use_noise.set_value(1.)

            uidx += 1
            log_entry = {'iteration': uidx, 'epoch': eidx}

            # compute cost, grads and copy grads to shared variables
            update_start = time.clock()
            cost = f_grad_shared(*inps)
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
            if sample_freq > 0 and numpy.mod(uidx, sample_freq) == 0:
                # FIXME: random selection?
                log_entry['samples'] = []
                for jj in xrange(numpy.minimum(5, x.shape[1])):
                    stats = [('source', ''), ('truth', ''), ('sample', ''),
                             ('align_sample', ''), ('weights', ''),
                             ('word_gates_src', ''), ('word_gates_trg', '')]
                    if model_options['use_character']:
                        stats += [('source (char)', ''), ('truth (char)', ''),
                                  ('sample (char)', '')]
                    log_entry['samples'].append(OrderedDict(stats))

                    sample_encoder_inps = [x[:, jj][:, None],
                                           x_mask[:, jj][:, None]]
                    if model_options['use_character']:
                        sample_encoder_inps += [
                            xc[:, :, jj][:, :, None],
                            xc_mask[:, :, jj][:, :, None]
                        ]

                    word_solutions = gen_sample(tparams,
                                                f_enc_init, f_sample_nexts,
                                                sample_encoder_inps,
                                                model_options,
                                                trng=trng,
                                                k=12,
                                                max_sent_len=100,
                                                max_word_len=30,
                                                argmax=False)

                    word_sample = word_solutions['samples']
                    word_alignment = word_solutions['alignments']
                    word_score = word_solutions['scores']

                    word_score = word_score / numpy.array(
                        [len(s) for s in word_sample])
                    ss = word_sample[word_score.argmin()]
                    word_alignment = word_alignment[word_score.argmin()]

                    if model_options['use_character']:
                        '''
                        word_src_gates = word_solutions['word_src_gates']
                        log_entry['samples'][-1]['word_gates_src'] = \
                            numpy.array2string(
                                word_src_gates[:x_mask[:, jj].sum(), :].T,
                                precision=2, max_line_width=500,
                                suppress_small=True)

                        word_trg_gates = word_solutions['word_trg_gates']
                        word_trg_gates = numpy.array(
                            word_trg_gates[word_score.argmin()])

                        log_entry['samples'][-1]['word_gates_trg'] = \
                            numpy.array2string(
                                word_trg_gates.T,
                                precision=2, max_line_width=500,
                                suppress_small=True)
                        '''

                    if model_options['init_decoder'] == 'adaptive':
                        word_weights = word_solutions['word_weights']
                        log_entry['samples'][-1]['weights'] = \
                            numpy.array2string(
                                word_weights.squeeze()[:x_mask[:, jj].sum()],
                                precision=3, max_line_width=500,
                                suppress_small=True)

                    for vv in x[:, jj]:
                        if vv == 0:
                            break
                        if vv in worddicts_r[1]:
                            token = worddicts_r[1][vv]
                        else:
                            token = UNK_TOKEN
                        log_entry['samples'][-1]['source'] += token + ' '
                    for vv in y[:, jj]:
                        if vv == 0:
                            break
                        if vv in worddicts_r[3]:
                            token = worddicts_r[3][vv]
                        else:
                            token = UNK_TOKEN
                        log_entry['samples'][-1]['truth'] += token + ' '
                    for tidx, vv in enumerate(ss):
                        if vv == 0:
                            break
                        if vv in worddicts_r[3]:
                            token = worddicts_r[3][vv]
                        else:
                            token = UNK_TOKEN

                        assert tidx >= 0 and tidx < len(word_alignment), \
                            '%d\t%d' % (tidx, len(word_alignment))

                        num_src_words = x_mask[:, jj].sum()-1
                        align_src_word_idx = \
                            (word_alignment[tidx][:num_src_words]).argmax()
                        if token == UNK_TOKEN:
                            aligned_token = '%s_<%d>' % \
                                (worddicts_r[1][x[align_src_word_idx, jj]],
                                 align_src_word_idx)
                        else:
                            aligned_token = '%s_<%d>' % \
                                (token, align_src_word_idx)

                        log_entry['samples'][-1]['sample'] += token + ' '
                        log_entry['samples'][-1]['align_sample'] \
                            += aligned_token + ' '

                    if model_options['use_character']:
                        num_chars, num_words, num_samples = xc.shape
                        for widx in xrange(num_words):
                            if xc_mask[:, widx, jj].sum() == 0:
                                break
                            for cidx in xrange(num_chars):
                                cc = xc[cidx, widx, jj]
                                if cc == 0:
                                    break
                                if cc in worddicts_r[0]:
                                    token = worddicts_r[0][cc]
                                else:
                                    token = UNK_TOKEN
                                log_entry['samples'][-1]['source (char)'] \
                                    += token
                            log_entry['samples'][-1]['source (char)'] += ' '

                        num_chars, num_words, num_samples = yc.shape
                        for widx in xrange(num_words):
                            if yc_mask[:, widx, jj].sum() == 0:
                                break
                            for cidx in xrange(num_chars):
                                cc = yc[cidx, widx, jj]
                                if cc == 0:
                                    break
                                if cc in worddicts_r[2]:
                                    token = worddicts_r[2][cc]
                                else:
                                    token = UNK_TOKEN
                                log_entry['samples'][-1]['truth (char)'] \
                                    += token
                            log_entry['samples'][-1]['truth (char)'] += ' '

                        word_characters = word_solutions['character_samples']

                        assert len(word_sample) == len(word_characters), \
                            '%d:%d' % (len(word_sample), len(word_characters))

                        word_characters = word_characters[word_score.argmin()]
                        for word in word_characters:
                            token = ''
                            for character in word:
                                if character == 0:
                                    break
                                if character in worddicts_r[2]:
                                    token += worddicts_r[2][character]
                                else:
                                    token += UNK_TOKEN

                            log_entry['samples'][-1]['sample (char)'] \
                                += token + ' '

            # validate model on validation set and early stop if necessary
            if numpy.mod(uidx, valid_freq) == 0:
                use_noise.set_value(0.)
                valid_errs = [
                    numpy.mean(
                        pred_probs(f_,
                                   model_options,
                                   valid_stream))
                    for f_ in f_log_probs
                ]

                for f_, err_ in zip(f_log_probs, valid_errs):
                    log_entry['validation_%s' % f_.name] = float(err_)

                for f_, err_ in zip(f_log_probs, valid_errs):
                    if not numpy.isfinite(err_):
                        raise RuntimeError(('NaN detected in validation error'
                                            ' of %s') % f_.name)

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
    valid_errs = [
        numpy.mean(
            pred_probs(f_,
                       model_options,
                       valid_stream))
        for f_ in f_log_probs
    ]

    total_valid_err = numpy.sum(valid_errs)

    if not best_p:
        best_p = unzip(tparams)

    params = copy.copy(best_p)
    save_params(params, model_filename, saveto_filename)

    rt.stop()

    return total_valid_err

if __name__ == "__main__":
    # Load the configuration file
    with io.open(sys.argv[1]) as f:
        config = json.load(f)
    # Create unique experiment ID and backup config file
    experiment_id = binascii.hexlify(os.urandom(3)).decode()
    shutil.copyfile(sys.argv[1], '{}.config.json'.format(experiment_id))
    train(experiment_id, config['model'], config['data'], config['validation'],
          **merge(config['training'], config['management']))
