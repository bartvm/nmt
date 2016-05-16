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
import traceback

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
from utils import (load_params, init_tparams, zipp, name_dict,
                   unzip, itemlist, prepare_character_tensor)
import optimizers

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def train(experiment_id, data_base_path,
          model_options, data_options, validation_options,
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
          time_limit,
          save_freq,   # save the parameters after every saveFreq updates
          sample_freq,   # generate some samples after every sampleFreq
          reload_from=None):

    start_time = time.time()

    def join_data_base_path(data_base, options):
        for kk, vv in six.iteritems(options):
            if kk in ['src', 'trg', 'src_char_vocab', 'trg_char_vocab',
                      'src_word_vocab', 'trg_word_vocab', 'valid_src',
                      'valid_trg', 'trans_valid_src']:
                options[kk] = os.path.join(data_base, options[kk])

        return options

    data_options = join_data_base_path(data_base_path, data_options)
    validation_options = join_data_base_path(data_base_path,
                                             validation_options)

    worddicts_r, train_stream, valid_stream = load_data(**data_options)

    LOGGER.info('Building model')
    params = init_params(model_options)
    # reload parameters
    checkpoint_filename = '{}.checkpoint.npz'.format(experiment_id)
    model_option_filename = '{}.config.json'.format(experiment_id)
    best_filename = '{}.{}.best.npz'.format(experiment_id, saveto)
    if reload_from and os.path.exists(reload_from):
        LOGGER.info('Loading parameters from {}'.format(reload_from))
        params = load_params(reload_from, params)

    LOGGER.info('Initializing parameters')
    tparams = init_tparams(params)

    # use_noise is for dropout
    trng, use_noise, encoder_vars, decoder_vars, \
        opt_ret, costs = build_model(tparams, model_options)

    inps = encoder_vars + decoder_vars

    LOGGER.info('Building sampler')
    f_sample_inits, f_sample_nexts \
        = build_sampler(tparams, model_options, trng)

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
    f_grad_shared, f_update, optimizer_state = \
        getattr(optimizers, optimizer)(lr, tparams, grads, inps, cost)

    optimizer_state = name_dict(optimizer_state)

    # TODO set_value optimizer_state
    if reload_from and os.path.exists(reload_from):
        LOGGER.info('Loading optimizer state from {}'.format(reload_from))
        optimizer_state = load_params(reload_from, optimizer_state,
                                      theano_var=True)

    LOGGER.info('Optimization')

    log = Logger(filename='{}.log.jsonl.gz'.format(experiment_id))

    best_model = None
    best_score = 0
    bad_counter = 0

    uidx = 0
    uidx_restore = [0]
    estop = False
    if reload_from and os.path.exists(reload_from):
        rmodel = numpy.load(reload_from)
        if 'uidx' in rmodel:
            uidx_restore = rmodel['uidx']

    # hold a copy of parameters being used for validation
    valid_params = unzip(tparams)
    valid_opt_state = unzip(optimizer_state)
    valid_uidx = [uidx]

    # evaluation score will be stored into the following queue
    valid_ret_queue = queue.Queue()
    process_queue = queue.Queue()

    rt = None
    if eval_intv > 0:
        rt = prepare_validation_timer(valid_params, valid_opt_state,
                                      valid_uidx,
                                      process_queue,
                                      checkpoint_filename,
                                      model_option_filename,
                                      eval_intv, valid_ret_queue,
                                      **validation_options)

    def cancel_validation_process():
        # TODO delete the validation model file
        LOGGER.info('Now attempting to stop the timer')
        if rt:
            rt.stop()

        LOGGER.info('Please wait for terminating all child processes')
        while not process_queue.empty():
            proc = process_queue.get()
            if proc.returncode is None:
                # if the process is still running, terminate it
                proc.send_signal(signal.SIGINT)
                proc.communicate()
                # XXX other options to kill it
                # os.killpg(proc.pid, signal.SIGINT)
                # os.kill(proc.pid, signal.SIGINT)

    def _timer_signal_handler(signum, frame):

        cancel_validation_process()

        while not valid_ret_queue.empty():
            valid_ret_queue.get()

        raise Exception("Received an interrupt signal")

    signal.signal(signal.SIGINT, _timer_signal_handler)

    train_start = time.clock()

    try:
        if rt:
            rt.start()
        for epoch in xrange(0, max_epochs):
            n_samples = 0
            for xc, x, x_mask, \
                    yc, y, y_mask in train_stream.get_epoch_iterator():

                n_samples += len(x)

                uidx += 1
                if uidx < uidx_restore[0]:
                    continue

                x, x_mask, y, y_mask = x.T, x_mask.T, y.T, y_mask.T

                encoder_inps = [x, x_mask]
                decoder_inps = [y, y_mask]
                if model_options['use_character']:
                    xc, xc_mask = prepare_character_tensor(xc)
                    yc, yc_mask = prepare_character_tensor(yc)

                    encoder_inps += [xc, xc_mask]
                    decoder_inps += [yc, yc_mask]

                inps = encoder_inps + decoder_inps

                use_noise.set_value(1.)

                log_entry = {'iteration': uidx, 'epoch': epoch}

                # compute cost, grads and copy grads to shared variables
                update_start = time.clock()
                cost = f_grad_shared(*inps)
                f_update(lrate)

                log_entry['cost'] = float(cost)
                log_entry['average_source_length'] = \
                    float(x_mask.sum(0).mean())
                log_entry['average_target_length'] = \
                    float(y_mask.sum(0).mean())
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

                    if best_model is not None:
                        best_p, best_state, best_uidx = best_model
                        params = best_p
                        opt_state = best_state
                        save_at_uidx = best_uidx
                    else:
                        params = unzip(tparams)
                        opt_state = unzip(optimizer_state)
                        save_at_uidx = [uidx]

                    # save params to exp_id.npz and symlink model.npz to it
                    params_and_state = merge(params, opt_state,
                                             {'uidx': save_at_uidx})
                    save_params(params_and_state, best_filename)

                    # update validation parameter
                    valid_params = unzip(tparams, valid_params)
                    valid_opt_state = unzip(optimizer_state, valid_opt_state)
                    valid_uidx[0] = uidx

                # generate some samples with the model and display them
                if sample_freq > 0 and numpy.mod(uidx, sample_freq) == 0:
                    # FIXME: random selection?
                    log_entry['samples'] = []
                    for jj in xrange(numpy.minimum(5, x.shape[1])):
                        stats = [('source', ''), ('truth', ''), ('sample', ''),
                                 ('align_sample', ''), ('weights', ''),
                                 ('word_gates_src', ''),
                                 ('word_gates_trg', '')]
                        if model_options['use_character']:
                            stats += [('source (char)', ''),
                                      ('truth (char)', ''),
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
                                                    f_sample_inits,
                                                    f_sample_nexts,
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

                        if model_options['use_character'] and \
                           model_options['unk_gate']:
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

                        if model_options['init_decoder'] == 'adaptive':
                            word_weights = word_solutions['word_weights']
                            log_entry['samples'][-1]['weights'] = \
                                numpy.array2string(
                                    word_weights.squeeze()[
                                        :x_mask[:, jj].sum()],
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
                                log_entry['samples'][-1]['source (char)'] += \
                                    ' '

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

                            word_characters = \
                                word_solutions['character_samples']

                            assert len(word_sample) == len(word_characters), \
                                '%d:%d' % (len(word_sample),
                                           len(word_characters))

                            word_characters = \
                                word_characters[word_score.argmin()]

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
                            raise RuntimeError(('NaN detected in validation '
                                                'error of %s') % f_.name)

                # collect validation scores (e.g., BLEU) from the child thread
                if not valid_ret_queue.empty():

                    valid_ret = valid_ret_queue.get()
                    if len(valid_ret) == 1:
                        cancel_validation_process()

                        while not valid_ret_queue.empty():
                            valid_ret_queue.get()

                        raise valid_ret[0]
                    else:
                        assert len(valid_ret) == 2
                        ret_model, scores = valid_ret

                    valid_bleu = scores[0]
                    log_entry['validation_bleu'] = valid_bleu

                    if valid_bleu > best_score:
                        best_model = ret_model
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

                if time_limit > 0 and \
                   (time.time() - start_time > time_limit * 60):

                    LOGGER.info('Time limit {} mins is over'.format(
                        time_limit))
                    estop = True

                    cancel_validation_process()

                    break

                log.log(log_entry)

            LOGGER.info('Completed epoch, seen {} samples'.format(n_samples))

            if estop:
                log.log(log_entry)
                break

        if best_model is not None:
            assert len(best_model) == 3
            best_p, best_state, best_uidx = best_model
            zipp(best_p, tparams)
            zipp(best_state, optimizer_state)

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

        if not best_model:
            best_p = unzip(tparams)
            best_state = unzip(optimizer_state)
            best_uidx = [uidx]

        best_p = copy.copy(best_p)
        best_state = copy.copy(best_state)
        params_and_state = merge(best_p,
                                 best_state,
                                 {'uidx': best_uidx})
        save_params(params_and_state, best_filename)

    except Exception:
        LOGGER.error(traceback.format_exc())
        total_valid_err = -1.
    else:
        # XXX add something needed
        print('Training Done')
    finally:
        # XXX close resources
        if rt:
            rt.stop()

    return total_valid_err

if __name__ == "__main__":
    # Load the configuration file
    with io.open(sys.argv[1]) as f:
        config = json.load(f)
    if len(sys.argv) == 3:
        data_base_path = os.path.realpath(sys.argv[2])
    else:
        data_base_path = os.getcwd()

    # Create unique experiment ID and backup config file
    experiment_id = binascii.hexlify(os.urandom(3)).decode()
    shutil.copyfile(sys.argv[1], '{}.config.json'.format(experiment_id))
    train(experiment_id, data_base_path, config['model'], config['data'],
          config['validation'],
          **merge(config['training'], config['management']))
