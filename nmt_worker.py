from nmt_base import *
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
          n_words_src=3000,  # source vocabulary size
          n_words=3000,  # target vocabulary size
          maxlen=15,  # maximum length of the description
          optimizer='rmsprop',
          batch_size=64,
          valid_batch_size=64,
	  ctx_len_emb = 5,
          train_len = 1,
          saveto='model.npz',
          validFreq=1000,
          saveFreq=1000,   # save the parameters after every saveFreq updates
          sampleFreq=100,   # generate some samples after every sampleFreq
          datasets=[
              '/u/goyalani/dl4mt-material/europarl-v7.fr-en.en.tok',
              '/u/goyalani/dl4mt-material/europarl-v7.fr-en.fr.tok'],
          valid_datasets=['/u/goyalani/dl4mt-material/newstest2011.en.tok',
                          '/u/goyalani/dl4mt-material/newstest2011.fr.tok'],
          dictionaries=[
              '/u/goyalani/dl4mt-material/europarl-v7.fr-en.en.tok.pkl',
              '/u/goyalani/dl4mt-material/europarl-v7.fr-en.fr.tok.pkl'],  
	
          use_dropout=False,
	  overwrite=False,
	  valid_sync=False,
          reload_=False):

    worker = Worker(control_port=5567)
    

    # Model options
    model_options = locals().copy()

    # load dictionaries and invert them
    worddicts = [None] * len(dictionaries)
    worddicts_r = [None] * len(dictionaries)
    for ii, dd in enumerate(dictionaries):
        with open(dd, 'rb') as f:
	    worddicts[ii] = load_dict(dd)
        worddicts_r[ii] = dict()
    for kk, vv in six.iteritems(worddicts[ii]):
        worddicts_r[ii][vv] = kk

    # reload options
    if reload_ and os.path.exists(saveto):
        with open('%s.pkl' % saveto, 'rb') as f:
            models_options = pkl.load(f)

    print 'Loading data'
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
                              

    print 'Building model'
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        params = load_params(saveto, params)

    tparams = init_tparams(params)
    worker.init_shared_params(tparams.values(), param_sync_rule=EASGD(0.5))
    print 'Params init done'

    # use_noise is for dropout
    trng, use_noise, \
        x, x_mask, y, y_mask, \
        opt_ret, \
	cost, \
        unk_ctx = \
        build_model(tparams, model_options)
    inps = [x, x_mask, y, y_mask, unk_ctx]

    print 'Buliding sampler'
    f_init, f_next  = build_sampler(tparams, model_options, trng)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile)
    print 'Done'

    cost = cost.mean()

    # apply L2 regularization on weights
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # regularize the alpha weights
    if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * (
            (tensor.cast(y_mask.sum(0)//x_mask.sum(0), 'float32')[:, None] -
             opt_ret['dec_alphas'].sum(0))**2).sum(1).mean()
        cost += alpha_reg

    # after all regularizers - compile the computational graph for cost
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=profile)
    print 'Done'

    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print 'Done'

    # apply gradient clipping here
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        grads = new_grads

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)
    print 'Done'

    print 'Optimization'
    best_p = None

    # Training data iterator!
    def train_iter():
        while True:
		for x, y in train_stream.get_epoch_iterator():
                	x, x_mask, y, y_mask = prepare_data(x, y, maxlen=maxlen)
                	yield x, x_mask, y, y_mask
			
				
    train_it = train_iter()

    # Making sure that the worker start training with the most recent params
    worker.copy_to_local()	

    while True:
        n_samples = 0
        step = worker.send_req('next')
        print step
	if step == 'train':
            	use_noise.set_value(1.)
		for i in xrange(train_len):
			x, x_mask, y, y_mask = next(train_it)
			if x is None:
                		print 'Minibatch with zero sample under length ', maxlen
                		continue
			if len(x) == 0:
                		print 'No input is present ', x
                		continue
		
                        # For this particular, mini-batch, it 
                        # outputs the context matrix for all the 
                        # unknown words.	
                        unk_ctx = get_ctx_matrix(x, ctx_len_emb)
			
                        if unk_ctx is None:
        			print 'No unknown word is present ', unk_ctx
                		continue
			if len(unk_ctx) == 0:
                		print 'No unknown word is present ', unk_ctx
                		continue
		
			# compute cost, grads and copy grads to shared variables    
			cost = f_grad_shared(x, x_mask, y, y_mask, unk_ctx)
	        
			# do the update on parameters
                	f_update(lrate)
		
		print 'Train cost:', cost
   
		step = worker.send_req(dict(done=train_len))
            	print "Syncing with global params"
            	worker.sync_params(synchronous=True) 
		
	if step == 'valid':
            if valid_sync:
                worker.copy_to_local()
            use_noise.set_value(0.)
	    valid_errs = pred_probs(f_log_probs, 
                                    prepare_data, 
                                    ctx_len_emb,
                                    model_options, 
                                    valid)
	    valid_err = valid_errs.mean()
            res = worker.send_req(dict(test_err=float(valid_err),
                                       valid_err=float(valid_err)))

            if res == 'best':
                best_p = unzip(tparams)

            print ('Valid ', valid_err,
                   'Test ', valid_err)
            if valid_sync:
                worker.copy_to_local()

        if step == 'stop':
            break

    # Release all shared ressources.
    worker.close()
	
    print 'Saving...'

    if best_p is not None:
    	params = best_p
    else:
        params = unzip(tparams)
	
    use_noise.set_value(0.)

    if saveto:
	numpy.savez(saveto, **best_p)
	print 'model saved'	

    params = copy.copy(best_p)
    numpy.savez(saveto, zipped_params=best_p, **params)

def main(job_id, params):
    print params
    basedir = params['basedir'][0];
    valid_err = train(saveto=params['model'][0],
          reload_= params['reload'][0],
          dim = params['dim'][0],
          n_words = params['n-words'][0],
          n_words_src = params['n-words'][0],
          decay_c = params['decay-c'][0],
          clip_c = params['clip-c'][0],
          lrate = params['learning-rate'][0],
          optimizer = params['optimizer'][0],
          maxlen = 80,
          batch_size = 64,
          valid_batch_size = 64,
          datasets = [
	    ('%s/wmt16/'
             'wmt16.de-en.tok.true.clean.shuf.en' %basedir),
            ('%s/wmt16/'
             'wmt16.de-en.tok.true.clean.shuf.de' %basedir)
	  ],
          valid_datasets=[
	    '%s/wmt16/newstest2011.en.tok' %basedir,
            '%s/wmt16/newstest2011.fr.tok' %basedir
          ],
          dictionaries=[
		('%s/wmt16/'
                  'wmt16.de-en.vocab.en' %basedir),
                ('%s/wmt16/'
                  'wmt16.de-en.vocab.de' %basedir)
          ],
          validFreq=500000,
          dispFreq=1,
          saveFreq=100,
          sampleFreq=50,
          use_dropout=params['use-dropout'][0])
    return


if __name__ == '__main__':
    basedir = '/data/lisatmp4/vanmerb'
    mode=sys.argv[1]
    main(0, {
        'mode' : [mode],
        'basedir' : ['/data/lisatmp4/vanmerb'],
        'model': ['%s/model/model_attention.npz'%basedir],
        'dim': [124],
        'n-words': [3000],
        'optimizer': ['adadelta'],
        'decay-c': [0.],
        'clip-c': [1.],
        'use-dropout': [False],
        'learning-rate': [0.001],
        'reload': [False]})

