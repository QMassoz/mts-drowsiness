local M = {}

function M.parse(arg)
	local cmd = torch.CmdLine()
	cmd:text()
	cmd:text('Supervised learning workflow')
	cmd:text()
	cmd:text('Options:')
	cmd:text('----- General options -----')
	cmd:option('-gen',		'data/processed/',	'path to saved generated files')
	cmd:option('-dataset',		'ieye2eld',		'options: ieye2eld or eld2multid')
	cmd:option('-cache',		'nil',			'cache .t7 file containing train and valid sets, set to "-dataset" by default')
	cmd:option('-seed',			1,				'manual set RNG seed')
	cmd:option('-backend',		'cudnn',		'options: cudnn | cunn | cpu')
	cmd:option('-cudnn',		'deterministic','cudnn convolution mode: fastest | default | deterministic')
	cmd:option('-progress',		'true',			'print progress bar')
	cmd:option('-nthreads',		1,				'number of threads for asynchronous preprocessing')
	
	cmd:text('-- Testing options --')
	cmd:option('-testModel',	'nil',			'path to model that will be tested (on training, validation, and test set)')
	cmd:option('-testOnly',		'false',		'evaluate the model(s) on the test set only')

	cmd:text('----- Optimization options -----')
	cmd:option('-optimizer',	'rmsprop',		'options: sgd | rmsprop | adam')
	cmd:option('-lr',			0.001,			'learning rate')
	cmd:option('-lrd',			0,				'learning rate decay (sgd)')
	cmd:option('-lr_epoch',		0,				'number of epochs without improvements (of validation error) before halving the lr (sgd), pass 0 to disable')
	cmd:option('-wd',			0,				'weight decay')
	cmd:option('-alpha',		0.95,			'alpha value (rmsprop)')
	cmd:option('-batch_size',	32,				'mini-batch size (1 = pure stochastic)')
	cmd:option('-max_epochs',	30,				'number of full passes through the training data')
	cmd:option('-min_train_error',	'nil',		'early stop if the training error goes below this threshold')
	cmd:option('-earlystop',	'nil',			'early stop if no improvements for this number of epochs')
	cmd:option('-crossval_fold',1,				'fold number of the crossvalidation')

	cmd:text('----- Data augmentation options -----')
	cmd:option('-flip',			'true',			'horizontal flipping to augment data (ieye2eld)')
	cmd:option('-augment',		'false',		'random sampling to augment data: true, false, random, balance (eld2multid)')
	cmd:option('-n_augment',	256,			'(dataset specific) numeric value for data augmentation')

	cmd:text('----- Model options -----')
	cmd:option('-dropout',		0,				'dropout probability')
	cmd:option('-layers',		'nil',			'model architecture (dataset dependent)')
	cmd:option('-last_pooling',	'max',			'global pooling before FC layer: max | average')
	cmd:option('-bn_momentum',	'nil',			'batch normalization momentum')

	cmd:text('----- Criterion options -----')
	cmd:option('-weight_eval',	'false',		'flag to weight the eval criterion or not')

	cmd:text('----- Saving options -----')
	cmd:option('-save_file',	'models/current.net',	'file path to save the best iteration model')
	cmd:text()

	opt = cmd:parse(arg)

	-- Convert 'false'/'true' to false/true
	opt.flip = opt.flip ~= 'false'
	opt.testOnly = opt.testOnly ~= 'false'
	opt.skip = opt.skip ~= 'false'
	if opt.augment == 'false' then
		opt.augment = false
	elseif opt.augment == 'true' then
		opt.augment = true
	end
	opt.progress = opt.progress == 'true'
	opt.rmv_incomplete_batch = opt.rmv_incomplete_batch ~= 'false'

	-- Convert 'nil' to nil (not useful anymore but whatever)
	for k,v in pairs(opt) do
		if v == 'nil' then
			opt[k] = nil
		end
	end
	

	return opt
end

return M
