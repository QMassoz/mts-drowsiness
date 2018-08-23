local opts = require 'opts'
local opt = opts.parse(arg)

local function get_number_crossval(opt)
	require 'paths'
	local cachePath = opt.cache or paths.concat(opt.gen, opt.dataset .. '.t7')
	if not paths.filep(cachePath) then
		error('Cache ' .. cachePath .. ' not found.. please build it!')
	end
	local cache = torch.load(cachePath)
	local N = #cache.crossval_splits
	return N
end

-- Init
local N, means = get_number_crossval(opt)
table.insert(arg, '-crossval_fold')
table.insert(arg, '1')
local tmp_best_model = {}
local tmp_best_epoch = {}
local tmp_train_losses = {}
local tmp_valid_losses = {}
local tmp_metric_str = ''

-- Cross-validation
for k=1,N do
	print(sys.COLORS.red .. '--> CROSS-VALIDATION [' .. k .. '/' .. N .. '] <--' .. sys.COLORS.white)

	-- execute k-th fold
	arg[#arg] = k
	dofile('main.lua') -- train_losses, valid_losses, best_epoch, best_model, best_loss, metric_str computed here

	-- construct best_model, best_epoch, train_losses, valid_losses,  metric_str
	table.insert(tmp_best_model, best_model:clone():clearState())
	table.insert(tmp_best_epoch, best_epoch)
	table.insert(tmp_train_losses, train_losses[best_epoch])
	table.insert(tmp_valid_losses, valid_losses[best_epoch])
	tmp_metric_str = tmp_metric_str .. string.format('[%d]%.4f @%d',k,valid_losses[best_epoch],best_epoch)
end

-- Output global variables
metric_str = tmp_metric_str
best_epoch = 1
train_losses = {torch.Tensor(tmp_train_losses):mean()}
valid_losses = {torch.Tensor(tmp_valid_losses):mean()}
best_loss = valid_losses
best_model = tmp_best_model


