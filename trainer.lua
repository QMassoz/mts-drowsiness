require 'optim'
require 'xlua'
local M = {}
local Trainer = torch.class('Trainer', M)

function Trainer:__init(model, criterion, criterion_eval, opt)
    self.optim_state  = {} -- stores optimization algorithm's settings, and state during iterations
    self.optimizer = {} -- stores a function corresponding to the optimization routine
    if opt.optimizer == 'sgd' then
        self.optim_state = {
            learningRate = opt.lr,
            weightDecay = opt.wd,
            momentum = 0.9,
            dampening = 0,
            nesterov = true,
            learningRateDecay = opt.lrd
        }
        self.optimizer = optim.sgd
    elseif opt.optimizer == 'rmsprop' then
        self.optim_state = {
            learningRate = opt.lr,
            alpha = opt.alpha,
            weightDecay = opt.wd
        } 
        self.optimizer = optim.rmsprop
    elseif opt.optimizer == 'adam' then
        self.optim_state = {
            learningRate = opt.lr,
            beta1 = 0.9,    -- often 0.9
            beta2 = 0.999,  -- often 0.999
            epsilon = 1e-8
        }
        self.optimizer = optim.adam
    end

    self.model = model
    self.criterion = criterion
    self.criterion.sizeAverage = true
    self.criterion_eval = criterion_eval
    self.criterion_eval.sizeAverage = true
    self.params, self.gradParameters = self.model:getParameters()
    self.epoch_counter = 0
end

function Trainer:lr(scale)
    print('<Trainer> adapting learning rate')
    self.optim_state.learningRate = self.optim_state.learningRate * scale
    return self.optim_state.learningRate
end

function Trainer:train(inputs, targets)    
    local function feval(p)
        if p ~= self.params then
            self.params:copy(p)
        end

        self.model:zeroGradParameters()
        -- forward
        local batch_outputs = self.model:forward(inputs)
        local batch_loss = self.criterion:forward(batch_outputs, targets)
        -- backward
        local dloss_doutput = self.criterion:backward(batch_outputs, targets) 
        self.model:backward(inputs, dloss_doutput)

        return batch_loss, self.gradParameters
    end

    local _, batch_loss = self.optimizer(feval, self.params, self.optim_state)
    return batch_loss[1]
end

function Trainer:train_epoch(dataLoader, batch_size)
    self.epoch_counter = self.epoch_counter + 1
    self.model:training()
    self.model:clearState()
    
    local tic = torch.tic()
    local loss = 0
    local shuffled_indices, loop = dataLoader:loader(batch_size)
    for k, batch_inputs, batch_targets in loop do -- retrieve batch data
        if opt.progress then
            xlua.progress(k,#shuffled_indices)
        end
        local batch_loss = self:train(batch_inputs, batch_targets)
        loss = loss + batch_loss * shuffled_indices[k]:size(1)
        collectgarbage()
    end
    local time = torch.toc(tic) * 1000 / dataLoader:size() -- exec time [ms/sample]
    return loss / dataLoader:size(), time
end

function Trainer:eval(dataLoader, batch_size)
    self.model:evaluate()
    self.model:clearState()

    local tic = torch.tic()
    local loss = 0
    local batch_size = batch_size or 250
 
    local shuffled_indices, loop = dataLoader:loader(batch_size)
    for k, batch_inputs, batch_targets in loop do -- retrieve batch data
        if opt.progress then
            xlua.progress(k,#shuffled_indices)
        end
        local batch_loss = self.criterion_eval:forward(self.model:forward(batch_inputs),batch_targets)
        loss = loss + batch_loss * shuffled_indices[k]:size(1)
        collectgarbage()
    end

    local time = torch.toc(tic) * 1000 / dataLoader:size() -- exec time [ms/sample]
    return loss / dataLoader:size(), time
end

return M.Trainer