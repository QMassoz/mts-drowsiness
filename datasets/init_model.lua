local M = {}
require 'paths'

function M.create(inputSize, outputSize, opt)
    -- Load cache
    local dirPath = paths.concat('models', opt.dataset)
    if not paths.dirp(dirPath) then
        paths.mkdir(dirPath)
    end

    -- Get criterion and model
    local dataset = require('datasets/' .. opt.dataset .. '/model')

    -- model
    local model = dataset.model(inputSize, outputSize, opt)

    -- Set the CUDNN flags
    if opt.backend == 'cudnn' then
        cudnn.convert(model, cudnn)

        if opt.cudnn == 'fastest' then
            cudnn.fastest = true
            cudnn.benchmark = true
        elseif opt.cudnn == 'deterministic' then
            model:apply(function(m) -- deterministic mode
                if m.setMode then m:setMode(
                    'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM',
                    'CUDNN_CONVOLUTION_BWD_DATA_ALGO_1',
                    'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1') end
            end)
        end
    end

    -- criterion
    local criterion, criterion_eval = dataset.criterion(opt)
    if not criterion_eval then
        criterion_eval = criterion:clone()
    end

    -- convert to cuda
    if opt.backend == 'cudnn' or opt.backend == 'cunn' then
        model = model:cuda()
        criterion = criterion:cuda()
        criterion_eval = criterion_eval:cuda()
    else
        model = model:float()
        criterion = criterion:float()
        criterion_eval = criterion_eval:float()
    end


    return model, criterion, criterion_eval
end

function M.deadcheck(model, module_name, set, N)
    -- compute responses
    local N = N or 250
    --[[local idx
    if set:size() >= N then
        idx = torch.randperm(set:size())[{{1,N}}]:long()
    else
        idx = torch.range(1,set:size()):long()
    end
    model:evaluate()
    model:forward(set:get(idx))]]
    local shuffled_indices, loop = set:loader(N)
    local _, batch_inputs, _ = loop()
    model:evaluate()
    model:forward(batch_inputs)



    -- check responses of specified modules
    local module_name = module_name or 'nn.ReLU'
    local str = '<model> ' .. module_name .. ' deadcheck: '

    local module_nodes = model:findModules(module_name)
    for i = 1, #module_nodes do
        local output = module_nodes[i].output
        if output:dim() == 4 then
            output = output:abs():sum(3):sum(4):sum(1):view(-1)
        elseif output:dim() == 2 then
            output = output:abs():sum(1):view(-1)
        end
        
        -- print ndead / ntotal
        str = str .. output:eq(0):sum() .. '/' .. output:size(1) 
        if i ~= #module_nodes then
            str = str .. ', '
        end
    end
    return str
end

return M
