local M = {}
require 'nn'
require 'cudnn'
require 'cunn'
nninit = require 'nninit'

local function convnet(inputSize, netLayers, opt)
    print('<model> creating convnet')
    -- init
    local n_conv_layers = #netLayers - 1
    local n_output = netLayers[#netLayers]
    local conv_dims = torch.cat(torch.Tensor({inputSize[1]}), torch.Tensor(netLayers):narrow(1,1,n_conv_layers))

    local opt = opt or {}
    local conv_size = opt.conv_size or 3
    local p_dropout = opt.dropout or 0
    local last_pooling = opt.last_pooling or 'max'
    
    -- Model composition
    local model = nn.Sequential()
    local pad = (conv_size-1)/2

    for i=2,n_conv_layers+1 do
        model:add(nn.SpatialConvolution(conv_dims[i-1], conv_dims[i], conv_size,conv_size, 1,1, pad,pad)
            :init('weight',nninit.kaiming,{gain='relu'})
            :init('bias',nninit.constant,0.01))
        model:add(nn.ReLU(true))
        model:add(nn.SpatialBatchNormalization(conv_dims[i])) 

        model:add(nn.SpatialConvolution(conv_dims[i], conv_dims[i], conv_size,conv_size, 1,1, pad,pad)
            :init('weight',nninit.kaiming,{gain='relu'})
            :init('bias',nninit.constant,0.01))
        model:add(nn.ReLU(true))

        if i~=n_conv_layers+1 then
            model:add(nn.SpatialBatchNormalization(conv_dims[i])) 
            model:add(nn.SpatialMaxPooling(2,2, 2,2))
        end
    end

    if last_pooling == 'max' then
        model:add(nn.Max(3,3)) --efficient global max
        model:add(nn.Max(2,2))
    elseif last_pooling == 'average' then
        model:add(nn.Mean(3,3)) --efficient global mean
        model:add(nn.Mean(2,2))
    else
        error("unknown option last_pooling=" .. last_pooling)
    end

    if p_dropout > 0 then 
        model:add(nn.Dropout(p_dropout))
    end
    model:add(nn.Linear(conv_dims[#conv_dims],1))
 
    return model
end

function M.model(inputSize, outputSize, opt)
    local layers = {}
    if opt.layers  then
        -- Convert opt.layers (string) to layers (table)
        opt.layers:gsub("%d+", function(c) table.insert(layers,tonumber(c)) end)
    else
        layers = {32, 64, 128, 256}
    end

    table.insert(layers, outputSize)
    

    -- Build the model
    local model = convnet(inputSize, layers, opt)

    return model
end

function M.criterion(opt)
    return nn.MSECriterion(true)
end

return M
