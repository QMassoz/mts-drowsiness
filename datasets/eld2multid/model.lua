local M = {}
require 'nn'
require 'cudnn'
require 'cunn'
nninit = require 'nninit'

local function Sequential(tbl,model)
    local module = model or nn.Sequential()
    for i=1, #tbl do
        if tbl[i] then
            if torch.type(tbl[i]) == 'table' then
                module = Sequential(tbl[i],module)
            else
                module:add(tbl[i])
            end
        end
    end
    return module
end
local function Concat(tbl)
    local module = nn.Concat(2)
    for i=1, #tbl do
        if tbl[i] then
            module:add(tbl[i])
        end
    end
    return module
end


local function temporalpooling(n_input,n_output,t0,d)
    local model = Sequential{
        nn.SpatialConvolution(n_input,n_output, 3,3, 1,1, 1,1)
            :init('weight',nninit.kaiming,{gain='relu'})
            :init('bias',nninit.constant,0.01),
        nn.ReLU(true)
        }
    if t0 == -1 then
        model:add(nn.Mean(3,3)):add(nn.Mean(2,2))
    else
        model:add(nn.SigmoidPooling(t0/d,d/2))
    end
    return model
end
function temporal_convnet(inputSize, outputSize, layers, opt)
    local n1,t1, n2,t2, n3 = table.unpack(layers)

    require 'modules/SigmoidPooling'
    require 'modules/MixConcatTable'

    -- PART I) FEATURES EXTRACTION
    local feat_extractor, n2i, d
    feat_extractor = Sequential{
            Concat{
                nn.Identity(),
                Sequential{
                    nn.SpatialConvolution(inputSize,n1, t1,1, 1,1, (t1-1)/2,0)
                        :init('weight',nninit.kaiming,{gain='relu'})
                        :init('bias',nninit.constant,0.01),
                    nn.ReLU(true),
                    nn.SpatialBatchNormalization(n1),
                    nn.Contiguous()
                }
            },
            nn.SpatialMaxPooling(3,1,3,1),
            Concat{
                nn.Identity(),
                Sequential{
                    nn.SpatialConvolution(2+n1,n1, t1,1, 1,1, (t1-1)/2,0)
                        :init('weight',nninit.kaiming,{gain='relu'})
                        :init('bias',nninit.constant,0.01),
                    nn.ReLU(true),
                    nn.SpatialBatchNormalization(n1),
                    nn.Contiguous()
                }
            } 
    }  
    n2i = 2+2*n1
    d=3

    local model = Sequential{
        feat_extractor,
        opt.dropout>0 and nn.SpatialDropout(opt.dropout/2) or false,
        nn.ConcatTable()
            :add(Sequential{
                nn.SpatialConvolution(n2i,n2, t2,1, 1,1, (t2-1)/2,0)
                    :init('weight',nninit.kaiming,{gain='relu'})
                    :init('bias',nninit.constant,0.01),
                nn.ReLU(true),
                nn.SpatialBatchNormalization(n2),
                nn.SigmoidPooling(150/d,d/2),
                opt.dropout>0 and nn.Dropout(opt.dropout) or false
            })
            :add(Sequential{
                nn.SpatialConvolution(n2i,n2, t2,1, 1,1, (t2-1)/2,0)
                    :init('weight',nninit.kaiming,{gain='relu'})
                    :init('bias',nninit.constant,0.01),
                nn.ReLU(true),
                nn.SpatialBatchNormalization(n2),
                nn.SigmoidPooling(450/d,d/2),
                opt.dropout>0 and nn.Dropout(opt.dropout) or false
            })
            :add(Sequential{
                nn.SpatialConvolution(n2i,n2, t2,1, 1,1, (t2-1)/2,0)
                    :init('weight',nninit.kaiming,{gain='relu'})
                    :init('bias',nninit.constant,0.01),
                nn.ReLU(true),
                nn.SpatialBatchNormalization(n2),
                nn.SigmoidPooling(900/d,d/2),
                opt.dropout>0 and nn.Dropout(opt.dropout) or false
            })
            :add(Sequential{
                nn.SpatialConvolution(n2i,n2, t2,1, 1,1, (t2-1)/2,0)
                    :init('weight',nninit.kaiming,{gain='relu'})
                    :init('bias',nninit.constant,0.01),
                nn.ReLU(true),
                nn.SpatialBatchNormalization(n2),
                nn.Mean(3,3),
                nn.Mean(2,2),
                opt.dropout>0 and nn.Dropout(opt.dropout) or false
            }),
        
        nn.MixConcatTable(4, n2, n3),
        nn.ParallelTable()
            :add(Sequential{
                opt.dropout>0 and nn.Dropout(opt.dropout/2) or false,
                nn.Linear(n3*2,2),
                nn.SoftMax()
            })
            :add(Sequential{
                opt.dropout>0 and nn.Dropout(opt.dropout/2) or false,
                nn.Linear(n3*2,2),
                nn.SoftMax()
            })
            :add(Sequential{
                opt.dropout>0 and nn.Dropout(opt.dropout/2) or false,
                nn.Linear(n3*2,2),
                nn.SoftMax()
            })
            :add(Sequential{
                opt.dropout>0 and nn.Dropout(opt.dropout/2) or false,
                nn.Linear(n3*2,2),
                nn.SoftMax()
            })
    }

    return model
end

function M.model(inputSize, outputSize, opt)
    local layers = {}
    if opt.layers then
        opt.layers:gsub("%d+", function(c) table.insert(layers,tonumber(c)) end)
    else
        layers = {32, 15, 32, 31, 16}
    end
    -- Build the model
    local model = temporal_convnet(inputSize[1], outputSize, layers, opt)
    return model
end

function M.criterion(opt)
    local criterion = nn.ParallelCriterion()
        :add(nn.BCECriterion(),.25)
        :add(nn.BCECriterion(),.25)
        :add(nn.BCECriterion(),.25)
        :add(nn.BCECriterion(),.25)

    require 'modules/WeightedKLDCriterion'
    local w = function(i)
        if opt.weight_eval == 'auto' then
            local weights = torch.Tensor{1,10}
            if opt.histClasses and not opt.testModel then
                local hist = opt.histClasses[{{2*i-1,2*i}}]
                if torch.abs(hist):sum() == 0 then
                    error('Weird weights...')
                end
                weights = torch.cinv(hist+1e-12):cmul(hist:gt(0):double()) 
                weights = weights* hist[hist:gt(0)]:mean() -- (mean=total/nClasses)/occurence  
                --weights = torch.cinv(torch.log(1.05+hist/hist:sum()))
            end
            return weights
        elseif opt.weight_eval == 'fixed' then
            if     i==1 then return torch.Tensor{0.5728,3.9318} 
            elseif i==2 then return torch.Tensor{0.5378,7.1173} 
            elseif i==3 then return torch.Tensor{0.5275,9.5810} 
            elseif i==4 then return torch.Tensor{0.5222,11.7821} end
        else
            return torch.Tensor{1,4}
        end
    end

    print('<model> evaluation criterion with weights', 
        table.unpack(torch.cat({w(1),w(2),w(3),w(4)},1):totable()))
    local criterion_eval = nn.ParallelCriterion()
        :add(nn.WeightedKLDCriterion(w(1)),.25)
        :add(nn.WeightedKLDCriterion(w(2)),.25)
        :add(nn.WeightedKLDCriterion(w(3)),.25)
        :add(nn.WeightedKLDCriterion(w(4)),.25)

    return criterion, criterion_eval
end

return M
