require 'nn'
require 'torch'
require 'cutorch'
local opts = require 'opts'
local Trainer = require 'trainer'
local datasets = require 'datasets/init_dataset'
local models = require 'datasets/init_model'
local analysers = require 'datasets/init_analyser'

torch.setnumthreads(1)

-- Parse options
local opt = opts.parse(arg)
torch.manualSeed(opt.seed)
cutorch.manualSeedAll(opt.seed)

-- Load data
print("<main> data: " .. opt.dataset)-- data file created from build_features.lua
LS, VS, TS = datasets.create(opt) -- LearningSet, ValidationSet, TestingSet
local inputSize = LS.inputSize -- {#channels, width, height}
local outputSize = LS.outputSize

-- Model
model, criterion, criterion_eval = models.create(inputSize, outputSize, opt)

-- Test only routine
if opt.testModel then
    model = torch.load(opt.testModel)
    if torch.type(model) == 'table' then -- feed forward neural networks with cross validation
        local N, train_loss, val_loss, test_loss, time = #model,0,0,0,0
        LS,VS,TS = {},{},{}
        train_losses,valid_losses,test_losses = {},{},{}
        
        for k=1,N do
            opt.crossval_fold = k -- kth fold
            LS_k, VS_k, TS_k = datasets.create(opt) -- kth fold LearningSet, ValidationSet, TestingSet
            local model_k = model[k]:cuda() -- kth fold model

            local trainer = Trainer(model_k, criterion, criterion_eval, opt)
            if not opt.testOnly then
                train_loss, time = trainer:eval(LS_k, opt.batch_size)
                val_loss, _ = trainer:eval(VS_k, opt.batch_size)
                LS[#LS+1] = LS_k 
                VS[#VS+1] = VS_k 
            end
            if TS_k then
                test_loss, time = trainer:eval(TS_k, opt.batch_size)
                TS[#TS+1] = TS_k
            end
            
            train_losses[#train_losses+1] = train_loss
            valid_losses[#valid_losses+1] = val_loss
            test_losses[#test_losses+1] = test_loss

            print(string.format('[%d/%d] train= %6.8f, valid= %6.8f, test= %6.8f (%5.3f ms/sample)', 
                    k, N, train_loss, val_loss, test_loss, time))
            --local deacheck_str = models.deadcheck(model_k,'nn.ReLU', VS_k, opt.batch_size)
            --print(sys.COLORS.green .. deacheck_str .. sys.COLORS.white)

            model_k:clearState():float()
        end
        print('\nSummary of cross validation:\nLearning Set\tValidation Set\tTesting Set')
        for k=1,N do
            print(string.format('%.8f\t%.8f\t%.8f', train_losses[k], valid_losses[k], test_losses[k]))
        end
        print(string.format('---------\n%.8f\t%.8f\t%.8f', torch.Tensor(train_losses):mean(), torch.Tensor(valid_losses):mean(), torch.Tensor(test_losses):mean()))

    else -- feedforward neural networks
        model = torch.load(opt.testModel):cuda()
        print(model)
        local trainer = Trainer(model, criterion, criterion_eval, opt)
        local train_loss, val_loss, test_loss, time = 0,0,0,0
        if not opt.testOnly then
            train_loss, time = trainer:eval(LS, opt.batch_size)
            val_loss, _ = trainer:eval(VS, opt.batch_size)
        end
        if TS then
            test_loss, time = trainer:eval(TS, opt.batch_size)
        end
        print(string.format('train= %6.8f, valid= %6.8f, test= %6.8f (%5.3f ms/sample)', 
                train_loss, val_loss, test_loss, time))
        local deacheck_str = models.deadcheck(model,'nn.ReLU', VS, opt.batch_size)
        print(sys.COLORS.green .. deacheck_str .. sys.COLORS.white)
    end
    M = analysers.init(opt,model,LS,VS,TS)
    return
end


-- Optimization loop
local trainer = Trainer(model, criterion, criterion_eval, opt)
print(model)

train_losses, valid_losses = {}, {} -- losses for each iteration/minibatch
best_model = {}
best_loss = math.huge
best_epoch = 0
metric_str = '-'
local time_training, time_inference, deadcheck_str -- local variables
local epoch_counter = 0
for e = 1, opt.max_epochs do
    epoch_counter = epoch_counter + 1 

    -- train epoch & validate epoch
    train_losses[#train_losses + 1], time_training  = trainer:train_epoch(LS, opt.batch_size)
    valid_losses[#valid_losses + 1], time_inference = trainer:eval(VS, opt.batch_size)
    --deadcheck_str = models.deadcheck(model,'cudnn.ReLU', VS, opt.batch_size) -- deadcheck for feedforward networks only

    -- keep best model
    local epoch_msg = string.format('epoch %s: train= %6.8f, valid= %6.8f (%5.3f & %5.3f ms/sample)', 
                        e, train_losses[#train_losses], valid_losses[#valid_losses],time_training,time_inference)
    if best_loss > valid_losses[#valid_losses] then  
        -- fill best_model
        best_model = model:clearState():clone():float()
        best_model:evaluate()
        if opt.backend == 'cudnn' then
            cudnn.convert(best_model, nn)
        end
        -- fill other variables
        epoch_counter = 0
        best_epoch = e
        best_loss = valid_losses[#valid_losses]
        metric_str = VS:metric(opt,best_model)


        print(sys.COLORS.red .. epoch_msg .. sys.COLORS.white)
        if metric_str ~= '-' then
            print(sys.COLORS.red .. 'metric(s): ' .. metric_str .. sys.COLORS.white)
        end
        if deadcheck_str then print(sys.COLORS.green .. deadcheck_str .. sys.COLORS.white) end
        if opt.dataset == 'eyedetect' or opt.dataset == 'face2eyedetect' then print(criterion.loss) end
    else
        print(epoch_msg) 
    end

    -- anneal learning rate
    if opt.lr_epoch > 0 and epoch_counter >= opt.lr_epoch then
        epoch_counter = 0
        local new_lr = trainer:lr(0.5)
        print('NEW LEARNING RATE: ' .. new_lr)
    end
    
    -- stop criterions
    if opt.min_train_error and train_losses[#train_losses] < tonumber(opt.min_train_error) then
        break
    elseif valid_losses[#valid_losses] ~= valid_losses[#valid_losses] then -- valid_losses = NaN
        break
    elseif train_losses[#train_losses] ~= train_losses[#train_losses] then -- train_losses = NaN
        break
    elseif opt.earlystop and epoch_counter >= tonumber(opt.earlystop) then
        break
    end
end
collectgarbage()



-- Save (best) model
if best_epoch > 0 then
    require 'paths'
    print("---------- SAVING " .. opt.save_file .. " ----------")
    print(string.format('-> BEST EPOCH %s: training loss= %6.6f, validation loss= %6.6f ----',
                        best_epoch, train_losses[best_epoch], valid_losses[best_epoch]))
    print('-> metric(s): ' .. metric_str)
    torch.save(opt.save_file, best_model)
    function save(suffix)
    -- save the best_model in a formated format 
        local timestamp = os.date('%d-%m-%Y_%Hh%M') --os.date('%d.%m.%Y_%H.%M.%S')
        --local model_name = paths.basename(opt.dataset, paths.extname(opt.dataset))
        local model_name = opt.dataset
        local suffix = suffix or ''
        if suffix ~= '' then
            suffix = '_' .. suffix
        end
        local filename = 'models/' .. opt.dataset .. '/' .. model_name .. string.format('_%6.4f_',best_loss) 
                    .. timestamp .. suffix .. '.net'
        print("---------- SAVING " .. filename .. " ----------")
        torch.save(filename, best_model:clearState())
    end
end


-- Plot 
function lossplot() -- plot the training and validation loss curves
    require 'gnuplot'
    gnuplot.plot(
      {'training loss',   torch.FloatTensor(train_losses), '-'},
      {'validation loss', torch.FloatTensor(valid_losses), '-'}
    )
end
M = analysers.init(opt,best_model,LS,VS,TS)
