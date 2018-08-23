require 'datasets/dataset_base' 
local M = {}
local Dataset, parent = torch.class('Dataset','DatasetBase',M)

function Dataset:__init(data, opt, split)
    -- data
    print("<Dataset> creating " .. split .. " split")
    parent.__init(self,data,opt,split)

    -- selecting the crossvalidation-fold to use
    self.data = self.data[opt.crossval_fold]

    -- augmentation
    self.augment = opt.augment
    self.n_augment = opt.n_augment or 1056
    if split=='train' and self.augment then
        self.augment_groups = self:make_augment_groups(self.data.medianRT_values)
    end

    -- targets to probability of drowsiness
    self.data.RT = self.data.targets:clone()
    self.data.targets = self:targets2pdrowsy(self.data.targets)
    -- medianRT_values to probability of drowsiness
    if split=='train' and self.augment then
        self.data.pdrowsy_values = self:medianRT_values2pdrowsy(self.data.medianRT_values:clone())
    end

    -- preprocessing
    self.flip = opt.flip and 0.5 or 0

    -- parameters
    self.inputSize = data.inputSize -- {#channels, width, height}
    self.outputSize = 8

    -- GPU loading mode
    self:default_init(opt)
end

function Dataset:metric(opt,model)
    return '-'
end

function Dataset:make_augment_groups(targets)
    local N,T = targets:size(1),targets:size(2)
    local augment_groups = {}
    augment_groups[1] = torch.nonzero((targets:gt(0) + targets:le(470)):eq(2):double():sum(3):view(N,T):eq(4))
    augment_groups[2] = torch.nonzero(targets:ge(470):double():sum(3):view(N,T):eq(4))
    augment_groups[3] = torch.nonzero(targets:ge(470):double():sum(3):view(N,T):eq(3))
    augment_groups[4] = torch.nonzero(targets:ge(470):double():sum(3):view(N,T):eq(2))
    augment_groups[5] = torch.nonzero(targets:ge(470):double():sum(3):view(N,T):eq(1))
    return augment_groups
end
function Dataset:targets2pdrowsy(targets)
    local idx_400 = targets:le(400)
    local idx_500 = targets:ge(500)
    local idx_mid = 1-idx_400-idx_500

    targets[idx_400] = 0
    targets[idx_500] = 1
    targets[idx_mid] = (targets[idx_mid]-470)/10000000+0.5
    targets = torch.cat(1-targets,targets,3):view(targets:size(1),8)
    return targets
end
function Dataset:medianRT_values2pdrowsy(medianRT_values)
    local idx_400 = medianRT_values:le(400)
    local idx_500 = medianRT_values:ge(500)
    local idx_mid = 1-idx_400-idx_500

    medianRT_values[idx_400] = 0
    medianRT_values[idx_500] = 1
    medianRT_values[idx_mid] = 0.5
    medianRT_values = torch.cat(1-medianRT_values,medianRT_values,4):view(medianRT_values:size(1),medianRT_values:size(2),8)
    return medianRT_values
end
function Dataset:batch2GPU(inputs,targets)
    -- inputs
    self.batchGPU_inputs = self.batchGPU_inputs or torch.CudaTensor()
    self.batchGPU_inputs:resize(inputs:size()):copy(inputs)
    -- targets
    self.batchGPU_targets = self.batchGPU_targets or {torch.CudaTensor(),torch.CudaTensor(),torch.CudaTensor(),torch.CudaTensor()}
    for k=1,4 do
        self.batchGPU_targets[k]:resize(targets[k]:size()):copy(targets[k]) 
    end   
    return self.batchGPU_inputs, self.batchGPU_targets
end
function Dataset:extract_sample(test, s)
    --[[
        test = ID of the test
        s    = anchor 'timestamp' in minutes (from 1 to 10)
    --]]
    -- inputs
    local i_end   = torch.round(s*1800)
    local i_start = i_end - 1800+1
    local inputs = self.data.eye_values[{test,{i_start,i_end},{} }]:clone()
    inputs:csub(self.data.mean)
    -- targets
    local targets = self.data.pdrowsy_values[{test,i_end,{}}]
    -- reaction speeds
    local RTs = self.data.medianRT_values[{test,i_end,{}}]
    -- output
    return inputs:t(),targets,RTs
end
function Dataset:get(i)
    if self.split == 'valid' or self.split == 'test' or not self.augment then
        local inputs = self.data.inputs:index(1,i)
        local targets = self.data.targets:index(1,i)
        local RTs = self.data.RT:index(1,i)
        return inputs, targets:split(2,2), RTs
    elseif self.augment == 'balance' or self.augment == 'random' then
        local groups = torch.floor((i:float()-1)/self.n_augment)+1
        local N = i:size(1)
        local ntests = self.data.eye_values:size(1)

        local inputs = torch.Tensor(N,2,1,1800)
        local targets = torch.Tensor(N,8)
        local RTs = torch.Tensor(N,4)

        for k=1,N do
            local i,t,rt
            if self.augment == 'balance' then
                local g = self.augment_groups[groups[k]]
                local rline = torch.random(1,g:size(1))
                i,t,rt = self:extract_sample(g[{rline,1}], g[{rline,2}]/1800)
            elseif self.augment == 'random' then
                i,t,rt = self:extract_sample(torch.random(1,ntests), torch.uniform(1,10))
            end
            inputs[{k,{},1,{}}] = i
            targets[{k,{}}] = t
            RTs[{k,{}}] = rt
        end

        return inputs, targets:split(2,2), RTs
    end
end

function Dataset:size()
    if self.split == 'valid' or self.split == 'test' or not self.augment then
        return self.data.inputs:size(1)
    else
        return #self.augment_groups*self.n_augment
    end
end

require 'image'
function Dataset:preprocess(inputs,targets)
    if self.split == 'train' then
        local proc_inputs = torch.Tensor(inputs:size())
        for i=1,inputs:size(1) do
            if self.flip>0 and torch.uniform(0,1)<=self.flip then
                proc_inputs[{i,{},{},{}}]:copy(image.flip(inputs[{i,{},{},{}}],1))
            end
        end
        return inputs, targets
    else
        return inputs,targets
    end
end

return M.Dataset