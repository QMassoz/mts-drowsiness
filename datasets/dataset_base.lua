local threads = require 'threads'
threads.serialization('threads.sharedserialize')

local DatasetBase = torch.class('DatasetBase')


---------------------
------- BASIC ------- (can be easily modified)
---------------------
function DatasetBase:__init(data, opt, split)
    self.data = data[split] -- split = 'train' or 'valid' or 'test'
    self.dataset = opt.dataset
    self.split = split
    self.manualSeed = opt.seed
    self.nthreads = 1
end

function DatasetBase:default_init(opt)
    -- data GPU loading mode
    if opt.backend == 'cunn' or opt.backend == 'cudnn' then
        if self.preprocess then
            print(sys.COLORS.yellow .. '<DatasetBase> preprocess() found, backend=' .. opt.backend ..' -> postloading GPU mode' .. sys.COLORS.white)
            self:set_GPU_loading_mode('postloading', false)
        else
            print(sys.COLORS.yellow .. '<DatasetBase> preprocess() not found, backend=' .. opt.backend ..' -> preloading GPU mode' .. sys.COLORS.white)
            self:set_GPU_loading_mode('preloading', false)
        end
    else
        print(sys.COLORS.yellow .. '<DatasetBase> backend=' .. opt.backend ..' -> CPU mode' .. sys.COLORS.white)
        self:set_GPU_loading_mode('noloading', false)
    end

    if self.split == 'valid' or  self.split == 'test' or not self.preprocess then
        self:thread_config(1)
    elseif self.split == 'train' then
        self:thread_config(opt.nthreads or 1, opt.dataset)
    end
end


--** function metric: compute (an) alternative metric(s) (string) that will be displayed&saved (e.g. accuracy of classification)
-- @input opt: table of options
-- @input model: model to compute the metric(s) with
-- @output metric_str: a string containing the alternative metric(s)
-- note: can be overwritten
function DatasetBase:metric(opt,model)
    return '-'
end

--** function get: retrieve a batch of data indexed by the LongTensor i
-- @input i: torch.LongTensor containing the indices of the batch
-- @output inputs, targets
-- note: can be overwritten
function DatasetBase:get(i) -- typical get function
    local inputs = self.data.inputs:index(1,i)
    local targets = self.data.targets:index(1,i)
    return inputs, targets
end

--** function preprocess: preprocess a batch of data
-- @input inputs, targets
-- @output inputs, targets (preprocessed)
-- note: inactive if not defined
-- function DatasetBase:preprocess(inputs, targets)
--    return inputs, targets
--end

--** function size: returns the number of examples
function DatasetBase:size()
    return self.data.inputs:size(1)
end

--** function data2CPU: load all of the data to the CPU into floats
function DatasetBase:data2CPU()
    self.data.inputs = self.data.inputs:float()
    self.data.targets = self.data.targets:float()
end

--** function data2CPU: load all of the data to the GPU
function DatasetBase:data2GPU()
    self.data.inputs = self.data.inputs:cuda()
    self.data.targets = self.data.targets:cuda()
end

--** function batch2GPU: load a (preprocessed CPU) batch into the GPU (used if GPU_loading_mode = postloading)
-- @input inputs, targets: on CPU
-- @output inputs, targets: on GPU
function DatasetBase:batch2GPU(inputs, targets)
    -- inputs
    self.batchGPU_inputs = self.batchGPU_inputs or torch.CudaTensor()
    self.batchGPU_inputs:resize(inputs:size()):copy(inputs)
    -- targets
    self.batchGPU_targets = self.batchGPU_targets or torch.CudaTensor()
    self.batchGPU_targets:resize(targets:size()):copy(targets)    
    return self.batchGPU_inputs, self.batchGPU_targets
end


--** function loader: returns the iterative function to plug into 'for n, inputs, targets in loop do'
-- @input batch_size: the batch size
-- @output shuffles indices, loop
-- usage: shuffles_indices, loop = x:loader(batch_size) [mono-threading by default]
-- modification: 
DatasetBase.loader = DatasetBase.loader_onethread

---------------------
------ ADVANCED ----- (best if not modified)
---------------------
function DatasetBase:set_GPU_loading_mode(mode, verbose)
    assert(mode)    
    local verbose = verbose ~= false
    if mode==0 or mode=='noloading' then
        if verbose then print(sys.COLORS.yellow .. '<DatasetBase:set_GPU_loading_mode> CPU mode' .. sys.COLORS.white) end
        self.GPU_loading_mode = 'noloading'
        self:data2CPU()
    elseif mode==1 or mode=='preloading' then
        if verbose then print(sys.COLORS.yellow .. '<DatasetBase:set_GPU_loading_mode> preloading GPU mode' .. sys.COLORS.white) end
        if self.preprocess then print(sys.COLORS.red .. '<DatasetBase:set_GPU_loading_mode> !!preprocess() found + preloading GPU mode = be careful!!' .. sys.COLORS.white) end
        self.GPU_loading_mode = 'preloading'
        self:data2GPU()
    elseif mode==2 or mode=='postloading' then
        if verbose then print(sys.COLORS.yellow .. '<DatasetBase:set_GPU_loading_mode> postloading GPU mode' .. sys.COLORS.white) end
        self.GPU_loading_mode = 'postloading'
        self:data2CPU()
    else
        error('<DatasetBase:set_GPU_loading_mode> provided mode unknown: ' .. mode)
    end
end

function DatasetBase:thread_config(nthreads, dataset) -- switch between mono-threading (nthreads=1) and multi-threading (nthreads>1)
    -- dataset
    assert(nthreads, nthreads==1 or dataset)
    self.nthreads = nthreads

    -- kill threads if existing
    if self.workers then
            print(sys.COLORS.yellow .. '<DatasetBase:thread_config> terminating existing workers'.. sys.COLORS.white)
            self.workers:terminate()
            self.workers = nil
        end

    -- mono-threading
    if self.nthreads <= 1 then
        print(sys.COLORS.yellow .. '<DatasetBase:thread_config> configuring loader with mono-threading'.. sys.COLORS.white)
        self.loader = DatasetBase.loader_onethread
    -- multi-threading
    else
        print(sys.COLORS.yellow .. '<DatasetBase:thread_config> configuring loader with multi-threading [nthreads=' .. nthreads .. ']'.. sys.COLORS.white)
        local dataset = 'datasets/' .. dataset .. '/dataset-load'
        local require_cutorch = self.GPU_loading_mode == 'preloading' or self.GPU_loading_mode == 'postloading'
        local function init(threadid)
            require(dataset)
            if require_cutorch then
                require 'cutorch'
            end
        end
        local function main(threadid)
            torch.setnumthreads(1)
            _G.dataset = self
        end
        local nworkers = self.nthreads - 1
        self.workers = threads.Threads(nworkers, init, main)
        self.loader = DatasetBase.loader_multithread
    end
end

--- MONO-THREADING --
function DatasetBase:loader_onethread(batch_size) -- loader function for 1 thread
    local shuffled_indices = torch.randperm(self:size()):long():split(batch_size)
    local n=0
    local function loop() -- iterative function to plug into 'for n, inputs, targets in loop do'
        n = n + 1
        if n > #shuffled_indices then
            return nil
        else
            local batch_inputs, batch_targets = self:get(shuffled_indices[n])
            if self.preprocess then
                batch_inputs, batch_targets = self:preprocess(batch_inputs, batch_targets)
            end
            if self.GPU_loading_mode == 'postloading' then
                batch_inputs, batch_targets = self:batch2GPU(batch_inputs, batch_targets)
            end
            return n, batch_inputs, batch_targets
        end
    end
    return shuffled_indices, loop   
end

-- MULTI-THREADING --
function DatasetBase:loader_multithread(batch_size)
    self.workers:synchronize()
    local shuffled_indices = torch.randperm(self:size()):long():split(batch_size)
    local idx, worker_output = 1, nil
    local function callback(worker_indices, rng_seed)
        torch.manualSeed(rng_seed)
        local inputs, targets = _G.dataset:get(worker_indices)
        if _G.dataset.preprocess then
            inputs, targets = _G.dataset:preprocess(inputs, targets)
        end
        return {inputs=inputs, targets=targets}
    end
    local function endcallback(_output_)
        worker_output = _output_
        if self.GPU_loading_mode == 'postloading' then
            worker_output.inputs, worker_output.targets = self:batch2GPU(worker_output.inputs, worker_output.targets)
        end
    end
    local function fillqueue() -- fill the queue of the workers
        while idx <= #shuffled_indices and self.workers:acceptsjob() do
            self.workers:addjob(callback, endcallback, shuffled_indices[idx], torch.random())
            idx = idx + 1
        end
    end

    local n = 0
    function loop() -- iterative function to plug into 'for n, inputs, targets in loop do'
        fillqueue()
        if not self.workers:hasjob() then
            return nil
        end
        self.workers:dojob() -- execute the next endcallback in the queue
        if self.workers:haserror() then
            print('<DatasetBase:loader_threaded()> workers:haserror() has returned a true value, synchronizing all the workers')
            self.workers:synchronize()
        end
        fillqueue()
        n = n+1
        return n, worker_output.inputs, worker_output.targets
    end

    return shuffled_indices, loop
end
