require 'datasets/dataset_base'
local M = {}
local Dataset, parent = torch.class('Dataset','DatasetBase',M)

function Dataset:__init(data, opt, split)
    print("<Dataset> creating " .. split .. " split")
    parent.__init(self,data,opt,split,preprocess_data)

    self.inputSize = data.inputSize -- {#channels, width, height}
    self.outputSize = self.data.targets:size(2)

    -- Flip data
    if opt.flip then
        print("<Dataset> augmenting data with flipping")
        require 'image'
        local concat_flip = function(x, dim) 
            return torch.cat(x ,image.flip(x:contiguous(),dim), 1) 
        end
        self.data.inputs = concat_flip(self.data.inputs,4)
        self.data.targets = concat_flip(self.data.targets,2)
    end

    -- preprocessing config
    self.rotation = opt.rotation
    
    -- GPU loading mode
    self:default_init(opt)
end

function Dataset:preprocess(inputs,targets)
    return inputs,targets
end

-- Computed from entire face2eye128 training set
local meanstd = {
    mean = {0.45895628600413},
    std  = {1}
}

return M.Dataset