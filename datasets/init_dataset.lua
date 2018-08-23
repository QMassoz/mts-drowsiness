local M = {}
require 'paths'

function M.create(opt)
   -- Load cache
   local cachePath = opt.cache or paths.concat(opt.gen, opt.dataset .. '.t7')
   if not paths.filep(cachePath) then
      paths.mkdir(opt.gen)
      local script = paths.dofile(opt.dataset .. '/dataset-build.lua')
      script.build(opt, cachePath)
   end
   local cache = torch.load(cachePath)
   -- Create train, val, and test splits
   local loaders = {}
   for i, split in ipairs{'train', 'valid','test'} do
      local dataset = require('datasets/' .. opt.dataset .. '/dataset-load')
      if cache[split] then
         loaders[i] =  dataset(cache, opt, split)
      else
         loaders[i] = false
         print(sys.COLORS.red .. '<init_dataset> ' .. split .. ' split not found' .. sys.COLORS.white)
      end
   end
   return table.unpack(loaders)
end

return M
