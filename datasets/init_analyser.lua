local M = {}
require 'paths'

function M.init(opt, model, LS, VS, TS)
   -- Load cache
   local analyserPath = paths.concat('datasets', opt.dataset, 'analyse.lua')
   if not paths.filep(analyserPath) then
      return {}
   end
   local analyser = require('datasets/' .. opt.dataset .. '/analyse')
   return analyser.init(opt, model, LS, VS, TS)
end

return M
