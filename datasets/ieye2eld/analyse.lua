local M = {}

require 'torch'
require 'cudnn'
require 'cunn'

function M.init(opt,model,LS,VS,TS)


	print(" ")
	print(sys.COLORS.cyan .. "<ieye2eld-analyse> call list:" .. sys.COLORS.white)
	print("  - pred,gt = M.errScatterSet(set): scatter plot of the specified set errors")

	local function errScatterSet(set)
		local set = set or TS
		-- plot the training and validation regression errors
	    model:evaluate()
	    local batch_size = 100
	    local split = set.split
	    local set = set.data
	    
	    local set_yt = torch.Tensor(set.targets:size()):cuda()
	    local idx = torch.range(1,set.targets:size(1)):long():split(batch_size)
		for _, batch_idx in ipairs(idx) do
	        local batch_inputs = set.inputs:index(1,batch_idx)
	        set_yt[{{batch_idx[1],batch_idx[-1]},{}}]:copy(model:cuda():forward(batch_inputs:cuda()))
		end
	    local set_y = set.targets:cuda()
	    local set_err = set_yt - set_y

	    require 'gnuplot'
	    for i=1,set.targets:size(2) do
	        if i>1 then
	        	gnuplot.figure()
	        end
	        gnuplot.plot({
	            split..' data',
	            set_y[{{},i}],
	            set_yt[{{},i}],
	            '+'})
	        gnuplot.xlabel('y')
	        gnuplot.ylabel('y_model')
	        gnuplot.title('scatter error - output ' .. i)
	    end

	    print(split.. ' MSE:', torch.pow(set_err,2):mean())
	    return set_yt, set_y
	end



	-- return all functions
	return {
		errScatterSet=errScatterSet 
	}
end

return M