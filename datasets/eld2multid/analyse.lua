local M = {}

require 'torch'
require 'cudnn'
require 'cunn'

local function array_concat(v1, v2)
-- concatenate 2 tables into 1
	local l = #v1
	for i,v in ipairs(v2) do
		v1[l+i] = v
	end
	return v1
end
local function pearson(a)
	-- compute the mean
	local x1, y1 = 0, 0
	for _, v in pairs(a) do
		x1, y1 = x1 + v[1], y1 + v[2]
	end
	x1, y1 = x1/#a, y1/#a
	-- compute the coefficient
	local x2, y2, xy = 0, 0, 0
	for _, v in pairs(a) do
		local tx, ty = v[1]-x1, v[2]-y1
		xy, x2, y2 = xy+tx*ty, x2+tx*tx, y2+ty*ty
	end
	return xy / math.sqrt(x2) / math.sqrt(y2)
end

function M.init(opt,model,LS,VS,TS)

	print(" ")
	print(sys.COLORS.cyan .. "<eld2multid> call list:" .. sys.COLORS.white)
	print("  - pred,gt = M.classificationError(set): compute the confusion matrix")
	print("  - M.crossClassificationError(set,idx): compute the confusion matrix with cross classes")
	print("  - M.anyDrowsinessClassificationError(set): compute the confusion matrix with gt=drowsiness of any timescale")


	local function compute(set,model,batch_size)
		local pred = {torch.CudaTensor(set:size(),2), torch.CudaTensor(set:size(),2), torch.CudaTensor(set:size(),2), torch.CudaTensor(set:size(),2)}
		local gt = {torch.CudaTensor(set:size(),2), torch.CudaTensor(set:size(),2), torch.CudaTensor(set:size(),2), torch.CudaTensor(set:size(),2)}
		local RTs = torch.CudaTensor(set:size(),set.outputSize/2)
		local idx = torch.range(1,set:size()):long():split(batch_size)
		local medianRTs = {}
		for k, batch_idx in ipairs(idx) do
			xlua.progress(k,#idx)
		    local batch_inputs, batch_targets, batch_RTs = set:get(batch_idx)
		    local cuda_batch_inputs, cuda_batch_targets = set:batch2GPU(batch_inputs, batch_targets)
		    local o = model:cuda():forward(cuda_batch_inputs)
		    for k=1,4 do
			    pred[k][{ {batch_idx[1],batch_idx[-1]},{} }]:copy(o[k])
			    gt[k][{ {batch_idx[1],batch_idx[-1]},{} }]:copy(cuda_batch_targets[k])
			end
			RTs[{ {batch_idx[1],batch_idx[-1]} }]:copy(batch_RTs)
		end
		return pred,gt,RTs
	end
	local function pearson(a)
		-- compute the mean
		local x1, y1 = 0, 0
		for _, v in pairs(a) do
			x1, y1 = x1 + v[1], y1 + v[2]
		end
		x1, y1 = x1/#a, y1/#a
		-- compute the coefficient
		local x2, y2, xy = 0, 0, 0
		for _, v in pairs(a) do
			local tx, ty = v[1]-x1, v[2]-y1
			xy, x2, y2 = xy+tx*ty, x2+tx*tx, y2+ty*ty
		end
		return xy / math.sqrt(x2) / math.sqrt(y2)
	end
	local function confusion(pred,gt,flag)
		require 'optim'
		local conf = optim.ConfusionMatrix({'alert','drowsy'})
		conf:zero()
		for i=1,pred:size(1) do
			if not flag or gt[i]:eq(1):sum() == 1 then
				conf:add(pred[i], gt[i])
			end
		end
		return conf
	end
	local function reactiontimes_within_window(window, RT_ts, RT)
		--[[
			window = table with 2 numbers {start, end}
			RT_ts  = torch.Tensor with the appearance times (in ascending order)
			RT     = torch.Tensor with the reaction times (synchronized with RT_ts)
		--]]
		assert(#window == 2 and window[1] < window[2])
		assert(RT_ts:nElement() == RT:nElement())

		-- Find the start of the window
		local istart = -1
		for i=1,RT_ts:size(1) do
			if RT_ts[i] >= window[1] or (RT_ts[i]+RT[i]) >= window[1] then
				istart = i
				break
			end
		end
		if istart == -1 then
			error("Invalid window; could not find the start of it")
		end
		-- Find the end of the window
		local iend = -1
		for i=istart,RT_ts:size(1) do
			if RT_ts[i] > window[2] then
				if i==istart then
					error('No reaction times within the given window!') -- istart==iend and out of the window
				end
				iend = i-1
				break
			end
		end
		if iend == -1 then
			iend = RT_ts:size(1)
		end
		-- Return the reaction times within the window
		return RT[{{istart,iend}}]:clone()
	end
	local function produce_gt_medianRT(reaction_times, reaction_times_ts)
		local function extract_medianRT(test, s, w1, w2)
		    --test = ID of the test; s = anchor 'timestamp' in minutes (from 1 to 10)
		    -- reaction speeds
		    local window = {(s+w1)*60000,(s+w2)*60000}
		    local RTs = reactiontimes_within_window(window, 
		                    reaction_times_ts[test],
		                    reaction_times[test])
		    -- output
		    if RTs:nElement() > 0 then
		    	return 1/torch.cinv(RTs):mean()
		    else
		    	return 0
		    end
		end
		local function extract_instantRT(test)
			local ts = reaction_times_ts[test]/600000*18000
			local rt = reaction_times[test]
			local N = rt:size(1)
			local RT0 = torch.Tensor(18000)
			for k=1,N do
				if k==1 then
					RT0[{{1,ts[k]}}] = rt[k]
				end
				if k==N then
					RT0[{{ts[k],-1}}] = rt[k]
				else
					local f1 = math.floor(ts[k] + math.min(30, (ts[k]+ts[k+1])/2))
					local f2 = math.ceil(ts[k+1] - math.min(30, (ts[k]+ts[k+1])/2))
					RT0[{{ts[k],f1}}] = rt[k]
					RT0[{{f2,ts[k+1]}}] = rt[k+1]
					RT0[{{f1,f2}}] = rt[k] + torch.range(0,f2-f1) *(rt[k+1]-rt[k]) /(f2-f1)
				end
			end
			return RT0
		end
		local medianRT_values = torch.FloatTensor(#reaction_times,18000,4):zero()
		for t=1,#reaction_times do
			local RT0_test = extract_instantRT(t)
			for i=1800,18000 do
				local RT0 = RT0_test[i]
				local RT15 = extract_medianRT(t,i/1800,-0.25,0.05)
				local RT30 = extract_medianRT(t,i/1800,-0.50,0.05)
				local RT60 = extract_medianRT(t,i/1800,-1.00,0.05)
				if RT0 > 0 and RT15 > 0 and RT30 > 0 and RT60 > 0 then
					medianRT_values[{t,i,{}}] = torch.FloatTensor{RT0,RT15,RT30,RT60}
				end
			end
		end
		return medianRT_values
	end




















	-- classificationError
	local function classificationError(set, noPrintFlag)
		local set = set or TS
		local pred = {torch.CudaTensor(0,2),torch.CudaTensor(0,2),torch.CudaTensor(0,2),torch.CudaTensor(0,2)}
		local gt = {torch.CudaTensor(0,2),torch.CudaTensor(0,2),torch.CudaTensor(0,2),torch.CudaTensor(0,2)}
		local RTs = torch.CudaTensor(0,4)
		if torch.type(set) == 'table' then -- crossvalidation
			for k = 1,#set do
				model[k]:evaluate()
				local p,g,RT = compute(set[k],model[k],opt.batch_size,1)
				for k=1,4 do
					pred[k] = torch.cat(pred[k],p[k],1)
					gt[k] = torch.cat(gt[k],g[k],1)
				end
				RTs = torch.cat(RTs,RT,1)
				model[k]:clearState():float()
			end
		else -- single fold
			model:evaluate()
			pred,gt,RTs = compute(set,model,opt.batch_size,1)
		end
		if not noPrintFlag then
			print('---- CONFUSION MATRICES pure ----')
			print(confusion(pred[1],gt[1],true))
			print(confusion(pred[2],gt[2],true))
			print(confusion(pred[3],gt[3],true))
			print(confusion(pred[4],gt[4],true))

			print('---- correlation with RTs ----')
			print('Correlation timescale=05s:', pearson(torch.cat(RTs[{{},1}],pred[1][{{},2}],2):totable()))
			print('Correlation timescale=15s:', pearson(torch.cat(RTs[{{},2}],pred[2][{{},2}],2):totable()))
			print('Correlation timescale=30s:', pearson(torch.cat(RTs[{{},3}],pred[3][{{},2}],2):totable()))
			print('Correlation timescale=60s:', pearson(torch.cat(RTs[{{},4}],pred[4][{{},2}],2):totable()))
		end


		return pred, gt, RTs
	end


	local function crossClassificationError(set,idx)
		local set = set or TS
		local idx = idx or {1,2,3,4}
		local pred,gt = classificationError(set,true)

		local N = 2^#idx
		require 'optim'
		local names = {'0','1'}
		for k=2,#idx do
			local n = {}
			for d=1,#names do
				table.insert(n, names[d] .. '0')
			end
			for d=1,#names do
				table.insert(n, names[d] .. '1')
			end
			names = n
		end
		local conf = optim.ConfusionMatrix(names)

		local function class(x,i)
			local c = 1
			for n,k in ipairs(idx) do
				c = c + 2^(n-1) * (x[k][{i,2}] > 0.5 and 1 or 0)
			end
			return c
		end
		for i=1,pred[1]:size(1) do
			local pred_class = class(pred,i)
			local gt_class = class(gt,i)
			conf:add(pred_class, gt_class)
		end
		return conf
	end

	local function anyDrowsinessClassificationError(set)
		local set = set or TS
		local pred,g = classificationError(set,true)
		local function class(x,i)
			local c = 1
			for n=1,4 do
				c = c + (x[n][{i,2}] > 0.5 and 1 or 0)
			end
			return c
		end
		local conf = torch.zeros(5,5)
		local gt = torch.zeros(g[1]:size())
		for i=1,gt:size(1) do
			local p = torch.FloatTensor{g[1][{i,2}],g[2][{i,2}],g[3][{i,2}],g[4][{i,2}]}
			local c = class(pred,i)
			if torch.any(p:eq(1)) then
				local c_gt = 1+p:eq(1):sum()
				conf[{c_gt,c}] = conf[{c_gt,c}]+1 
			elseif torch.all(p:eq(0)) then
				conf[{1,c}] = conf[{1,c}]+1
			end
		end
		print(conf)
		print('TNR= ' .. conf[{1,1}] / conf[{1,{}}]:sum() .. '%')
		print('TPR= ' .. conf[{{2,5},{2,5}}]:sum() / conf[{{2,5},{}}]:sum() .. '%')
	end



	-- return all functions
	return {
		classificationError = classificationError,
		crossClassificationError = crossClassificationError,
		anyDrowsinessClassificationError=anyDrowsinessClassificationError,
	}
end

return M