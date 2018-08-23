function load_timestamps(file)
	require 'csvigo'
	-- Load file
	local f = csvigo.load{path=file,separator=' ',mode='raw', header='false',verbose=false}
	f = torch.Tensor(f)
	-- Return relative timestamps [ms] of the frames
	return f[{{}, 8}]
end

function parse_timestamps_string(str)
	local t = {}
	for v in string.gmatch(str,"%d+") do
		t[#t+1] = v
	end
	return t
end

function load_reactiontimes(file)
	require 'csvigo'
	-- Load file
	local f = csvigo.load{path=file,separator=';',mode='raw', header='false', verbose=false}
	-- Parse reference timestamp
	local T0 = parse_timestamps_string(f[1][1])
	T0 = torch.Tensor(T0):narrow(1,4,4)
	-- Parse start and end of the stimuli
	local Tstart, Tend = {}, {}
	for i=2,#f do
		table.insert(Tstart, parse_timestamps_string(f[i][1]))
		table.insert(Tend, parse_timestamps_string(f[i][2]))
	end
	Tstart = torch.Tensor(Tstart):narrow(2,4,4)
	Tend = torch.Tensor(Tend):narrow(2,4,4)
	-- Compute appearance time [ms] relative to the reference timestamps
	local T0exp = T0:view(1,4):expand(Tstart:size(1),4)
	local Tappearance = (Tstart - T0exp) * torch.Tensor({ {3600000}, {60000}, {1000}, {1} })
	-- Compute reaction times [ms]
	local RTs = (Tend - Tstart) * torch.Tensor({ {3600000}, {60000}, {1000}, {1} })
	-- Return (1) appearance times [ms] and (2) the reaction times [ms] of the stimuli
	return Tappearance:squeeze(), RTs:squeeze(), T0
end

function reactiontimes_within_window(window, RT_ts, RT)
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

DataLoader = {}
function DataLoader:load(main_folder, subjects, meanRS_pop, stdRS_pop)
	require 'paths'
    require 'xlua'

	-- Initialization
    local main_folder = main_folder or 'data/raw'
    if not paths.dirp(main_folder) then
        error(main_folder .. ' does not exist!')
    end
    print(sys.COLORS.green .. '<eye2multid-build> loading data from folder ' .. main_folder .. sys.COLORS.white)
    local norm_targets = (meanRS_pop and stdRS_pop) or false


    -- get eye_values
    print('<eye2multid-build> loading eye values from ' .. #subjects .. ' subjects:')
    local eye_values = {}
    require 'image'
    for i, subject in ipairs(subjects) do
    	xlua.progress(i,#subjects) -- progress bar
    	if not norm_targets or paths.filep(paths.concat(main_folder, 'rt/'..subject..'-1.txt')) then -- PVT1 must exist if normalizing targets
	    	for _,test in pairs({1,2,3}) do
	    		local filename = paths.concat(main_folder, 'eld-seq/'..subject..'-' .. test .. '.t7')
	    		if paths.filep(filename) then -- PVT must exist
	    			local eye_values_test = torch.load(filename)
					table.insert(eye_values, eye_values_test:totable())
				end
	    	end
	    end
    end
    eye_values = torch.Tensor(eye_values)

    -- get reaction times
    print('<eye2multid-build> loading reaction times:')
    local reaction_times = {}
    local reaction_times_ts = {}

    local table_of_subjects = {} -- subject (internal) id for normalization purpose (same number of lines than eye_values, reaction_times, and reaction_times_ts)
    local subjects_RSmeans={}
    local subjects_RSstds={}

    local idx = 0
    for i, subject in ipairs(subjects) do
    	xlua.progress(i,#subjects) -- progress bar
    	if not norm_targets or paths.filep(paths.concat(main_folder, 'rt/'..subject..'-1.txt')) then -- PVT1 must exist if normalizing targets
	    	idx = idx + 1
	    	for _,test in pairs({1,2,3}) do
	    		local rt_filename =  paths.concat(main_folder, 'rt/'..subject..'-' .. test .. '.txt')
	    		if paths.filep(rt_filename) then -- PVT must exist
	    			local rt_ts, rt, _ = load_reactiontimes(rt_filename)
	    			table.insert(reaction_times, rt:float())
	    			table.insert(reaction_times_ts, rt_ts:float())

	    			if norm_targets then
		    			table.insert(table_of_subjects, idx) -- subject !internal! index
		    			if test==1 then
		    				table.insert(subjects_RSmeans, torch.cinv(rt):mean())
		    				table.insert(subjects_RSstds, torch.cinv(rt):std())
		    			end
		    		end
	    		end
	    	end
	    end
    end
    assert(#reaction_times == eye_values:size(1))
    if norm_targets then
	    assert(#table_of_subjects == #reaction_times)
	    assert(torch.Tensor(table_of_subjects):max() == #subjects_RSmeans)
	end

	-- normalize reaction times
	if norm_targets then
		for i = 1,#reaction_times do
			reaction_times[i]:cinv()
				:csub(subjects_RSmeans[table_of_subjects[i]])
				--:div(subjects_RSstds[table_of_subjects[i]])
				--:mul(stdRS_pop)
				:add(meanRS_pop)
				:cinv()
		end
	end

	-- generate inputs, targets
	print('<eye2multid-build> generating inputs and targets:')
	local inputs, targets = {}, {}
    for test=1,eye_values:size(1) do
    	xlua.progress(test,eye_values:size(1)) -- progress bar
    	for n=1,reaction_times_ts[test]:size(1) do
    		local timestamp = math.floor(reaction_times_ts[test][n]/600000*18000)
    		if timestamp >= 1800 then
	    		-- input
	    		local i_end   = timestamp
	    		local i_start = i_end - 1800 +1
	    		local input_c = eye_values[{ test,{i_start,i_end},{} }]
	    		input_c = input_c:t():contiguous():view(2,1,1800)
	       		table.insert(inputs,input_c:totable())
	    		-- target
	    		local RT0 = reaction_times[test][n]
	    		local s = timestamp/1800
	    		local RT15= reactiontimes_within_window({(s-.25)*60000, (s+0.05)*60000}, reaction_times_ts[test], reaction_times[test])
	    		local RT30= reactiontimes_within_window({(s-.5)*60000, (s+0.05)*60000}, reaction_times_ts[test], reaction_times[test])
	    		local RT75= reactiontimes_within_window({(s-1)*60000, (s+0.05)*60000}, reaction_times_ts[test], reaction_times[test])
	    		RT15 = 1/torch.cinv(RT15):mean()
	    		RT30 = 1/torch.cinv(RT30):mean()
	    		RT75 = 1/torch.cinv(RT75):mean()
	    		table.insert(targets,{RT0,RT15,RT30,RT75})
	    	end
    	end
	end

	inputs = torch.FloatTensor(inputs)
	targets = torch.FloatTensor(targets)


    -- return data
    local newObj = {}

    newObj.inputs = inputs
    newObj.targets = targets
    	-- eye_values to compute the inputs sequence
    newObj.eye_values = eye_values
   		-- reaction_times, and reaction_times_ts to compute the label
    newObj.reaction_times = reaction_times
    newObj.reaction_times_ts = reaction_times_ts
    	-- test2subject, subjects_RSmean, and subjects_RSstd used to normalize the RTs
    newObj.test2subject = table_of_subjects
    newObj.subjects_RSmean = torch.FloatTensor(subjects_RSmeans)
    newObj.subjects_RSstd = torch.FloatTensor(subjects_RSstds)

    self.__index = self
    return setmetatable(newObj, self)
end
function DataLoader:remove_mean_inputs(mean)
    print('<eye2multid-build> global normalization of the inputs')
    local mean = mean or {}
    local nChannels = self.inputs:size(2)
    -- compute mean
    if next(mean) == nil then
        for i=1,nChannels do
           mean[i] = self.inputs[{ {},i,{},{} }]:mean()
           self.inputs[{ {},i,{},{} }]:add(-mean[i])
        end
    -- use given mean
    else
        for i=1,nChannels do
            self.inputs[{ {},i,{},{} }]:add(-mean[i])
        end
    end
    return mean
end

function DataLoader:produce_medianRT()
	print("<eye2multid-build> producing medianRT_values")
	local function extract_medianRT(test, s, w1, w2)
	    --[[
	        test = ID of the test
	        s    = anchor 'timestamp' in minutes (from 1 to 10)
	    --]]
	    -- reaction speeds
	    local window = {(s+w1)*60000,(s+w2)*60000}
	    local RTs = reactiontimes_within_window(window, 
	                    self.reaction_times_ts[test],
	                    self.reaction_times[test])
	    -- output
	    if RTs:nElement() > 0 then
	    	return 1/torch.cinv(RTs):mean()
	    else
	    	return 0
	    end
	end
	local function extract_instantRT(test)
		local ts = self.reaction_times_ts[test]/600000*18000
		local rt = self.reaction_times[test]
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

	self.medianRT_values = torch.FloatTensor(self.eye_values:size(1),18000,4):zero()
	for t=1,self.eye_values:size(1) do
		local RT0_test = extract_instantRT(t)
		for i=1800,18000 do
			local RT0 = RT0_test[i]
			local RT15 = extract_medianRT(t,i/1800,-0.25,0.05)
			local RT30 = extract_medianRT(t,i/1800,-0.50,0.05)
			local RT60 = extract_medianRT(t,i/1800,-1.00,0.05)
			if RT0 > 0 and RT15 > 0 and RT30 > 0 and RT60 > 0 then
				self.medianRT_values[{t,i,{}}] = torch.FloatTensor{RT0,RT15,RT30,RT60}
			end
		end
	end
	return self.medianRT_values
end


function sets_difference(a, b)
    local aa = {}
    for k,v in pairs(a) do aa[v]=true end
    for k,v in pairs(b) do aa[v]=nil end
    local ret = {}
    local n = 0
    for k,v in pairs(a) do
        if aa[v] then n=n+1 ret[n]=v end
    end
    return ret
end

local M = {}
require 'paths'
function M.build(opt, cacheFile)
	torch.setnumthreads(1)
	print("<eld2multid_CV-build> building data...")
	local all_subjects = {1,2,3,4,5,6,8,10,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,33,34,35}
	local crossval_splits = {
		{17,26,33,35,12},--1 
		{12,13,16,23,18},--2
		{12,23,34,35,13},--3
		{ 1, 8,26,27,30},--4
		{ 1,16,18,20,35},--5
		{ 8,22,26,28,33},--6
		{18,20,27,35,23},--8
		{ 3,16,25,29, 6},--10
		{22,25,30,34,14},--12
		{ 2,12,14,15,20},--13
		{15,18,19,21, 5},--14
		{ 1, 5, 6, 8, 2},--15
		{ 2, 6,19,21,29},--16
		{13,26,28,35,16},--17
		{ 1, 4,14,16,21},--18
		{ 6,17,33,34,25},--19
		{ 2, 5,10,15,26},--20
		{10,15,19,33, 4},--21
		{ 5,13,23,30,10},--22
		{ 4,10,12,18,34},--23
		{ 3,22,29,34,17},--25
		{ 8,23,25,27, 3},--26
		{10,13,21,29,28},--27
		{ 3,17,30,33,27},--28
		{14,19,20,28, 1},--29
		{ 3,20,22,25,15},--30
		{ 2, 5, 6,14, 8},--33
		{ 4,21,28,30,19},--34
		{ 4,17,27,29,22} --35
	}

	print('* COMPUTING POPULATION STATISTICS')
	local population = DataLoader:load('data/raw', all_subjects,1,1)
	local mean_pop = population.subjects_RSmean:mean()
	local std_pop = 0--population.subjects_RSstd:mean()
	print('mean', mean_pop)
	print('std', std_pop)
	
	local train,valid,test = {},{},{}
	for k=1,#crossval_splits do
		-- load data
		local val_subjects = crossval_splits[k]
		local train_subjects = sets_difference(all_subjects,val_subjects)
		train_subjects = sets_difference(train_subjects, {all_subjects[k]})
		print('. cross validation ' .. k)
		local LS = DataLoader:load('data/raw', train_subjects, mean_pop, std_pop)
		local VS = DataLoader:load('data/raw', val_subjects, mean_pop, std_pop)
		local TS = DataLoader:load('data/raw', {all_subjects[k]}, mean_pop, std_pop)
		-- remove inputs mean
		LS.mean = LS.inputs:mean()
		VS.mean = LS.inputs:mean()
		TS.mean = LS.inputs:mean()
		LS.inputs:csub(LS.mean)
		VS.inputs:csub(LS.mean)
		TS.inputs:csub(LS.mean)
		-- produce medianRT()
		LS:produce_medianRT()

		table.insert(train,LS)
		table.insert(valid,VS)
		table.insert(test,TS)
	end

	if cacheFile then
		print('<eld2multid_CV-build> saving binary ' .. cacheFile)
		torch.save(cacheFile, {
			train = train,
			valid = valid,
			test = test,
			crossval_splits=crossval_splits,
			mean_pop=mean_pop,
			std_pop=std_pop,
			inputSize={2,1,1800} 
		})
	end
end

return M
