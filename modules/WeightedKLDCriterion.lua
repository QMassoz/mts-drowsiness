require 'nn'
local WeightedKLDCriterion, parent = torch.class('nn.WeightedKLDCriterion', 'nn.Criterion')

function WeightedKLDCriterion:__init(weight, sizeAverage)
	parent.__init(self)
	assert(weight:dim()==1)
	self.weight = weight
	if sizeAverage ~= nil then
		self.sizeAverage = sizeAverage
	else
		self.sizeAverage = true
	end
end

function WeightedKLDCriterion:updateOutput(input,target)
	assert(input:size(2) == target:size(2))
	--self.output = - torch.log(input+1e-12):cmul(target) * self.weight:view(-1,1) -- iif target contains only 1's and 0's
	self.output = - torch.log(torch.cdiv(input+1e-12,target+1e-12)):cmul(target):sum(2)
	-- weight
	for n = 1, self.output:size(1) do
		for k = 1, self.weight:size(1) do
			if target[{n,k}] > 0.5 then
				self.output[n] = self.output[n] * self.weight[k]
			end
		end
	end
	self.output = self.output:sum()

	if self.sizeAverage then
		self.output = self.output / input:size(1)
	end
	return self.output
end

function WeightedKLDCriterion:updateGradInput(input, target)
	error('backward is not implemented!')
	self.gradInput = input
	return self.gradInput
end