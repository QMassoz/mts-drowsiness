require 'nn'
local SigmoidPooling, parent = torch.class('nn.SigmoidPooling', 'nn.Module')
-- SigmoidPooling is a pooling function

function SigmoidPooling:__init(t0,s)
	parent.__init(self)
	assert(t0)
	self.output = torch.Tensor()
	self.gradInput = torch.Tensor()

	self.s = s or 1
	self.t0 = t0

	self.w = torch.Tensor()
end

function SigmoidPooling:updateOutput(input)
	local b=input:size(1)
	local c=input:size(2)
	local T=input:size(4)

	local s = self.s
	local t0 = T-self.t0

	local x = input:view(b,c,T) -- x [bxcxT]
	local t = torch.range(1,T):typeAs(input) 		 		 -- t [T]
	local sig = torch.cinv(1 + torch.exp(-s*(t-t0))) 		 -- sig [T]
	local w = sig / sig:sum()				-- w/sum(w) [T]
	self.w = w:view(1,1,T):expand(b,c,T)	-- w/sum(w) [b,c,T]
	self.output = torch.cmul(x,self.w):sum(3):view(b,c)
	return self.output
end

function SigmoidPooling:updateGradInput(input, gradOutput)
	local b=input:size(1)
	local c=input:size(2)
	local T=input:size(4)

	self.gradInput:resizeAs(self.w):copy(self.w)  	 		  -- gradInput = w/sum(w)       	   [bxcxT]
	self.gradInput:cmul(gradOutput:contiguous():view(b,c,1):expand(b,c,T)) -- gradInput = w/sum(w) * gradOutput [bxcxT]
	self.gradInput = self.gradInput:view(b,c,1,T) 	 		  -- gradInput =         " "           [bxcx1xT]
	return self.gradInput
end


function SigmoidPooling:clearState()
    if self.w then self.w:set() end
    return parent.clearState(self)
end