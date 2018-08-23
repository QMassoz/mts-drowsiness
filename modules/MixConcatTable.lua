require 'nn'
nninit = require 'nninit'
local MixConcatTable, parent = torch.class('nn.MixConcatTable', 'nn.Container')

function MixConcatTable:__init(N,n_input,n_output)
   parent.__init(self)
   self.N = N
   self.n_input = n_input
   self.n_output = n_output

   self.mix_module = nn.Sequential()
      :add(nn.JoinTable(2))
      :add(nn.Linear(N*n_input,n_output)
         :init('weight',nninit.kaiming,{gain='relu'})
         :init('bias',nninit.constant,0.01))
      :add(nn.ReLU(true))

   self.solo_module = {}
   for k=1,N do
      self.solo_module[k] = nn.Sequential()
         :add(nn.Linear(n_input,n_output)
            :init('weight',nninit.kaiming,{gain='relu'})
            :init('bias',nninit.constant,0.01))
         :add(nn.ReLU(true))
   end


   self.modules = {self.mix_module,table.unpack(self.solo_module)}
end

function MixConcatTable:updateOutput(input)
   assert(torch.type(input)=='table')

   local mix_output = self.mix_module:updateOutput(input)
   self.output = {}
   for k=1,self.N do
      self.output[k] = self.solo_module[k]:updateOutput(input[k])
      self.output[k] = torch.cat(self.output[k],mix_output,2)
   end

   return self.output
end

function MixConcatTable:updateGradInput(input, gradOutput)
   -- solo
   self.gradInput = {}
   for k=1,self.N do
      self.gradInput[k] = self.solo_module[k]:updateGradInput(input[k],gradOutput[k][{{},{1,self.n_output}}])
   end
   -- mix
   local gradOutput_mix = gradOutput[1][{{},{-self.n_output,-1}}]
   for k=2,self.N do
      gradOutput_mix:add(gradOutput[k][{{},{-self.n_output,-1}}])
   end
   local gradInput_mix = self.mix_module:updateGradInput(input,gradOutput_mix)
   -- mix to solo
   for k=1,self.N do
      self.gradInput[k] = self.gradInput[k] + gradInput_mix[k]
   end

   return self.gradInput
end

function MixConcatTable:accGradParameters(input, gradOutput,scale)
   -- solo 
   for k=1,self.N do
      self.solo_module[k]:accGradParameters(input[k],gradOutput[k][{{},{1,self.n_output}}],scale)
   end
   -- mix
   local gradOutput_mix = gradOutput[1][{{},{-self.n_output,-1}}]
   for k=2,self.N do
      gradOutput_mix:add(gradOutput[k][{{},{-self.n_output,-1}}])
   end
   self.mix_module:accGradParameters(input,gradOutput_mix,scale)

end

function MixConcatTable:__tostring__()
   local str = torch.type(self) .. '::\n' .. self.mix_module:__tostring__() .. '::\n' .. self.solo_module[1]:__tostring__()
   return str
end
