require 'SigmoidPooling'

model = nn.SigmoidPooling(5,3)
n=150
input = torch.Tensor(2,3,n)
input[1] = torch.Tensor{torch.linspace(-500,499,n):totable(),torch.linspace(-200,799,n):totable(),torch.linspace(2,4,n):totable()}
input[2] = torch.Tensor{torch.linspace(-500,499,n):totable(),torch.linspace(-200,799,n):totable(),torch.linspace(2,4,n):totable()}
input = input:view(2,3,1,n)
gradOutput = torch.Tensor{{2,1,1},{-1,0,1}}

tic = torch.tic()
k = 1000
for i=1,k do
	model:forward(input)
	model:backward(input,gradOutput)
end
time = torch.toc(tic) *1000 / k
print('speed='.. time .. 'ms for length of ' .. n)
print(model:forward(input))
print(model:backward(input,gradOutput))

print(nn.Jacobian.testJacobian(model,input))


require 'gnuplot'
gnuplot.plot(model.w[{1,1,{}}])