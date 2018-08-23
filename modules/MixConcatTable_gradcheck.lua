require 'nn'
require 'cunn'
require 'modules/MixConcatTable'

n = 16
n_input = 5
n_output = 10

input = {torch.randn(n,n_input),torch.randn(n,n_input),torch.randn(n,n_input),torch.randn(n,n_input)}
gradOutput = {torch.randn(n,n_output*2),torch.randn(n,n_output*2),torch.randn(n,n_output*2),torch.randn(n,n_output*2)}


model = nn.MixConcatTable(#input,n_input,n_output)

print(model:forward(input))
print(model:backward(input,gradOutput))

local function checkgrad(f, g, x, go, eps)
  -- compute true gradient
  f(x)
  local grad = g(x,go)
  
  -- compute numeric approximations to gradient
  local eps = eps or 1e-5
  local diff = {}
  for k=1,#grad do
	  local grad_est = {}
	  for l=1,#x do
	  	grad_est[l]=torch.DoubleTensor(grad[k]:size())
		  for i = 1, grad[k]:size(1) do
		    for j=1, grad[k]:size(2) do

		      x[l][{i,j}] = x[l][{i,j}] + eps
		      local fpe = f(x)[k]
		      x[l][{i,j}] = x[l][{i,j}] - 2*eps
		      local fne = f(x)[k]
		      x[l][{i,j}] = x[l][{i,j}] + eps
		      grad_est[l][{i,j}] = (fpe - fne) / (2*eps)
		    end
		end
	  end

	  -- computes (symmetric) relative error of gradient
	  diff[k] = torch.norm(grad[k] - grad_est[k]) / torch.norm(grad[k] + grad_est[k])
	end
  return diff, grad, grad_est
end

-- returns loss(params)
f = function(x)
  return model:forward(x)
end
-- returns dloss(params)/dparams
g = function(x,gradOutput)
  return model:backward(x, gradOutput)
end
diff,grad,grad_est = checkgrad(f, g, input, gradOutput)
print(diff)