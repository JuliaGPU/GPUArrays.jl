using ReverseDiff, GPUArrays
using ReverseDiff: GradientConfig, gradient

f(W, b, x) = W*x + b
softmax(xs) = exp.(xs)/sum(exp(xs))
mse(x, y) = mean((x .- y).^2)

function net(W, b, x, y)
  ŷ = softmax(f(W, b, x))
  mse(ŷ, y)
end
JLBackend.init()

W = GPUArray(randn(5, 10))
b = GPUArray(randn(5))
x = GPUArray(rand(10))
y = GPUArray(rand(5))
inputs = (W, b, x, y)
net(inputs...)

gradient(net, inputs)
