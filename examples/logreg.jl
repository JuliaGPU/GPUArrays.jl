using ReverseDiff, GPUArrays
using ReverseDiff: GradientConfig, gradient

f(W, b, x) = (+).(W*x, b)
function softmax(xs)
    (/).(exp.(xs), sum(exp.(xs)))
end
mse(x, y) = mean((^).((-).(x, y), 2))

function net(W, b, x, y)
  ŷ = softmax(f(W, b, x))
  mse(ŷ, y)
end
JLBackend.init()

W = GPUArray(randn(Float32, 5, 10))
b = GPUArray(randn(Float32, 5))
x = GPUArray(rand(Float32, 10))
y = GPUArray(rand(Float32, 5))
inputs = (W, b, x, y)
net(inputs...)

gradient(net, inputs)
