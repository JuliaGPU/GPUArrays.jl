using GPUArrays
using GPUArrays: BroadcastDescriptor
CLBackend.init()

a = GPUArray(rand(Float32, 10, 10, 10))
b = GPUArray(rand(Float32, 10))
lm = broadcast(+, a, b);
Array(lm) == Array(a) .+ Array(b)
