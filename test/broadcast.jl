using Iterators
using GPUArrays


g1 = GPUArray(rand(Float32, 4, 5, 3));
g2 = GPUArray(rand(Float32, 1, 5, 3));

broadcast(+, g1, g2)


g3 = GPUArray(rand(Float32, 1, 5, 1)); a3 = Array(g3);

isapprox(Array(g1 .+ g3), a1 .+ a3)
