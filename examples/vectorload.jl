using GPUArrays
using GPUArrays.CLBackend
using Transpiler.CLTranspiler: vstore, vload
using StaticArrays

ctx = CLBackend.init()
# Simple example, of how we can use Types to specialize our kernel code
function vectormap!{N, T}(f, out, a, b, V::Type{SVector{N, T}})
    i = linear_index(out) # get the kernel index it gets scheduled on
    vec = f(vload(V, a, i), vload(V, b, i))
    vstore(vec, out, i)
    return
end
# we need a `guiding array`, to get the context and indicate on what size we
# want to execute the kernel! This kind of scheme might change in the future


# same here, x is just passed to supply a kernel size!
function testv{N}(f, out, a, b, ::Val{N})
    gpu_call(out, vectormap!, (f, out, a, b, SVector{N, Float32}), length(out) ÷ N)
    GPUArrays.synchronize(out)
end
testn(f, out, a, b) = (out .= f.(a, b); GPUArrays.synchronize(out))
using BenchmarkTools
const N = 8 * 10^6
a = GPUArray(rand(Float32, N))
b = GPUArray(rand(Float32, N))
out = similar(a)
function test{T}(a::T, b)
    x = sqrt.(sin.(a) .* b) ./ 10f0
    y = 33f0 .* x + cos.(b)
    y .* 10f0
end

# deciding what vector size to use becomes very easy now!
b16 = @benchmark testv(test, $out, $a, $b, $(Val{16}()))
b8 = @benchmark testv(test, $out, $a, $b, $(Val{8}()))
b4 = @benchmark testv(test, $out, $a, $b, $(Val{4}()))
bbase = @benchmark testn(test, $out, $a, $b)
