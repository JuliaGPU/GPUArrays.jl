using GPUArrays, CUDAnative
import GPUArrays: GPUArray, CUBackend, cu_map
cuctx = CUBackend.init()
const cu = CUDAnative
using Base.Test

for dims in ((4048,), (1024,1024), (77,), (1923,209))
    for T in (Float32,)
        A = GPUArray(rand(Float32, (4048,)))
        @test sum(A) ≈ sum(Array(A))
        @test maximum(A) ≈ maximum(Array(A))
        @test minimum(A) ≈ minimum(Array(A))
        @test sumabs(A) ≈ sumabs(Array(A))
        @test prod(A) ≈ prod(Array(A))
    end
end



dev = CuDevice(0)
ctx = CuContext(dev)
function test(a, b)
    a + b
end
@target ptx function test2(out, a, b)
    out[1] = test(a,b)
    nothing
end
using CUDAnative, CUDAdrv

dev = CuDevice(0)
ctx = CuContext(dev)
d_out = CuArray(Float32, 1)
@cuda (1,1) kernel(d_out, 1f0, 1f0)

@target ptx function kernel(out)
    wid, lane = fldmod1(threadIdx().x, warpsize)
    out[1] = wid
    nothing
end


Array(d_out)
