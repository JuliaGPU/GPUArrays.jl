using GPUArrays, CUDAnative
import GPUArrays: GPUArray, CUBackend, cu_map
cuctx = CUBackend.init()
const cu = CUDAnative

A = GPUArray(rand(Float32, 40, 40));

A .= identity.(10f0)
const cu = Base
function test(a, b)
    x = cu.sqrt(cu.sin(a) * b) / 10
    y = 33x + cu.cos(b)
    y*10
end

test.(A, 10f0)
