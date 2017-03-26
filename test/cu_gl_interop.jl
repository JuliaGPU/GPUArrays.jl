using GLAbstraction
using ModernGL, CUDAnative
using JTensors
import JTensors: JTensor, GLBackend, CUBackend, cu_map

cuctx = CUBackend.init()
glctx = GLBackend.init()

a = JTensor(zeros(Float32, 50, 50), glctx, usage=GL_DYNAMIC_DRAW)

@target ptx function kernel_vadd(a)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if length(a) >= i
        @inbounds a[i] = a[i] + 10f0
    end
    return nothing
end

cu_map(a) do cu_arr
    len = length(cu_arr)
    threads = min(len, 1024)
    blocks = ceil(Int, len/threads)
    @show threads blocks
    @cuda (blocks, threads) kernel_vadd(buffer(cu_arr))
end
Array(a)

tex = JTensor(Texture(zeros(Float32, 50, 50), minfilter=:nearest, x_repeat=:clamp_to_edge), (50,50), glctx)

cu_map(tex) do cu_arr
    len = length(cu_arr)
    threads = min(len, 1024)
    blocks = ceil(Int, len/threads)
    @cuda (blocks, threads) kernel_vadd(buffer(cu_arr))
    println(Array(cu_arr))
end
