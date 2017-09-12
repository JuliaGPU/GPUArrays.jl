using GPUArrays, CLArrays, OpenCL
import Transpiler: cli
# TODO generalize to CUDAnative
function transpose_kernel!(
        state, At, A, width, height, A_local, ::Val{BLOCK}
    ) where BLOCK

    ui1 = UInt32(1)
    bidx_x = blockidx_x(state) - ui1
    bidx_y = blockidx_y(state) - ui1
    tidx_x = threadidx_x(state) - ui1
    tidx_y = threadidx_y(state) - ui1

    base_idx_a = bidx_x * BLOCK + bidx_y * (BLOCK * width)
    base_idx_a_t = bidx_y * BLOCK + bidx_x * (BLOCK * height)

    glob_idx_a = base_idx_a + tidx_x + width * tidx_y
    glob_idx_a_t = base_idx_a_t + tidx_x + height * tidx_y

    A_local[tidx_y * BLOCK + tidx_x + ui1] = A[glob_idx_a + ui1]

    cli.barrier(cli.CLK_LOCAL_MEM_FENCE)
    At[glob_idx_a_t + ui1] = A_local[tidx_x * BLOCK + tidx_y + ui1]
    return
end

function max_block_size(dev, h::Int, w::Int)
    dim1, dim2 = GPUArrays.blocks(dev)[1:2]
    wgsize = GPUArrays.threads(dev)
    wglimit = floor(Int, sqrt(wgsize))
    return gcd(dim1, dim2, h, w, wglimit)
end

function Base.transpose!{T}(At::GPUArray{T, 2}, A::GPUArray{T, 2})
    dev = GPUArrays.device(A)
    block_size = max_block_size(dev, size(A)...)
    outsize = UInt32.(size(At))
    lmem = GPUArrays.LocalMemory{T}(block_size * (block_size + 1))
    args = (At, A, outsize..., lmem, Val{block_size}())
    gpu_call(transpose_kernel!, At, args, (block_size, block_size))
    At
end
CLArrays.cl_empty_compile_cache!()
N = 256
a, b = rand(Float32, N, N), rand(Float32, N, N)
B = CLArray(b)
At = transpose!(zeros(CLArray{Float32}, N, N), B)
