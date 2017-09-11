function transpose_kernel!{BLOCK}(
        At, A, width, height, A_local, ::Val{BLOCK}
    )
    u1 = Cuint(1)
    base_idx_a = blockidx_x(A) * BLOCK + blockidx_y(A) * (BLOCK * width)
    base_idx_a_t = blockidx_y(A) * BLOCK + blockidx_x(A) * (BLOCK * height)

    glob_idx_a = base_idx_a + threadidx_x(A) + width * threadidx_y(A)
    glob_idx_a_t = base_idx_a_t + threadidx_x(A) + height * threadidx_y(A)

    A_local[threadidx_y(A) * BLOCK + threadidx_x(A) + u1] = A[glob_idx_a + u1]

    synchronize_threads(A)

    At[glob_idx_a_t + u1] = A_local[threadidx_x(A) * BLOCK + threadidx_y(A) + u1]
    return
end

function Base.transpose!{T}(At::CLArray{T, 2}, A::CLArray{T, 2})
    ctx = context(A)
    block_size = cl.max_block_size(ctx.queue, size(A, 1), size(A, 2))
    outsize = map(Int32, size(At))
    lmem = GPUArrays.LocalMemory{Float32}(block_size * (block_size + 1))
    args = (At, A, outsize..., lmem, Val{block_size}())
    func = GPUArrays.CLFunction(At, transpose_kernel!, args...)
    func(args, outsize, (block_size, block_size))
    At
end
