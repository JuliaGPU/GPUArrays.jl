
# function transpose_kernel!(
#         state, At, A::AbstractArray{T}, width, height, ::Val{BLOCK}, ::Val{LMem}
#     ) where {BLOCK, LMem, T}
#
#     bidx_x = blockidx_x(state) - 1
#     bidx_y = blockidx_y(state) - 1
#     tidx_x = threadidx_x(state) - 1
#     tidx_y = threadidx_y(state) - 1
#
#     A_local = @LocalMemory(state, T, LMem)
#
#     base_idx_a = bidx_x * BLOCK + bidx_y * (BLOCK * width)
#     base_idx_a_t = bidx_y * BLOCK + bidx_x * (BLOCK * height)
#
#     glob_idx_a = base_idx_a + tidx_x + width * tidx_y
#     glob_idx_a_t = base_idx_a_t + tidx_x + height * tidx_y
#     glob_idx_a >= length(A) && return
#     A_local[tidx_y * BLOCK + tidx_x + 1] = A[glob_idx_a + 1]
#     synchronize_threads(state)
#     At[glob_idx_a_t + 1] = A_local[tidx_x * BLOCK + tidx_y + 1]
#     return
# end

function transpose_blocks!(
        state, odata::AbstractArray{T}, idata, ::Val{SHMEM}, ::Val{TDIM}, ::Val{BLOCK_ROWS}, ::Val{NROW}
    ) where {T, SHMEM, TDIM, BLOCK_ROWS, NROW}

    tile = @LocalMemory(state, T, SHMEM)
    bidx_x = blockidx_x(state) - 1
    bidx_y = blockidx_y(state) - 1
    tidx_x = threadidx_x(state) - 1
    tidx_y = threadidx_y(state) - 1

    x = bidx_x * TDIM + tidx_x + 1
    y = bidx_y * TDIM + tidx_y + 1
    dims = size(idata)

    (x <= dims[2] && (y + (BLOCK_ROWS * 3)) <= dims[1]) || return

    for j = 0:3
        j0 = j * BLOCK_ROWS
        @inbounds tile[tidx_x + 1, tidx_y + j0 + 1] = idata[y + j0, x]
    end

    synchronize_threads(state)
    for j = 0:3
        j0 = j * BLOCK_ROWS
        @inbounds odata[x, y + j0] = tile[tidx_x + 1, tidx_y + j0 + 1]
    end

    return
end

function transpose!(At::GPUArray{T, 2}, A::GPUArray{T, 2}) where T
    if size(A, 1) == size(A, 2) && all(x-> x % 32 == 0, size(A))
        outsize = size(At)
        TDIM = 32; BLOCK_ROWS = 8
        nrows = TDIM ÷ BLOCK_ROWS
        shmemdim = (TDIM, (TDIM + 1))
        static_params = map(x-> Val(x), (shmemdim, TDIM, BLOCK_ROWS, nrows))
        args = (At, A, static_params...)

        griddim = ceil.(Int, size(A) ./ (TDIM, TDIM))
        blockdim = (TDIM, BLOCK_ROWS)
        # optimized version for 32x & square dimensions
        gpu_call(transpose_blocks!, At, args, (griddim, blockdim))
    else
        # simple fallback
        gpu_call(At, (At, A)) do state, At, A
            idx = @cartesianidx A state
            @inbounds At[idx[2], idx[1]] = A[idx[1], idx[2]]
            return
        end
    end
    At
end

function genperm(I::NTuple{N}, perm::NTuple{N}) where N
    ntuple(d-> (@inbounds return I[perm[d]]), Val(N))
end

function permutedims!(dest::GPUArray, src::GPUArray, perm::NTuple{N, Integer}) where N
    gpu_call(dest, (dest, src, perm)) do state, dest, src, perm
        I = @cartesianidx src state
        @inbounds dest[genperm(I, perm)...] = src[I...]
        return
    end
    return dest
end


function copyto!(A::AbstractArray, B::Adjoint{T, <: GPUArray}) where T
    copyto!(A, Adjoint(Array(B.parent)))
end
function copyto!(A::GPUArray, B::Adjoint{T, <: GPUArray}) where T
    transpose!(A, B.parent)
end
