import Base: transpose!, permutedims!
# function transpose_kernel!(
#         state, At, A::AbstractArray{T}, width, height, ::Val{BLOCK}, ::Val{LMem}
#     ) where {BLOCK, LMem, T}
#
#     ui1 = UInt32(1)
#     bidx_x = blockidx_x(state) - ui1
#     bidx_y = blockidx_y(state) - ui1
#     tidx_x = threadidx_x(state) - ui1
#     tidx_y = threadidx_y(state) - ui1
#
#     A_local = @LocalMemory(state, T, LMem)
#
#     base_idx_a = bidx_x * BLOCK + bidx_y * (BLOCK * width)
#     base_idx_a_t = bidx_y * BLOCK + bidx_x * (BLOCK * height)
#
#     glob_idx_a = base_idx_a + tidx_x + width * tidx_y
#     glob_idx_a_t = base_idx_a_t + tidx_x + height * tidx_y
#     glob_idx_a >= length(A) && return
#     A_local[tidx_y * BLOCK + tidx_x + ui1] = A[glob_idx_a + ui1]
#     synchronize_threads(state)
#     At[glob_idx_a_t + ui1] = A_local[tidx_x * BLOCK + tidx_y + ui1]
#     return
# end

function transpose_blocks!(
        state, odata::AbstractArray{T}, idata, ::Val{SHMEM}, ::Val{TDIM}, ::Val{BLOCK_ROWS}, ::Val{NROW}
    ) where {T, SHMEM, TDIM, BLOCK_ROWS, NROW}

    ui1 = Cuint(1)
    tile = @LocalMemory(state, T, SHMEM)
    bidx_x = blockidx_x(state) - ui1
    bidx_y = blockidx_y(state) - ui1
    tidx_x = threadidx_x(state) - ui1
    tidx_y = threadidx_y(state) - ui1

    x = bidx_x * TDIM + tidx_x + ui1
    y = bidx_y * TDIM + tidx_y + ui1
    dims = size(idata)

    (x <= dims[2] && (y + (BLOCK_ROWS * Cuint(3))) <= dims[1]) || return

    for j = Cuint(0):Cuint(3)
        j0 = j * BLOCK_ROWS
        tile[tidx_x + ui1, tidx_y + j0 + ui1] = idata[y + j0, x]
    end

    synchronize_threads(state)
    for j = Cuint(0):Cuint(3)
        j0 = j * BLOCK_ROWS
        odata[x, y + j0] = tile[tidx_x + ui1, tidx_y + j0 + ui1]
    end
    return
end

function transpose!{T}(At::GPUArray{T, 2}, A::GPUArray{T, 2})
    if size(A, 1) == size(A, 2) && all(x-> x % 32 == 0, size(A))
        outsize = UInt32.(size(At))
        TDIM = Cuint(32); BLOCK_ROWS = Cuint(8)
        nrows = TDIM ÷ BLOCK_ROWS
        shmemdim = (TDIM, (TDIM + Cuint(1)))
        static_params = map(x-> Val{x}(), (shmemdim, TDIM, BLOCK_ROWS, nrows))
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
    ntuple(d-> I[perm[d]], Val{N})
end

function permutedims!(dest::GPUArray, src::GPUArray, perm)
    perm = Cuint.((perm...,))
    gpu_call(dest, (dest, src, perm)) do state, dest, src, perm
        I = @cartesianidx dest state
        @inbounds dest[I...] = src[genperm(I, perm)...]
        return
    end
    return dest
end
