using Base: CartesianIndex, tail, cat_fill!
using CUDAnative

@generated function ntuple_args(f, ::Val{N}, args::Vararg{<: Any, Nargs}) where {N, Nargs}
    expr = Expr(:tuple)
    for i = 1:N
        call = Expr(:call, :f, i)
        for j = 1:Nargs
            push!(call.args, :(args[$j]))
        end
        push!(expr.args, call)
    end
    quote
        Base.@_inline_meta
        $expr
    end
end

rep_shapes(A, i, o) = _rshps((), (), size(A), i, o)

_rshps(shp, shp_i, ::Tuple{}, ::Tuple{}, ::Tuple{}) = (shp, shp_i)
@inline _rshps(shp, shp_i, ::Tuple{}, ::Tuple{}, o) =
    _rshps((shp..., o[1]), (shp_i..., 1), (), (), tail(o))
@inline _rshps(shp, shp_i, ::Tuple{}, i, ::Tuple{}) = (n = i[1];
    _rshps((shp..., n), (shp_i..., n), (), tail(i), ()))
@inline _rshps(shp, shp_i, ::Tuple{}, i, o) = (n = i[1];
    _rshps((shp..., n * o[1]), (shp_i..., n), (), tail(i), tail(o)))
@inline _rshps(shp, shp_i, sz, i, o) = (n = sz[1] * i[1];
    _rshps((shp..., n * o[1]), (shp_i..., n), tail(sz), tail(i), tail(o)))
_rshps(shp, shp_i, sz, ::Tuple{}, ::Tuple{}) =
    (n = length(shp); N = n + length(sz); _reperr("inner", n, N))
_rshps(shp, shp_i, sz, ::Tuple{}, o) =
    (n = length(shp); N = n + length(sz); _reperr("inner", n, N))
_rshps(shp, shp_i, sz, i, ::Tuple{}) =
    (n = length(shp); N = n + length(sz); _reperr("outer", n, N))
_reperr(s, n, N) = throw(ArgumentError("number of " * s * " repetitions " *
    "($n) cannot be less than number of dimensions of input ($N)"))

function repeat_kernel(state, A::AbstractArray{T}, out::AbstractArray{T}, inner, outer, Asize, outSize, inner_shape) where T
    ilin = linear_index(state)
    idx = GPUArrays.gpu_ind2sub(outSize, ilin)
    if (idx[1] > Asize[1] || idx[2] > Asize[2])
        return
    end

    # save_print("idx ", idx)

    for n1 in UInt32(1):UInt32(outer[UInt32(1)])
        for n2 in UInt32(1):UInt32(outer[UInt32(2)])
            inner_start_indices = ntuple_args(Val{length(inner)}(), inner, idx, n1, n2, inner_shape) do i, inner, idx, n1, n2, inner_shape
                if UInt32(i) == UInt32(1)
                    @inbounds return ((UInt32(1)) + (idx[i] - UInt32(1)) * inner[i]) + ((n1 - UInt32(1)) * inner_shape[UInt32(1)])
                elseif UInt32(i) == UInt32(2)
                    @inbounds return ((UInt32(1)) + (idx[i] - UInt32(1)) * inner[i]) + ((n2 - UInt32(1)) * inner_shape[UInt32(2)])
                else
                    @inbounds return ((UInt32(1)) + (idx[i] - UInt32(1)) * inner[i])
                end
            end

            inner_end_indices = ntuple_args(Val{length(inner)}(), inner, idx, n1, n2, inner_shape) do i, inner, idx, n1, n2, inner_shape
                if UInt32(i) == UInt32(1)
                    @inbounds return ((inner[i]) + (idx[i] - UInt32(1)) * inner[i]) + ((n1 - UInt32(1)) * inner_shape[UInt32(1)])
                elseif UInt32(i) == UInt32(2)
                    @inbounds return ((inner[i]) + (idx[i] - UInt32(1)) * inner[i]) + ((n2 - UInt32(1)) * inner_shape[UInt32(2)])    
                else
                    @inbounds return ((inner[i]) + (idx[i] - UInt32(1)) * inner[i])
                end
            end

            for i in inner_start_indices[1]:inner_end_indices[1]
                for j in inner_start_indices[2]:inner_end_indices[2]
                    @inbounds out[i, j] = A[idx[1], idx[2]]
                end
            end
        end
    end

    synchronize_threads(state)

    return
end


function repeat_back_kernel(state, A::AbstractArray{T}, delta::AbstractArray{T}, out::AbstractArray{T}, inner, outer, ::Val{dims}, Asize) where {T, dims}
    ilin = linear_index(state)
    idx = GPUArrays.gpu_ind2sub(Asize, ilin)
    if (idx[1] > Asize[1] || idx[2] > Asize[2])
        return
    end
    
    @inbounds src_idx_1 = mod1(div(idx[1] - 1, inner[1]) + 1, Asize[1])
    @inbounds src_idx_2 = mod1(div(idx[2] - 1, inner[2]) + 1, Asize[2])
    synchronize_threads(state)
    @inbounds const_here = out[src_idx_1, src_idx_2]
    # synchronize_threads(state)

    # CUDAnative.@cuprintf("idx[1]: %d\nidx[2]: %d\nsrc_idx_1: %d\nsrc_idx_2: %d\nconst_here: %d\ndelta[idx...]: %f\n\n", idx[1], idx[2], src_idx_1, src_idx_2, const_here, delta[idx...])
    
    @inbounds out[src_idx_1, src_idx_2] = const_here + delta[idx...]
    synchronize_threads(state)
    
    return
end

function gpu_repeat(A::GPUArray, inner, outer) 
    shape, inner_shape = rep_shapes(A, inner, outer)
    R = similar(A, shape)
    if any(iszero, shape)
        return R
    end
    gpu_call(repeat_kernel, delta, (A, R, inner, outer, UInt32.(size(A)), UInt32.(shape), UInt32.(inner_shape)))
    return R
end

function gpu_repeat_grad(A::GPUArray, delta::GPUArray, inner, outer)
    R = zeros(typeof(A), size(A))
    gpu_call(repeat_back_kernel, A, (A, delta, R, inner, outer, Val{UInt32.(ndims(A))}(), UInt32.(size(A))))
    return R
end