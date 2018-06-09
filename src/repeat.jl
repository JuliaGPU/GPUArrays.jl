using Base: CartesianIndex, tail, cat_fill!

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

function repeat_kernel(state, A::AbstractArray{T}, out::AbstractArray{T},  inner, outer, Asize, outSize, inner_shape) where T
    ilin = linear_index(state)
    idx = GPUArrays.gpu_ind2sub(outSize, ilin)
    if (idx[1] > Asize[1] || idx[2] > Asize[2])
        return
    end
    # inner_indices = (1:n for n in inner)
    inner_indices = ntuple_args(Val{length(inner)}(), idx, inner) do i, idx, inner
        (1:inner[i]) + (idx[i] - 1) * inner[i]
    end

    for i in inner_indices
        out[i] = A[idx[1], idx[2]]
    end

    src_indices = ntuple_args(Val{length(inner_shape)}(), inner_shape) do i, inner_shape
        1:inner_shape[i]
    end

    dest_indices = ntuple_args(Val{length(inner_shape)}(), inner_shape) do i, inner_shape
        1:inner_shape[i]
    end

    for i in 1:length(outer)
        # for j in 2:outer[i]
            dest_indices = ntuple_args(Val{length(inner_shape)}(), inner_shape, out, dest_indices) do k, inner_shape, out, dest_indices
                if k == i
                    dest_indices[i] + inner_shape[i]
                else
                    dest_indices[i]
                end
            # end
            out[dest_indices...] = out[src_indices...]
        end
        src_indices = ntuple_args(Val{length(outSize)}(), outSize) do i, outSize
            1:outSize[i]
        end
        dest_indices = ntuple_args(Val{length(outSize)}(), outSize) do i, outSize
            1:outSize[i]
        end
    end






    
    # n = inner[1]
    # inner_indices[1] = (1:n) + ((c[1] - 1) * n)

   # fill the first inner block
    # if all(x -> x == 1, inner)
    #     out[indices(A)...] = A
    # else
    #     inner_indices = [1:n for n in inner]
    #     for c in CartesianRange(indices(A))
    #         for i in 1:ndims(A)
    #             n = inner[i]
    #             inner_indices[i] = (1:n) + ((c[i] - 1) * n)
    #         end
    #         cat_fill!(out, A[c], inner_indices)
    #     end
    # end

    # fill the outer blocks along each dimension
    # if all(x -> x == 1, outer)
    #     return R
    # end
    # src_indices  = [1:n for n in inner_shape]
    # dest_indices = copy(src_indices)
    # for i in 1:length(outer)
    #     B = view(out, src_indices...)
    #     for j in 2:outer[i]
    #         dest_indices[i] += inner_shape[i]
    #         out[dest_indices...] = B
    #     end
    #     src_indices[i] = dest_indices[i] = 1:shape[i]
    # end

end


function gpu_repeat(A::GPUArray, inner, outer) 
    shape, inner_shape = rep_shapes(A, inner, outer)
    R = similar(A, shape)
    if any(iszero, shape)
        return R
    end
    gpu_call(repeat_kernel, R, (A, R, inner, outer, UInt32.(size(A)), UInt32.(shape), UInt32.(inner_shape)))
    return R
end