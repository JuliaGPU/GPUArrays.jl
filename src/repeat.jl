using Base: CartesianIndex, tail, cat_fill!

const io_lock = ReentrantLock()
save_print(args...) = save_print(STDOUT, args...)
function save_print(io::IO, args...)
    @async begin
        try
            lock(io)
            lock(io_lock)
            print(io, string(args..., "\n"))
        finally
            unlock(io_lock)
            unlock(io)
        end
    end
end

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

    for m in UInt32(1):UInt32(length(outer))
        # inner_indices = (1:n for n in inner)
        for n in UInt32(1):UInt32(outer[m])
            inner_start_indices = ntuple_args(Val{length(inner)}(), inner, idx) do i, inner, idx
                if m == i
                    @inbounds return ((UInt32(1)) + (idx[i] - UInt32(1)) * inner[i]) + ((n - UInt32(1)) * inner_shape[m])
                else
                    @inbounds return ((UInt32(1)) + (idx[i] - UInt32(1)) * inner[i])
                end
            end

            inner_end_indices = ntuple_args(Val{length(inner)}(), inner, idx) do i, inner, idx
                if m == i
                    @inbounds return ((inner[i]) + (idx[i] - UInt32(1)) * inner[i]) + ((n - UInt32(1)) * inner_shape[m])
                else
                    @inbounds return ((inner[i]) + (idx[i] - UInt32(1)) * inner[i])
                end
            end

            # save_print("hohohohohohohohohohohohohohohohohohohohohohohohohohohoho\ninner_start_indices ", inner_start_indices, "\ninner_end_indices ", inner_end_indices, "\nidx ", idx, "\nm ", m," n ", n)

            # @inbounds out[inner_start_indices[1]:inner_end_indices[1], inner_start_indices[2]:inner_end_indices[2]] = A[idx[1], idx[2]]
            for i in inner_start_indices[1]:inner_end_indices[1]
                for j in inner_start_indices[2]:inner_end_indices[2]
                    @inbounds out[i, j] = A[idx[1], idx[2]]
                end
            end
        end
    end

    synchronize_threads(state)

    return


    src_indices_end = ntuple_args(Val{length(inner_shape)}(), inner_shape) do i, inner_shape
        inner_shape[i]
    end

    dest_indices_start = ntuple(Val{length(inner_shape)}) do i
        UInt32(1)
    end

    dest_indices_end = ntuple_args(Val{length(inner_shape)}(), inner_shape) do i, inner_shape
        inner_shape[i]
    end

    # save_print("length(outer) ", length(outer))
    # save_print("initially_dest_indices_start ", dest_indices_start)
    # save_print("initially_dest_indices_end ", dest_indices_end)
    # save_print("outer ", outer)
    # save_print("hohohohohohohohohohohohohohohohohohohohohohohohohohoho idx ", idx)
    for i in UInt32(1):UInt32(length(outer))
        for j in UInt32(2):UInt32(outer[i])
            dest_indices_start = ntuple_args(Val{length(inner_shape)}(), inner_shape, out, dest_indices_start) do k, inner_shape, out, dest_indices_start
                if k == i
                    dest_indices_start[k] + inner_shape[k]
                else
                    dest_indices_start[k]
                end
            end
            synchronize_threads(state)
            dest_indices_end = ntuple_args(Val{length(inner_shape)}(), inner_shape, out, dest_indices_end) do k, inner_shape, out, dest_indices_end
                if k == i
                    dest_indices_end[k] + inner_shape[k]
                else
                    dest_indices_end[k]
                end
            end
            # save_print("dest_indices_start ", dest_indices_start)
            # save_print("dest_indices_end ", dest_indices_end)
            m = UInt32(0)
            for k in dest_indices_start[1]:dest_indices_end[1]
                m += UInt32(1)
                n = UInt32(0)
                for l in dest_indices_start[2]:dest_indices_end[2]
                    n += UInt32(1)
                    save_print("hohohohohohohohohohohohohohohohohohohohohohohohohohoho idx ", idx, "\nhehehehehehehehehehehehehehehehehehehehehehehehehehehe k ", k, "l ", l)
                    out[k, l] = out[m, n]
                end
            end
            # out[dest_indices...] = out[src_indices...] #modify this
        end
        src_indices_end = ntuple_args(Val{length(outSize)}(), outSize) do i, outSize
            outSize[i]
        end
        dest_indices_start = ntuple_args(Val{length(outSize)}(), outSize) do i, outSize
            UInt32(1)
        end
        dest_indices_end = ntuple_args(Val{length(outSize)}(), outSize) do i, outSize
            outSize[i]
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