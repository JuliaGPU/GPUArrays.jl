# common Base functionality

if VERSION ≥ v"1.6"
    import Base: _RepeatInnerOuter

    function repeat_inner_kernel!(
        ctx::AbstractKernelContext,
        xs::AbstractArray{<:Any, N},
        inner::NTuple{N, Int},
        out::AbstractArray{<:Any, N}
    ) where {N}
        dest_inds = @cartesianidx(out).I
        # Get the "stride" index in each dimension, where the size
        # of the stride is given by `inner`. The stride-index then
        # corresponds to the index of the repeated value in `xs`.
        src_inds = ntuple(i -> (dest_inds[i] - 1) ÷ inner[i] + 1, N)

        @inbounds out[dest_inds...] = xs[src_inds...]

        return nothing
    end

    function repeat_inner(xs::AnyGPUArray, inner)
        out = similar(xs, eltype(xs), inner .* size(xs))
        gpu_call(repeat_inner_kernel!, xs, inner, out; total_threads=prod(size(out)))
        return out
    end

    function repeat_outer_kernel!(
        ctx::AbstractKernelContext,
        xs::AbstractArray{<:Any, N},
        xssize::NTuple{N},
        outer::NTuple{N},
        out::AbstractArray{<:Any, N}
    ) where {N}
        dest_inds = @cartesianidx(out).I
        # Outer is just wrapping around the edges of `xs`.
        src_inds = ntuple(i -> (dest_inds[i] - 1) % xssize[i] + 1, N)

        @inbounds out[dest_inds...] = xs[src_inds...]

        return nothing
    end

    function repeat_outer(xs::AnyGPUArray, outer)
        out = similar(xs, eltype(xs), outer .* size(xs))
        gpu_call(repeat_outer_kernel!, xs, size(xs), outer, out; total_threads=prod(size(out)))
        return out
    end

    # Overload methods used by `Base.repeat`.
    # No need to implement `repeat_inner_outer` since this is implemented in `Base` as
    # `repeat_outer(repeat_inner(arr, inner), outer)`.
    function _RepeatInnerOuter.repeat_inner(xs::AnyGPUArray{<:Any, N}, dims::NTuple{N}) where {N}
        return repeat_inner(xs, dims)
    end

    function _RepeatInnerOuter.repeat_outer(xs::AnyGPUArray{<:Any, N}, dims::NTuple{N}) where {N}
        return repeat_outer(xs, dims)
    end

    function _RepeatInnerOuter.repeat_outer(xs::AnyGPUArray{<:Any, 1}, dims::Tuple{Any})
        return repeat_outer(xs, dims)
    end

    function _RepeatInnerOuter.repeat_outer(xs::AnyGPUArray{<:Any, 2}, dims::NTuple{2, Any})
        return repeat_outer(xs, dims)
    end
else
    function Base.repeat(a::AbstractGPUVecOrMat, m::Int, n::Int = 1)
        o, p = size(a, 1), size(a, 2)
        b = similar(a, o*m, p*n)
        if length(b) == 0
            return b
        end
        gpu_call(b, a, o, p, m, n; total_threads=n) do ctx, b, a, o, p, m, n
            j = linear_index(ctx)
            j > n && return
            d = (j - 1) * p + 1
            @inbounds for i in 1:m
                c = (i - 1) * o + 1
                for r in 1:p
                    for k in 1:o
                        b[k - 1 + c, r - 1 + d] = a[k, r]
                    end
                end
            end
            return
        end
        return b
    end

    function Base.repeat(a::AbstractGPUVector, m::Int)
        o = length(a)
        b = similar(a, o*m)
        if length(b) == 0
            return b
        end
        gpu_call(b, a, o, m; total_threads=m) do ctx, b, a, o, m
            i = linear_index(ctx)
            i > m && return
            c = (i - 1)*o + 1
            @inbounds for i in 1:o
                b[c + i - 1] = a[i]
            end
            return
        end
        return b
    end
end
## PermutedDimsArrays

using Base: PermutedDimsArrays

# PermutedDimsArrays' custom copyto! doesn't know how to deal with GPU arrays
function PermutedDimsArrays._copy!(dest::PermutedDimsArray{T,N,<:Any,<:Any,<:AbstractGPUArray}, src) where {T,N}
    dest .= src
    dest
end
