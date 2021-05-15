# common Base functionality

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
    any(==(0), size(out)) && return out # consistent with `Base.repeat`

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
    any(==(0), size(out)) && return out # consistent with `Base.repeat`

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

## PermutedDimsArrays

using Base: PermutedDimsArrays

# PermutedDimsArrays' custom copyto! doesn't know how to deal with GPU arrays
function PermutedDimsArrays._copy!(dest::PermutedDimsArray{T,N,<:Any,<:Any,<:AbstractGPUArray}, src) where {T,N}
    dest .= src
    dest
end

## concatenation

# hacky overloads to make simple vcat and hcat with numbers work as expected.
# we can't really make this work in general without Base providing
# a dispatch mechanism for output container type.
@inline Base.vcat(a::Number, b::AbstractGPUArray) =
    vcat(fill!(similar(b, typeof(a), (1,size(b)[2:end]...)), a), b)
@inline Base.hcat(a::Number, b::AbstractGPUArray) =
    hcat(fill!(similar(b, typeof(a), (size(b)[1:end-1]...,1)), a), b)
