# common Base functionality
import Base: _RepeatInnerOuter

# Handle `out = repeat(x; inner)` by parallelizing over `out` array
function repeat_inner_dst_kernel!(
    ctx::AbstractKernelContext,
    xs::AbstractArray{<:Any, N},
    inner::NTuple{N, Int},
    out::AbstractArray{<:Any, N}
) where {N}
    # Get the "stride" index in each dimension, where the size
    # of the stride is given by `inner`. The stride-index (sdx) then
    # corresponds to the index of the repeated value in `xs`.
    odx = @cartesianidx out
    dest_inds = odx.I
    sdx = ntuple(N) do i
        @inbounds (dest_inds[i] - 1) ÷ inner[i] + 1
    end
    @inbounds out[odx] = xs[CartesianIndex(sdx)]
    return nothing
end

# Handle `out = repeat(x; inner)` by parallelizing over the `xs` array
function repeat_inner_src_kernel!(
    ctx::AbstractKernelContext,
    xs::AbstractArray{<:Any, N},
    inner::NTuple{N, Int},
    out::AbstractArray{<:Any, N}
) where {N}
    # Get single element from src
    idx = @cartesianidx xs
    @inbounds val = xs[idx]

    # Loop over "repeat" indices of inner
    for rdx in CartesianIndices(inner)
        # Get destination CartesianIndex
        odx = ntuple(N) do i
            @inbounds (idx[i]-1) * inner[i] + rdx[i]
        end
        @inbounds out[CartesianIndex(odx)] = val
    end
    return nothing
end

function repeat_inner(xs::AnyGPUArray, inner)
    out = similar(xs, eltype(xs), inner .* size(xs))
    any(==(0), size(out)) && return out # consistent with `Base.repeat`
    if argmax(inner) == firstindex(inner)
        # Parallelize over the destination array
        gpu_call(repeat_inner_dst_kernel!, xs, inner, out; elements=prod(size(out)))
    else
        # Parallelize over the source array
        gpu_call(repeat_inner_src_kernel!, xs, inner, out; elements=prod(size(xs)))
    end
    return out
end

function repeat_outer_kernel!(
    ctx::AbstractKernelContext,
    xs::AbstractArray{<:Any, N},
    xssize::NTuple{N},
    outer::NTuple{N},
    out::AbstractArray{<:Any, N}
) where {N}
    # Get index to input element
    idx = @cartesianidx xs
    @inbounds val = xs[idx]

    # Loop over repeat indices, copying val to out
    for rdx in CartesianIndices(outer)
        # Get destination CartesianIndex
        odx = ntuple(N) do i
            @inbounds idx[i] + xssize[i] * (rdx[i] -1)
        end
        @inbounds out[CartesianIndex(odx)] = val
    end

    return nothing
end

function repeat_outer(xs::AnyGPUArray, outer)
    out = similar(xs, eltype(xs), outer .* size(xs))
    any(==(0), size(out)) && return out # consistent with `Base.repeat`
    gpu_call(repeat_outer_kernel!, xs, size(xs), outer, out; elements=length(xs))
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
