# common Base functionality
import Base: _RepeatInnerOuter

# Handle `out = repeat(x; inner)` by parallelizing over `out` array This can benchmark
# faster if repeating elements along the first axis (i.e. `inner=(n, ones...)`), as data
# access can be contiguous on write.
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

# Handle `out = repeat(x; inner)` by parallelizing over the `xs` array This tends to
# benchmark faster by having fewer read operations and avoiding the costly division
# operation. Additionally, when repeating over the trailing dimension. `inner=(ones..., n)`,
# data access can be contiguous during both the read and write operations.
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

    # Pick which kernel to launch based on `inner`, using the heuristic that if the largest
    # entry in `inner` is `inner[1]`, then we should parallelize over `out`. Otherwise, we
    # should parallelize over `xs`. This choice is purely for performance. Better heuristics
    # may exist, but hopefully, this is good enough.
    #
    # Using `repeat_inner_src_kernel!`, requires fewer read ops (`prod(size(xs))` vs.
    # `prod(size(out))`) and generally benchmarks faster than `repeat_inner_dst_kernel!`.
    # However, for `inner = (n, 1, 1)`, `repeat_inner_dst_kernel!` benchmarked faster as it
    # avoids strided memory access over `out`.
    # See https://github.com/JuliaGPU/GPUArrays.jl/pull/400#issuecomment-1172641982 for the
    # relevant benchmarks.
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


## reshape

function Base.reshape(a::AbstractGPUArray{T,M}, dims::NTuple{N,Int}) where {T,N,M}
  if prod(dims) != length(a)
      throw(DimensionMismatch("new dimensions $(dims) must be consistent with array size $(size(a))"))
  end

  if N == M && dims == size(a)
      return a
  end

  derive(T, N, a, dims, 0)
end


## reinterpret

function Base.reinterpret(::Type{T}, a::AbstractGPUArray{S,N}) where {T,S,N}
  err = _reinterpret_exception(T, a)
  err === nothing || throw(err)

  if sizeof(T) == sizeof(S) # for N == 0
    osize = size(a)
  else
    isize = size(a)
    size1 = div(isize[1]*sizeof(S), sizeof(T))
    osize = tuple(size1, Base.tail(isize)...)
  end

  return derive(T, N, a, osize, 0)
end

function _reinterpret_exception(::Type{T}, a::AbstractArray{S,N}) where {T,S,N}
  if !isbitstype(T) || !isbitstype(S)
    return ReinterpretBitsTypeError{T,typeof(a)}()
  end
  if N == 0 && sizeof(T) != sizeof(S)
    return ReinterpretZeroDimError{T,typeof(a)}()
  end
  if N != 0 && sizeof(S) != sizeof(T)
      ax1 = axes(a)[1]
      dim = length(ax1)
      if Base.rem(dim*sizeof(S),sizeof(T)) != 0
        return ReinterpretDivisibilityError{T,typeof(a)}(dim)
      end
      if first(ax1) != 1
        return ReinterpretFirstIndexError{T,typeof(a),typeof(ax1)}(ax1)
      end
  end
  return nothing
end

struct ReinterpretBitsTypeError{T,A} <: Exception end
function Base.showerror(io::IO, ::ReinterpretBitsTypeError{T, <:AbstractArray{S}}) where {T, S}
  print(io, "cannot reinterpret an `$(S)` array to `$(T)`, because not all types are bitstypes")
end

struct ReinterpretZeroDimError{T,A} <: Exception end
function Base.showerror(io::IO, ::ReinterpretZeroDimError{T, <:AbstractArray{S,N}}) where {T, S, N}
  print(io, "cannot reinterpret a zero-dimensional `$(S)` array to `$(T)` which is of a different size")
end

struct ReinterpretDivisibilityError{T,A} <: Exception
  dim::Int
end
function Base.showerror(io::IO, err::ReinterpretDivisibilityError{T, <:AbstractArray{S,N}}) where {T, S, N}
  dim = err.dim
  print(io, """
      cannot reinterpret an `$(S)` array to `$(T)` whose first dimension has size `$(dim)`.
      The resulting array would have non-integral first dimension.
      """)
end

struct ReinterpretFirstIndexError{T,A,Ax1} <: Exception
  ax1::Ax1
end
function Base.showerror(io::IO, err::ReinterpretFirstIndexError{T, <:AbstractArray{S,N}}) where {T, S, N}
  ax1 = err.ax1
  print(io, "cannot reinterpret a `$(S)` array to `$(T)` when the first axis is $ax1. Try reshaping first.")
end


## reinterpret(reshape)

function Base.reinterpret(::typeof(reshape), ::Type{T}, a::AbstractGPUArray) where {T}
  N, osize = _base_check_reshape_reinterpret(T, a)
  return derive(T, N, a, osize, 0)
end

# taken from reinterpretarray.jl
# TODO: move these Base definitions out of the ReinterpretArray struct for reuse
function _base_check_reshape_reinterpret(::Type{T}, a::AbstractGPUArray{S}) where {T,S}
  isbitstype(T) || throwbits(S, T, T)
  isbitstype(S) || throwbits(S, T, S)
  if sizeof(S) == sizeof(T)
      N = ndims(a)
      osize = size(a)
  elseif sizeof(S) > sizeof(T)
      d, r = divrem(sizeof(S), sizeof(T))
      r == 0 || throwintmult(S, T)
      N = ndims(a) + 1
      osize = (d, size(a)...)
  else
      d, r = divrem(sizeof(T), sizeof(S))
      r == 0 || throwintmult(S, T)
      N = ndims(a) - 1
      N > -1 || throwsize0(S, T, "larger")
      axes(a, 1) == Base.OneTo(sizeof(T) ÷ sizeof(S)) || throwsize1(a, T)
      osize = size(a)[2:end]
  end
  return N, osize
end

@noinline function throwbits(S::Type, T::Type, U::Type)
  throw(ArgumentError("cannot reinterpret `$(S)` as `$(T)`, type `$(U)` is not a bits type"))
end

@noinline function throwintmult(S::Type, T::Type)
  throw(ArgumentError("`reinterpret(reshape, T, a)` requires that one of `sizeof(T)` (got $(sizeof(T))) and `sizeof(eltype(a))` (got $(sizeof(S))) be an integer multiple of the other"))
end

@noinline function throwsize0(S::Type, T::Type, msg)
  throw(ArgumentError("cannot reinterpret a zero-dimensional `$(S)` array to `$(T)` which is of a $msg size"))
end

@noinline function throwsize1(a::AbstractArray, T::Type)
    throw(ArgumentError("`reinterpret(reshape, $T, a)` where `eltype(a)` is $(eltype(a)) requires that `axes(a, 1)` (got $(axes(a, 1))) be equal to 1:$(sizeof(T) ÷ sizeof(eltype(a))) (from the ratio of element sizes)"))
end


## views

struct Contiguous end
struct NonContiguous end

# NOTE: this covers more cases than the I<:... in Base.FastContiguousSubArray
GPUIndexStyle() = Contiguous()
GPUIndexStyle(I...) = NonContiguous()
GPUIndexStyle(::Union{Base.ScalarIndex, CartesianIndex}...) = Contiguous()
GPUIndexStyle(i1::Colon, ::Union{Base.ScalarIndex, CartesianIndex}...) = Contiguous()
GPUIndexStyle(i1::AbstractUnitRange, ::Union{Base.ScalarIndex, CartesianIndex}...) = Contiguous()
GPUIndexStyle(i1::Colon, I...) = GPUIndexStyle(I...)

viewlength() = ()
@inline viewlength(::Real, I...) = viewlength(I...) # skip scalar

if VERSION >= v"1.8.0-DEV.120"
@inline viewlength(i1::AbstractUnitRange, I...) = (Base.length(i1), viewlength(I...)...)
@inline viewlength(i1::AbstractUnitRange, ::Base.ScalarIndex...) = (Base.length(i1),)
else
@inline viewlength(i1::AbstractUnitRange, I...) = (length(i1), viewlength(I...)...)
@inline viewlength(i1::AbstractUnitRange, ::Base.ScalarIndex...) = (length(i1),)
end

# adaptor to upload an array to the GPU
struct ToGPU
    array::AbstractGPUArray
end
function Adapt.adapt_storage(to::ToGPU, xs::Array)
    arr = similar(to.array, eltype(xs), size(xs))
    copyto!(arr, xs)
    arr
end

@inline function Base.view(A::AbstractGPUArray, I::Vararg{Any,N}) where {N}
    J = to_indices(A, I)
    J_gpu = map(j->adapt(ToGPU(A), j), J)
    @boundscheck checkbounds(A, J...)
    unsafe_view(A, J_gpu, GPUIndexStyle(I...))
end

@inline function unsafe_view(A, I, ::Contiguous)
    unsafe_contiguous_view(Base._maybe_reshape_parent(A, Base.index_ndims(I...)), I, viewlength(I...))
end
@inline function unsafe_contiguous_view(a::AbstractGPUArray{T}, I::NTuple{N,Base.ViewIndex}, dims::NTuple{M,Integer}) where {T,N,M}
    offset = Base.compute_offset1(a, 1, I)

    derive(T, M, a, dims, offset)
end

@inline function unsafe_view(A, I, ::NonContiguous)
    Base.unsafe_view(Base._maybe_reshape_parent(A, Base.index_ndims(I...)), I...)
end
