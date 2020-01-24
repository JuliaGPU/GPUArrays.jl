# core definition of the AbstractGPUArray type

export AbstractGPUArray

"""
    AbstractGPUArray{T, N} <: DenseArray{T, N}

Supertype for `N`-dimensional GPU arrays (or array-like types) with elements of type `T`.
Instances of this type are expected to live on the host, see [`AbstractDeviceArray`](@ref)
for device-side objects.
"""
abstract type AbstractGPUArray{T, N} <: DenseArray{T, N} end

const AbstractGPUVector{T} = AbstractGPUArray{T, 1}
const AbstractGPUMatrix{T} = AbstractGPUArray{T, 2}
const AbstractGPUVecOrMat{T} = Union{AbstractGPUArray{T, 1}, AbstractGPUArray{T, 2}}


# input/output

## serialization

import Serialization: AbstractSerializer, serialize, deserialize, serialize_type

function serialize(s::AbstractSerializer, t::T) where T <: AbstractGPUArray
    serialize_type(s, T)
    serialize(s, Array(t))
end
function deserialize(s::AbstractSerializer, ::Type{T}) where T <: AbstractGPUArray
    A = deserialize(s)
    T(A)
end

function to_cartesian(A, indices::Tuple)
    start = CartesianIndex(ntuple(length(indices)) do i
        val = indices[i]
        isa(val, Integer) && return val
        isa(val, UnitRange) && return first(val)
        isa(val, Colon) && return 1
        isa(val, Base.Slice{Base.OneTo{Int}}) && return 1
        error("GPU indexing only defined for integers or unit ranges. Found: $val")
    end)
    stop = CartesianIndex(ntuple(length(indices)) do i
        val = indices[i]
        isa(val, Integer) && return val
        isa(val, UnitRange) && return last(val)
        isa(val, Colon) && return size(A, i)
        isa(val, Base.Slice{Base.OneTo{Int}}) && return size(A, i)
        error("GPU indexing only defined for integers or unit ranges. Found: $val")
    end)
    CartesianIndices(start, stop)
end

## convert to CPU (keeping wrapper type)

Adapt.adapt_storage(::Type{<:Array}, xs::AbstractArray) = convert(Array, xs)
convert_to_cpu(xs) = adapt(Array, xs)

## showing

for (W, ctor) in (:AT => (A,mut)->mut(A), Adapt.wrappers...)
    @eval begin
        # display
        Base.print_array(io::IO, X::$W where {AT <: AbstractGPUArray}) =
            Base.print_array(io, $ctor(X, convert_to_cpu))

        # show
        Base._show_nonempty(io::IO, X::$W where {AT <: AbstractGPUArray}, prefix::String) =
            Base._show_nonempty(io, $ctor(X, convert_to_cpu), prefix)
        Base._show_empty(io::IO, X::$W where {AT <: AbstractGPUArray}) =
            Base._show_empty(io, $ctor(X, convert_to_cpu))
        Base.show_vector(io::IO, v::$W where {AT <: AbstractGPUArray}, args...) =
            Base.show_vector(io, $ctor(v, convert_to_cpu), args...)
    end
end

## collect to CPU (discarding wrapper type)

collect_to_cpu(xs::AbstractArray) = collect(convert_to_cpu(xs))

for (W, ctor) in (:AT => (A,mut)->mut(A), Adapt.wrappers...)
    @eval begin
        Base.collect(X::$W where {AT <: AbstractGPUArray}) = collect_to_cpu(X)
    end
end


# memory copying

## basic linear copies of identically-typed memory

# convert to something we can get a pointer to
materialize(x::AbstractArray) = Array(x)
materialize(x::AbstractGPUArray) = x
materialize(x::Array) = x

# TODO: do we want to support `copyto(..., WrappedArray{AbstractGPUArray})`
# if so (does not work due to lack of copy constructors):
#for (W, ctor) in (:AT => (A,mut)->mut(A), Adapt.wrappers...)
#    @eval begin
#        materialize(X::$W) where {AT <: AbstractGPUArray} = AT(X)
#    end
#end

for (D, S) in ((AbstractGPUArray, AbstractArray), (Array, AbstractGPUArray), (AbstractGPUArray, AbstractGPUArray))
    @eval begin
        function Base.copyto!(dest::$D{T, N}, rdest::NTuple{N, UnitRange},
                              src::$S{T, N}, ssrc::NTuple{N, UnitRange}) where {T, N}
            drange = CartesianIndices(rdest)
            srange = CartesianIndices(ssrc)
            copyto!(dest, drange, src, srange)
        end

        function Base.copyto!(dest::$D{T}, d_range::CartesianIndices{1},
                              src::$S{T}, s_range::CartesianIndices{1}) where T
            len = length(d_range)
            if length(s_range) != len
                throw(ArgumentError("Copy range needs same length. Found: dest: $len, src: $(length(s_range))"))
            end
            len == 0 && return dest
            d_offset = first(d_range)[1]
            s_offset = first(s_range)[1]
            copyto!(dest, d_offset, materialize(src), s_offset, len)
        end

        function Base.copyto!(dest::$D{T, N}, src::$S{T, N}) where {T, N}
            len = length(src)
            len == 0 && return dest
            copyto!(dest, 1, materialize(src), 1, len)
        end
    end
end

## generalized blocks of heterogeneous memory

Base.copyto!(dest::AbstractGPUArray, src::AbstractGPUArray) =
    copyto!(dest, CartesianIndices(dest), src, CartesianIndices(src))

function copy_kernel!(state, dest, dest_offsets, src, src_offsets, shape, shape_dest, shape_source, length)
    i = linear_index(state)
    if i <= length
        # TODO can this be done faster and smarter?
        idx = gpu_ind2sub(shape, i)
        dest_idx = gpu_sub2ind(shape_dest, idx .+ dest_offsets)
        src_idx = gpu_sub2ind(shape_source, idx .+ src_offsets)
        @inbounds dest[dest_idx] = src[src_idx]
    end
    return
end

function Base.copyto!(dest::AbstractGPUArray{T, N}, destcrange::CartesianIndices{N},
                      src::AbstractGPUArray{U, N}, srccrange::CartesianIndices{N}) where {T, U, N}
    shape = size(destcrange)
    if shape != size(srccrange)
        throw(DimensionMismatch("Ranges don't match their size. Found: $shape, $(size(srccrange))"))
    end
    len = length(destcrange)

    dest_offsets = first.(destcrange.indices) .- 1
    src_offsets = first.(srccrange.indices) .- 1
    gpu_call(copy_kernel!, dest,
             (dest, dest_offsets, src, src_offsets, shape, size(dest), size(src), len),
             len)
    dest
end

function Base.copyto!(dest::AbstractGPUArray{T, N}, destcrange::CartesianIndices{N},
                      src::AbstractArray{T, N}, srccrange::CartesianIndices{N}) where {T, N}
    # Is this efficient? Maybe!
    # TODO: compare to a pure intrinsic copyto implementation!
    # this would mean looping over linear sections of memory and
    # use copyto!(dest, offset::Integer, buffer(src), offset::Integer, amout::Integer)
    src_gpu = typeof(dest)(map(idx-> src[idx], srccrange))
    nrange = CartesianIndices(size(src_gpu))
    copyto!(dest, destcrange, src_gpu, nrange)
    dest
end

function Base.copyto!(dest::AbstractArray{T, N}, destcrange::CartesianIndices{N},
                      src::AbstractGPUArray{T, N}, srccrange::CartesianIndices{N}) where {T, N}
    # Is this efficient? Maybe!
    dest_gpu = similar(src, size(destcrange))
    nrange = CartesianIndices(size(dest_gpu))
    copyto!(dest_gpu, nrange, src, srccrange)
    copyto!(dest, destcrange, Array(dest_gpu), nrange)
    dest
end

## other

Base.copy(x::AbstractGPUArray) = identity.(x)

Base.deepcopy(x::AbstractGPUArray) = copy(x)


# reinterpret

#=
copied from julia base/array.jl
Copyright (c) 2009-2016: Jeff Bezanson, Stefan Karpinski, Viral B. Shah, and other contributors:

https://github.com/JuliaLang/julia/contributors

Permission is hereby granted, free of charge, to any person obtaining a copie of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
=#

import Base.reinterpret

"""
    unsafe_reinterpret(T, a, dims)

Reinterpret the array `a` to have a new element type `T` and size `dims`.
"""
function unsafe_reinterpret end

function reinterpret(::Type{T}, a::AbstractGPUArray{S,1}) where T where S
    nel = (length(a)*sizeof(S)) ÷ sizeof(T)
    # TODO: maybe check that remainder is zero?
    return reinterpret(T, a, (nel,))
end

function reinterpret(::Type{T}, a::AbstractGPUArray{S}) where T where S
    if sizeof(S) != sizeof(T)
        throw(ArgumentError("result shape not specified"))
    end
    reinterpret(T, a, size(a))
end

function reinterpret(::Type{T}, a::AbstractGPUArray{S}, dims::NTuple{N, Integer}) where T where S where N
    if !isbitstype(T)
        throw(ArgumentError("cannot reinterpret Array{$(S)} to ::Type{Array{$(T)}}, type $(T) is not a bits type"))
    end
    if !isbitstype(S)
        throw(ArgumentError("cannot reinterpret Array{$(S)} to ::Type{Array{$(T)}}, type $(S) is not a bits type"))
    end
    nel = div(length(a)*sizeof(S),sizeof(T))
    if prod(dims) != nel
        throw(DimensionMismatch("new dimensions $(dims) must be consistent with array size $(nel)"))
    end
    unsafe_reinterpret(T, a, dims)
end

function Base._reshape(A::AbstractGPUArray{T}, dims::Dims) where T
    n = length(A)
    prod(dims) == n || throw(DimensionMismatch("parent has $n elements, which is incompatible with size $dims"))
    return unsafe_reinterpret(T, A, dims)
end
#ambig
function Base._reshape(A::AbstractGPUArray{T, 1}, dims::Tuple{Integer}) where T
    n = Base._length(A)
    prod(dims) == n || throw(DimensionMismatch("parent has $n elements, which is incompatible with size $dims"))
    return unsafe_reinterpret(T, A, dims)
end


# filtering

# TODO: filter!

# revert of JuliaLang/julia#31929
Base.filter(f, As::AbstractGPUArray) = As[map(f, As)::AbstractGPUArray{Bool}]
