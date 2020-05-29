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

device(::AbstractGPUDevice) = error("Not implemented") # COV_EXCL_LINE
backend(::Type{<:AbstractGPUDevice}) = error("Not implemented") # COV_EXCL_LINE


# convenience aliases for working with wrapped arrays

# NOTE: these Unions are a hack. Ideally Base would have a Transpose <: WrappedArray <:
# AbstractArray and we could define our methods in terms of Union{AbstractGPUArray,
# WrappedArray{<:Any, <:AbstractGPUArray}}

const WrappedArray{AT} = @eval Union{$([W for (W,ctor) in Adapt.wrappers]...)} where AT

const WrappedGPUArray{T} = WrappedArray{<:AbstractGPUArray{T}}

const AbstractOrWrappedGPUArray{T} = Union{AbstractGPUArray{T}, WrappedGPUArray{T}}


# input/output

## serialization

using Serialization: AbstractSerializer, serialize_type

function Serialization.serialize(s::AbstractSerializer, t::T) where T <: AbstractGPUArray
    serialize_type(s, T)
    serialize(s, Array(t))
end
function Serialization.deserialize(s::AbstractSerializer, ::Type{T}) where T <: AbstractGPUArray
    A = deserialize(s)
    T(A)
end

## convert to CPU (keeping wrapper type)

Adapt.adapt_storage(::Type{<:Array}, xs::AbstractArray) = convert(Array, xs)
convert_to_cpu(xs) = adapt(Array, xs)

## showing

# display
Base.print_array(io::IO, X::AbstractOrWrappedGPUArray) =
    Base.print_array(io, adapt(Array, X))

# show
Base._show_nonempty(io::IO, X::AbstractOrWrappedGPUArray, prefix::String) =
    Base._show_nonempty(io, convert_to_cpu(X), prefix)
Base._show_empty(io::IO, X::AbstractOrWrappedGPUArray) =
    Base._show_empty(io, convert_to_cpu(X))
Base.show_vector(io::IO, v::AbstractOrWrappedGPUArray, args...) =
    Base.show_vector(io, convert_to_cpu(v), args...)

## collect to CPU (discarding wrapper type)

collect_to_cpu(xs::AbstractArray) = collect(convert_to_cpu(xs))
Base.collect(X::AbstractOrWrappedGPUArray) = collect_to_cpu(X)


# memory copying

## basic linear copies of identically-typed memory

# convert to something we can get a pointer to
materialize(x::AbstractArray) = Array(x)
materialize(x::AbstractGPUArray) = x
materialize(x::Array) = x

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

function copy_kernel!(ctx::AbstractKernelContext, dest, dest_offsets, src, src_offsets, shape, length)
    i = linear_index(ctx)
    if i <= length
        # TODO can this be done faster and smarter?
        idx = CartesianIndices(shape)[i]
        @inbounds dest[idx + dest_offsets] = src[idx + src_offsets]
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
    len == 0 && return dest

    dest_offsets = first(destcrange) - oneunit(CartesianIndex{N})
    src_offsets = first(srccrange) - oneunit(CartesianIndex{N})
    gpu_call(copy_kernel!,
             dest, dest_offsets, src, src_offsets, shape, len;
             total_threads=len)
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

# FIXME: this is slow, and shouldn't require broadcast in the common cast
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
