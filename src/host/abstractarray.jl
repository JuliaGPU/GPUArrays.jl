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

const WrappedGPUArray{T,N} = WrappedArray{T,N,AbstractGPUArray,AbstractGPUArray{T,N}}

const AbstractOrWrappedGPUArray{T,N} =
    Union{AbstractGPUArray{T,N},
          WrappedArray{T,N,AbstractGPUArray,AbstractGPUArray{T,N}}}


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
    Base.print_array(io, convert_to_cpu(X))

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

# expects the GPU array type to have linear `copyto!` methods (i.e. accepting an integer
# offset and length) from and to CPU arrays and between GPU arrays.

for (D, S) in ((AbstractOrWrappedGPUArray, Array),
               (Array, AbstractOrWrappedGPUArray),
               (AbstractOrWrappedGPUArray, AbstractOrWrappedGPUArray))
    @eval begin
        function Base.copyto!(dest::$D{<:Any, N}, rdest::NTuple{N, UnitRange},
                              src::$S{<:Any, N}, ssrc::NTuple{N, UnitRange}) where {N}
            drange = CartesianIndices(rdest)
            srange = CartesianIndices(ssrc)
            copyto!(dest, drange, src, srange)
        end

        function Base.copyto!(dest::$D, d_range::CartesianIndices{1},
                              src::$S, s_range::CartesianIndices{1})
            len = length(d_range)
            if length(s_range) != len
                throw(ArgumentError("Copy range needs same length. Found: dest: $len, src: $(length(s_range))"))
            end
            len == 0 && return dest
            d_offset = first(d_range)[1]
            s_offset = first(s_range)[1]
            copyto!(dest, d_offset, src, s_offset, len)
        end

        function Base.copyto!(dest::$D, src::$S)
            len = length(src)
            len == 0 && return dest
            copyto!(dest, 1, src, 1, len)
        end
    end
end

# kernel-based variant for copying between wrapped GPU arrays

function linear_copy_kernel!(ctx::AbstractKernelContext, dest, dstart, src, sstart, n)
    i = linear_index(ctx)-1
    if i < n
        @inbounds dest[dstart+i] = src[sstart+i]
    end
    return
end

function Base.copyto!(dest::AbstractOrWrappedGPUArray, dstart::Integer,
                      src::AbstractOrWrappedGPUArray, sstart::Integer, n::Integer)
    n == 0 && return dest
    n < 0 && throw(ArgumentError(string("tried to copy n=", n, " elements, but n should be nonnegative")))
    destinds, srcinds = LinearIndices(dest), LinearIndices(src)
    (checkbounds(Bool, destinds, dstart) && checkbounds(Bool, destinds, dstart+n-1)) || throw(BoundsError(dest, dstart:dstart+n-1))
    (checkbounds(Bool, srcinds, sstart)  && checkbounds(Bool, srcinds, sstart+n-1))  || throw(BoundsError(src,  sstart:sstart+n-1))

    gpu_call(linear_copy_kernel!,
             dest, dstart, src, sstart, n;
             total_threads=n)
    return dest
end

# variants that materialize the GPU wrapper before copying from or to the CPU

function Base.copyto!(dest::Array, dstart::Integer,
                      src::WrappedGPUArray, sstart::Integer, n::Integer)
    temp = similar(src, n)
    copyto!(temp, 1, src, sstart, n)
    copyto!(dest, dstart, temp, 1, n)
    return dest
end

function Base.copyto!(dest::WrappedGPUArray, dstart::Integer,
                      src::Array, sstart::Integer, n::Integer)
    temp = similar(dest, n)
    copyto!(temp, 1, src, sstart, n)
    copyto!(dest, dstart, temp, 1, n)
    return dest
end

## generalized blocks of heterogeneous memory

function cartesian_copy_kernel!(ctx::AbstractKernelContext, dest, dest_offsets, src, src_offsets, shape, length)
    i = linear_index(ctx)
    if i <= length
        idx = CartesianIndices(shape)[i]
        @inbounds dest[idx + dest_offsets] = src[idx + src_offsets]
    end
    return
end

function Base.copyto!(dest::AbstractOrWrappedGPUArray{<:Any, N}, destcrange::CartesianIndices{N},
                      src::AbstractOrWrappedGPUArray{<:Any, N}, srccrange::CartesianIndices{N}) where {N}
    shape = size(destcrange)
    if shape != size(srccrange)
        throw(DimensionMismatch("Ranges don't match their size. Found: $shape, $(size(srccrange))"))
    end
    len = length(destcrange)
    len == 0 && return dest

    dest_offsets = first(destcrange) - oneunit(CartesianIndex{N})
    src_offsets = first(srccrange) - oneunit(CartesianIndex{N})
    gpu_call(cartesian_copy_kernel!,
             dest, dest_offsets, src, src_offsets, shape, len;
             total_threads=len)
    dest
end

# XXX: these generalizations between non-linear CPU and GPU memory are very inefficient,
#       because it first materializes as linear memory.
# TODO: loop over linear sections of memory and perform linear copies

# NOTE: typed with Array because of ambiguities

function Base.copyto!(dest::AbstractGPUArray{T, N}, destcrange::CartesianIndices{N},
                      src::Array{T, N}, srccrange::CartesianIndices{N}) where {T, N}
    src_gpu = typeof(dest)(map(idx-> src[idx], srccrange))
    nrange = CartesianIndices(size(src_gpu))
    copyto!(dest, destcrange, src_gpu, nrange)
    dest
end

function Base.copyto!(dest::Array{T, N}, destcrange::CartesianIndices{N},
                      src::AbstractGPUArray{T, N}, srccrange::CartesianIndices{N}) where {T, N}
    dest_gpu = similar(src, size(destcrange))
    nrange = CartesianIndices(size(dest_gpu))
    copyto!(dest_gpu, nrange, src, srccrange)
    copyto!(dest, destcrange, Array(dest_gpu), nrange)
    dest
end

## other

Base.copy(x::AbstractGPUArray) = error("Not implemented") # COV_EXCL_LINE

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
