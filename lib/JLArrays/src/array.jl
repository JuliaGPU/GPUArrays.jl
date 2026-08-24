# host array type

function check_eltype(T)
  if !Base.allocatedinline(T)
    explanation = explain_allocatedinline(T)
    error("""
      JLArray only supports element types that are allocated inline.
      $explanation""")
  end
end

mutable struct JLArray{T, N} <: AbstractGPUArray{T, N}
    data::DataRef{Vector{UInt8}}

    offset::Int        # offset of the data in the buffer, in bytes

    dims::Dims{N}

    # allocating constructor
    function JLArray{T,N}(::UndefInitializer, dims::Dims{N}) where {T,N}
        check_eltype(T)
        maxsize = prod(dims) * sizeof(T)

        ref = GPUArrays.cached_alloc((JLArray, maxsize)) do
            data = Vector{UInt8}(undef, maxsize)
            DataRef(data) do data
                resize!(data, 0)
            end
        end

        obj = new{T, N}(ref, 0, dims)
        finalizer(unsafe_free!, obj)
        return obj
    end

    # low-level constructor for wrapping existing data
    function JLArray{T,N}(ref::DataRef{Vector{UInt8}}, dims::Dims{N};
                          offset::Int=0) where {T,N}
        check_eltype(T)
        obj = new{T,N}(ref, offset, dims)
        finalizer(unsafe_free!, obj)
        return obj
    end
end

GPUArrays.storage(a::JLArray) = a.data
GPUArrays.dense_array_type(a::JLArray{T, N}) where {T, N} = JLArray{T, N}
GPUArrays.dense_array_type(::Type{JLArray{T, N}}) where {T, N} = JLArray{T, N}
GPUArrays.dense_vector_type(a::JLArray{T, N}) where {T, N} = JLArray{T, 1}
GPUArrays.dense_vector_type(::Type{JLArray{T, N}}) where {T, N} = JLArray{T, 1}

# conversion of untyped data to a typed Array
function typed_data(x::JLArray{T}) where {T}
    unsafe_wrap(Array, pointer(x), x.dims)
end

function GPUArrays.derive(::Type{T}, a::JLArray, dims::Dims{N}, offset::Int) where {T,N}
    ref = copy(a.data)
    offset = a.offset + offset * sizeof(T)
    JLArray{T,N}(ref, dims; offset)
end


## convenience constructors

const JLVector{T} = JLArray{T,1}
const JLMatrix{T} = JLArray{T,2}
const JLVecOrMat{T} = Union{JLVector{T},JLMatrix{T}}

# type and dimensionality specified
JLArray{T,N}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N} =
    JLArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))
JLArray{T,N}(::UndefInitializer, dims::Vararg{Integer,N}) where {T,N} =
    JLArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))

# type but not dimensionality specified
JLArray{T}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N} =
  JLArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))
JLArray{T}(::UndefInitializer, dims::Vararg{Integer,N}) where {T,N} =
  JLArray{T,N}(undef, convert(Dims{N}, dims))

# empty vector constructor
JLArray{T,1}() where {T} = JLArray{T,1}(undef, 0)

Base.similar(a::JLArray{T,N}) where {T,N} = JLArray{T,N}(undef, size(a))
Base.similar(a::JLArray{T}, dims::Base.Dims{N}) where {T,N} = JLArray{T,N}(undef, dims)
Base.similar(a::JLArray, ::Type{T}, dims::Base.Dims{N}) where {T,N} = JLArray{T,N}(undef, dims)

function Base.copy(a::JLArray{T,N}) where {T,N}
    b = similar(a)
    @inbounds copyto!(b, a)
end


## derived types

export DenseJLArray, DenseJLVector, DenseJLMatrix, DenseJLVecOrMat,
       StridedJLArray, StridedJLVector, StridedJLMatrix, StridedJLVecOrMat,
       AnyJLArray, AnyJLVector, AnyJLMatrix, AnyJLVecOrMat

# dense arrays: stored contiguously in memory
DenseJLArray{T,N} = JLArray{T,N}
DenseJLVector{T} = DenseJLArray{T,1}
DenseJLMatrix{T} = DenseJLArray{T,2}
DenseJLVecOrMat{T} = Union{DenseJLVector{T}, DenseJLMatrix{T}}

# strided arrays
StridedSubJLArray{T,N,I<:Tuple{Vararg{Union{Base.RangeIndex, Base.ReshapedUnitRange,
                                            Base.AbstractCartesianIndex}}}} =
  SubArray{T,N,<:JLArray,I}
StridedJLArray{T,N} = Union{JLArray{T,N}, StridedSubJLArray{T,N}}
StridedJLVector{T} = StridedJLArray{T,1}
StridedJLMatrix{T} = StridedJLArray{T,2}
StridedJLVecOrMat{T} = Union{StridedJLVector{T}, StridedJLMatrix{T}}

Base.pointer(x::StridedJLArray{T}) where {T} = Base.unsafe_convert(Ptr{T}, x)
@inline function Base.pointer(x::StridedJLArray{T}, i::Integer) where T
    Base.unsafe_convert(Ptr{T}, x) + Base._memory_offset(x, i)
end

# anything that's (secretly) backed by a JLArray
AnyJLArray{T,N} = Union{JLArray{T,N}, WrappedArray{T,N,JLArray,JLArray{T,N}}}
AnyJLVector{T} = AnyJLArray{T,1}
AnyJLMatrix{T} = AnyJLArray{T,2}
AnyJLVecOrMat{T} = Union{AnyJLVector{T}, AnyJLMatrix{T}}


## array interface

Base.elsize(::Type{<:JLArray{T}}) where {T} = sizeof(T)

Base.size(x::JLArray) = x.dims
Base.sizeof(x::JLArray) = Base.elsize(x) * length(x)

Base.unsafe_convert(::Type{Ptr{T}}, x::JLArray{T}) where {T} =
    convert(Ptr{T}, pointer(x.data[])) + x.offset


## interop with Julia arrays

function JLArray{T,N}(xs::AbstractArray{<:Any,N}) where {T,N}
    A = JLArray{T,N}(undef, size(xs))
    copyto!(A, convert(Array{T}, xs))
    return A
end

# underspecified constructors
JLArray{T}(xs::AbstractArray{S,N}) where {T,N,S} = JLArray{T,N}(xs)
(::Type{JLArray{T,N} where T})(x::AbstractArray{S,N}) where {S,N} = JLArray{S,N}(x)
JLArray(A::AbstractArray{T,N}) where {T,N} = JLArray{T,N}(A)

# idempotency
JLArray{T,N}(xs::JLArray{T,N}) where {T,N} = xs

# adapt for the GPU
jl(xs) = adapt(JLArray, xs)
## don't convert isbits types since they are already considered GPU-compatible
Adapt.adapt_storage(::Type{JLArray}, xs::AbstractArray) =
  isbits(xs) ? xs : convert(JLArray, xs)
## if an element type is specified, convert to it
Adapt.adapt_storage(::Type{<:JLArray{T}}, xs::AbstractArray) where {T} =
  isbits(xs) ? xs : convert(JLArray{T}, xs)

# adapt back to the CPU
Adapt.adapt_storage(::Type{Array}, xs::JLArray) = convert(Array, xs)

# adapt for the device
@static if !isdefined(JLArrays.KernelAbstractions, :POCL) # KA v0.9
    Adapt.adapt_storage(::Adaptor, x::JLArray{T,N}) where {T,N} =
      JLDeviceArray{T,N}(x.data[], x.offset, x.dims)
else
    function Adapt.adapt_storage(::Adaptor, x::JLArray{T,N}) where {T,N}
        arr = typed_data(x)
        Adapt.adapt_storage(POCL.KernelAdaptor([pointer(arr)]), arr)
    end
end

## conversions

Base.convert(::Type{T}, x::T) where T <: JLArray = x


## memory operations

function Base.copyto!(dest::Array{T}, d_offset::Integer,
                      source::DenseJLArray{T}, s_offset::Integer,
                      amount::Integer) where T
    amount==0 && return dest
    @boundscheck checkbounds(dest, d_offset)
    @boundscheck checkbounds(dest, d_offset+amount-1)
    @boundscheck checkbounds(source, s_offset)
    @boundscheck checkbounds(source, s_offset+amount-1)
    GC.@preserve dest source Base.unsafe_copyto!(pointer(dest, d_offset),
                                                 pointer(source, s_offset), amount)
    return dest
end

Base.copyto!(dest::Array{T}, source::DenseJLArray{T}) where {T} =
    copyto!(dest, 1, source, 1, length(source))

function Base.copyto!(dest::DenseJLArray{T}, d_offset::Integer,
                      source::Array{T}, s_offset::Integer,
                      amount::Integer) where T
    amount==0 && return dest
    @boundscheck checkbounds(dest, d_offset)
    @boundscheck checkbounds(dest, d_offset+amount-1)
    @boundscheck checkbounds(source, s_offset)
    @boundscheck checkbounds(source, s_offset+amount-1)
    GC.@preserve dest source Base.unsafe_copyto!(pointer(dest, d_offset),
                                                 pointer(source, s_offset), amount)
    return dest
end

Base.copyto!(dest::DenseJLArray{T}, source::Array{T}) where {T} =
    copyto!(dest, 1, source, 1, length(source))

function Base.copyto!(dest::DenseJLArray{T}, d_offset::Integer,
                      source::DenseJLArray{T}, s_offset::Integer,
                      amount::Integer) where T
    amount==0 && return dest
    @boundscheck checkbounds(dest, d_offset)
    @boundscheck checkbounds(dest, d_offset+amount-1)
    @boundscheck checkbounds(source, s_offset)
    @boundscheck checkbounds(source, s_offset+amount-1)
    GC.@preserve dest source Base.unsafe_copyto!(pointer(dest, d_offset),
                                                 pointer(source, s_offset), amount)
    return dest
end

Base.copyto!(dest::DenseJLArray{T}, source::DenseJLArray{T}) where {T} =
    copyto!(dest, 1, source, 1, length(source))

function Base.resize!(a::DenseJLVector{T}, nl::Integer) where {T}
    # JLArrays aren't performance critical, so simply allocate a new one
    # instead of duplicating the underlying data allocation from the ctor.
    b = JLVector{T}(undef, nl)
    copyto!(b, 1, a, 1, min(length(a), nl))

    # replace the data, freeing the old one and increasing the refcount of the new one
    # to avoid it from being freed when we leave this function.
    unsafe_free!(a)
    a.data = copy(b.data)

    a.offset = b.offset
    a.dims = b.dims
    return a
end
