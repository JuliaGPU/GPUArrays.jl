# core definition of the AbstractGPUArray type


# storage handling

export DataRef, unsafe_free!

# DataRef provides a helper class to manage the storage of an array.
#
# There's multiple reasons we don't just put the data directly in a GPUArray struct:
# - to share data between multiple arrays, e.g., to create views;
# - to be able to early-free data and release GC pressure.
#
# To support this, wrap the data in a DataRef instead, and use it with the following methods:
# - `ref[]`: get the data;
# - `copy(ref)`: create a new reference, increasing the reference count;
# - `unsafe_free!(ref)`: decrease the reference count, and free the data if it reaches 0.
#
# The contained RefCounted struct should not be used directly.

# shared, reference-counted state.
mutable struct RefCounted{D}
  obj::D
  finalizer
  count::Threads.Atomic{Int}
end

function retain(rc::RefCounted)
    if rc.count[] == 0
        throw(ArgumentError("Attempt to retain freed data."))
    end
    Threads.atomic_add!(rc.count, 1)
    return
end

function release(rc::RefCounted, args...)
    if rc.count[] == 0
        throw(ArgumentError("Attempt to release freed data."))
    end
    refcount = Threads.atomic_add!(rc.count, -1)
    if refcount == 1 && rc.finalizer !== nothing
        rc.finalizer(rc.obj, args...)
    end
    return
end

function Base.getindex(rc::RefCounted)
    if rc.count[] == 0
        throw(ArgumentError("Attempt to use freed data."))
    end
    rc.obj
end

# per-object state, with a flag to indicate whether the object has been freed.
# this is to support multiple calls to `unsafe_free!` on the same object,
# while only lowering the reference count of the underlying data once.
mutable struct DataRef{D}
    rc::RefCounted{D}
    freed::Bool
end

function DataRef(finalizer, ref::D) where {D}
    rc = RefCounted{D}(ref, finalizer, Threads.Atomic{Int}(1))
    DataRef{D}(rc, false)
end
DataRef(ref; kwargs...) = DataRef(nothing, ref; kwargs...)

Base.sizeof(ref::DataRef) = sizeof(ref.rc[])

function Base.getindex(ref::DataRef)
    if ref.freed
        throw(ArgumentError("Attempt to use a freed reference."))
    end
    ref.rc[]
end

function Base.copy(ref::DataRef{D}) where {D}
    if ref.freed
        throw(ArgumentError("Attempt to copy a freed reference."))
    end
    retain(ref.rc)
    return DataRef{D}(ref.rc, false)
end

function unsafe_free!(ref::DataRef, args...)
    if ref.freed
        # multiple frees *of the same object* are allowed.
        # we should only ever call `release` once per object, though,
        # as multiple releases of the underlying data is not allowed.
        return
    end
    ref.freed = true
    release(ref.rc, args...)
    return
end

# array methods

storage(x::AbstractGPUArray) = error("Not implemented") # COV_EXCL_LINE

"""
    unsafe_free!(a::GPUArray)

Release the memory of an array for reuse by future allocations. This operation is
performed automatically by the GC when an array goes out of scope, but can be called
earlier to reduce pressure on the memory allocator.
"""
unsafe_free!(x::AbstractGPUArray) = unsafe_free!(storage(x))


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

## showing

struct ToArray end
Adapt.adapt_storage(::ToArray, xs::AbstractGPUArray) = convert(Array, xs)

# display
Base.print_array(io::IO, X::AnyGPUArray) =
    Base.print_array(io, adapt(ToArray(), X))

# show
Base._show_nonempty(io::IO, X::AnyGPUArray, prefix::String) =
    Base._show_nonempty(io, adapt(ToArray(), X), prefix)
Base._show_empty(io::IO, X::AnyGPUArray) =
    Base._show_empty(io, adapt(ToArray(), X))
Base.show_vector(io::IO, v::AnyGPUArray, args...) =
    Base.show_vector(io, adapt(ToArray(), v), args...)

## collect to CPU (discarding wrapper type)

collect_to_cpu(xs::AbstractArray) = collect(adapt(ToArray(), xs))
Base.collect(X::AnyGPUArray) = collect_to_cpu(X)


# memory copying

# expects the GPU array type to have linear `copyto!` methods (i.e. accepting an integer
# offset and length) from and to CPU arrays and between GPU arrays.

function Base.copy!(dst::AbstractGPUVector, src::AbstractGPUVector)
    axes(dst) == axes(src) || throw(ArgumentError(
    "arrays must have the same axes for `copy!`. consider using `copyto!` instead"))
    copyto!(dst, src)
end

for (D, S) in ((AnyGPUArray, Array),
               (Array, AnyGPUArray),
               (AnyGPUArray, AnyGPUArray))
    @eval begin
        function Base.copyto!(dest::$D{<:Any, N}, rdest::UnitRange,
                              src::$S{<:Any, N}, ssrc::UnitRange) where {N}
            drange = CartesianIndices((rdest,))
            srange = CartesianIndices((ssrc,))
            copyto!(dest, drange, src, srange)
        end

        Base.copyto!(dest::$D, src::$S) = copyto!(dest, 1, src, 1, length(src))
    end
end

# kernel-based variant for copying between wrapped GPU arrays
@kernel function linear_copy_kernel!(dest, dstart, src, sstart, n)
    i = @index(Global, Linear)
    if i <= n
        @inbounds dest[dstart+i-1] = src[sstart+i-1]
    end
end

function Base.copyto!(dest::AnyGPUArray, dstart::Integer,
                      src::AnyGPUArray, sstart::Integer, n::Integer)
    n == 0 && return dest
    n < 0 && throw(ArgumentError(string("tried to copy n=", n, " elements, but n should be nonnegative")))
    destinds, srcinds = LinearIndices(dest), LinearIndices(src)
    (checkbounds(Bool, destinds, dstart) && checkbounds(Bool, destinds, dstart+n-1)) || throw(BoundsError(dest, dstart:dstart+n-1))
    (checkbounds(Bool, srcinds, sstart)  && checkbounds(Bool, srcinds, sstart+n-1))  || throw(BoundsError(src,  sstart:sstart+n-1))
    kernel = linear_copy_kernel!(get_backend(dest))
    kernel(dest, dstart, src, sstart, n; ndrange=n)
    return dest
end

# variants that materialize the GPU wrapper before copying from or to the CPU

function Base.copyto!(dest::Array, dstart::Integer,
                      src::WrappedGPUArray, sstart::Integer, n::Integer)
    n == 0 && return dest
    temp = similar(parent(src), n)
    copyto!(temp, 1, src, sstart, n)
    copyto!(dest, dstart, temp, 1, n)
    return dest
end

function Base.copyto!(dest::WrappedGPUArray, dstart::Integer,
                      src::Array, sstart::Integer, n::Integer)
    n == 0 && return dest
    temp = similar(parent(dest), n)
    copyto!(temp, 1, src, sstart, n)
    copyto!(dest, dstart, temp, 1, n)
    return dest
end

# variants that converts values on the CPU when there's a type mismatch
#
# we prefer to convert on the CPU where there's typically more memory / less memory pressure
# to quickly perform these very lightweight conversions

function Base.copyto!(dest::Array{T}, dstart::Integer,
                      src::AnyGPUArray{U}, sstart::Integer,
                      n::Integer) where {T,U}
    n == 0 && return dest
    temp = Vector{U}(undef, n)
    copyto!(temp, 1, src, sstart, n)
    copyto!(dest, dstart, temp, 1, n)
    return dest
end

function Base.copyto!(dest::AnyGPUArray{T}, dstart::Integer,
                      src::Array{U}, sstart::Integer, n::Integer) where {T,U}
    n == 0 && return dest
    temp = Vector{T}(undef, n)
    copyto!(temp, 1, src, sstart, n)
    copyto!(dest, dstart, temp, 1, n)
    return dest
end

## generalized blocks of heterogeneous memory

@kernel function cartesian_copy_kernel!(dest, dest_offsets, src, src_offsets)
    I = @index(Global, Cartesian)
    @inbounds dest[I + dest_offsets] = src[I + src_offsets]
end

function Base.copyto!(dest::AnyGPUArray{<:Any, N}, destcrange::CartesianIndices{N},
                      src::AnyGPUArray{<:Any, N}, srccrange::CartesianIndices{N}) where {N}
    shape = size(destcrange)
    if shape != size(srccrange)
        throw(ArgumentError("Ranges don't match their size. Found: $shape, $(size(srccrange))"))
    end
    len = length(destcrange)
    len == 0 && return dest

    # linear copy if we can
    if N == 1
        d_offset = first(destcrange)[1]
        s_offset = first(srccrange)[1]
        return copyto!(dest, d_offset, src, s_offset, len)
    end

    dest_offsets = first(destcrange) - oneunit(CartesianIndex{N})
    src_offsets = first(srccrange) - oneunit(CartesianIndex{N})
    kernel = cartesian_copy_kernel!(get_backend(dest))
    kernel(dest, dest_offsets, src, src_offsets; ndrange=shape)
    dest
end

for (dstTyp, srcTyp) in (AbstractGPUArray=>Array, Array=>AbstractGPUArray)
    @eval function Base.copyto!(dst::$dstTyp{T,N}, dstrange::CartesianIndices{N},
                                src::$srcTyp{T,N}, srcrange::CartesianIndices{N}) where {T,N}
        isempty(dstrange) && return dst
        if size(dstrange) != size(srcrange)
            throw(ArgumentError("source and destination must have same size (got $(size(srcrange)) and $(size(dstrange)))"))
        end
        len = length(dstrange)
        len == 0 && return dest

        # linear copy if we can
        if N == 1
            d_offset = first(dstrange)[1]
            s_offset = first(srcrange)[1]
            return copyto!(dst, d_offset, src, s_offset, len)
        end

        # figure out how many dimensions of the Cartesian ranges map onto contiguous memory
        # in both source and destination. we will copy these one by one as linear ranges.
        contiguous_dims = 1
        for dim in 2:N
            # a slice is broken up if the previous dimension didn't cover the entire range
            if axes(src, dim-1) == axes(srcrange, dim-1) &&
            axes(dst, dim-1) == axes(dstrange, dim-1)
                contiguous_dims = dim
            else
                break
            end
        end

        m = prod(size(dstrange)[1:contiguous_dims])       # inner, contiguous length
        n = prod(size(dstrange)[contiguous_dims+1:end])   # outer non-contiguous length
        @assert m*n == length(srcrange) == length(dstrange)

        # copy linear slices
        for i in 1:m:m*n
            srcoff = LinearIndices(src)[srcrange[i]]
            dstoff = LinearIndices(dst)[dstrange[i]]
            # TODO: Use asynchronous memory copies
            copyto!(dst, dstoff, src, srcoff, m)
        end

        dst
    end
end

## other

Base.copy(x::AbstractGPUArray) = error("Not implemented") # COV_EXCL_LINE

Base.deepcopy(x::AbstractGPUArray) = copy(x)


# filtering

# TODO: filter!

# revert of JuliaLang/julia#31929
Base.filter(f, As::AbstractGPUArray) = As[map(f, As)::AbstractGPUArray{Bool}]

# appending

function Base.append!(a::AbstractGPUVector, items::AbstractVector)
    n = length(items)
    resize!(a, length(a) + n)
    copyto!(a, length(a) - n + 1, items, firstindex(items), n)
    return a
end

# this is needed because copyto! of most GPU arrays
# doesn't currently support Tuple sources
function Base.append!(a::AbstractGPUVector, items::Tuple)
    append!(a, collect(items))
    return a
end
