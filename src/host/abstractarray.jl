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
    cached::Bool
end

function DataRef(finalizer, ref::D) where {D}
    rc = RefCounted{D}(ref, finalizer, Threads.Atomic{Int}(1))
    DataRef{D}(rc, false, false)
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
    # copies of cached references are not managed by the cache, so
    # we need to mark them as such to make sure their refcount can drop.
    return DataRef{D}(ref.rc, false, false)
end

function unsafe_free!(ref::DataRef)
    if ref.cached
        # lifetimes of cached references are tied to the cache.
        return
    end
    if ref.freed
        # multiple frees *of the same object* are allowed.
        # we should only ever call `release` once per object, though,
        # as multiple releases of the underlying data is not allowed.
        return
    end
    ref.freed = true
    release(ref.rc)
    return
end


## alias detection

"""
    GPUArrays.memory_location(A::AbstractGPUArray) -> Union{Nothing, Tuple{UInt, Int}}

Where the elements of `A` are stored: a coordinate `base` that identifies the allocation
holding them (normally its address), and the byte `offset` of the first element from it.
GPUArrays defines `Base.dataids`, `Base.mightalias` and the check Base uses to compare two
`SubArray`s of the same parent for every `AbstractGPUArray`; with this hook, they compare the
byte ranges arrays cover instead of the allocations they share.

Two arrays are taken to share memory when their byte ranges, `sizeof(A)` bytes from
`base + offset`, overlap. For that to be sound:

- Every array derived from an allocation (a view, reshape or reinterpretation, which are
  built with `derive`) must report the same `base`, and a nonnegative `offset` into it.
  Memory wrapped from a pointer into the middle of an allocation (`unsafe_wrap`) reports a
  `base` of its own: it is still compared by byte range against arrays that implement this
  hook, but against wrapped arrays (e.g. a `SubArray`) and arrays of other types (e.g. an
  `Array`), only when both start at the same address.
- Arrays that share memory must report it in the same coordinates. That holds for
  addresses in a single address space, but not for memory that is reachable at two
  addresses (such as a host buffer and its device mapping) or that is wrapped twice with
  separate identities (such as two handles for one buffer): the hook cannot express such
  aliases, so a back-end that creates them has to detect them itself.
- Memory without an address may be identified by any other value that is unique to its
  allocation, such as a handle. Overlap with the coordinates of other memory only causes
  false positives.
- For valid, live arrays it must not have side effects (taking stream ownership,
  synchronizing) and must not throw, as Base checks for aliasing in many operations,
  including `copyto!` and broadcasting.

The default, `nothing`, keeps alias detection at the level of allocations: arrays alias
when they share the `DataRef` returned by `GPUArrays.storage`.
"""
memory_location(::AbstractGPUArray) = nothing

# `copy(::DataRef)` shares `rc`. Views and reshapes are built that way, so
# alias detection has to key off the shared record rather than the wrapper.
Base.dataids(ref::DataRef) = (objectid(ref.rc),)

# Base compares `dataids` whenever one of the arrays is wrapped (e.g., a `SubArray`), so they
# identify the allocation rather than the part of it the array covers: the shared `DataRef`,
# which arrays of a type without `memory_location` are compared by, and the allocation's
# coordinate. The start of the array is included too, to match an array wrapped around
# memory starting at the same address (whose `dataids` are that address). Empty arrays alias
# nothing, which Julia 1.10's `mightalias` does not check.
function Base.dataids(A::AbstractGPUArray)
    isempty(A) && return ()
    ids = Base.dataids(storage(A))
    loc = memory_location(A)
    if loc === nothing
        return isempty(ids) ? (objectid(A),) : ids
    end
    base, offset = loc
    return (ids..., base, base + offset)
end

function Base.mightalias(A::AbstractGPUArray, B::AbstractGPUArray)
    (isempty(A) || isempty(B)) && return false
    a, b = memory_location(A), memory_location(B)
    # the selector bytes of isbits-union arrays are stored outside the elements' byte range
    if a === nothing || b === nothing ||
       Base.isbitsunion(eltype(A)) || Base.isbitsunion(eltype(B))
        return !Base._isdisjoint(Base.dataids(A), Base.dataids(B))
    end
    # nonempty arrays without bytes (of a zero-size element type) can't be written to
    nA, nB = sizeof(A), sizeof(B)
    (iszero(nA) || iszero(nB)) && return false
    # widened, so that coordinates near `typemax(UInt)` (e.g. handles) don't wrap around
    startA, startB = widen(a[1]) + a[2], widen(b[1]) + b[2]
    return startA <= startB ? startB - startA < nA : startA - startB < nB
end

# Base only compares the indices of two `SubArray`s when their parents are the same memory,
# which it checks for dense arrays by comparing pointers. Converting a GPU array to a pointer
# can have side effects or throw, so compare their locations instead, or only their identity
# for arrays without one. Locations don't establish where the selector bytes of isbits-union
# arrays are, so those only match themselves too. When parents don't match, Base compares
# their `dataids`, so returning false is always sound.
function Base._parentsmatch(A::AbstractGPUArray, B::AbstractGPUArray)
    A === B && return true
    a, b = memory_location(A), memory_location(B)
    (a === nothing || b === nothing) && return false
    (Base.isbitsunion(eltype(A)) || Base.isbitsunion(eltype(B))) && return false
    return a == b && Base.elsize(A) == Base.elsize(B) && size(A) == size(B)
end
Base._parentsmatch(::AbstractGPUArray, ::DenseArray) = false
Base._parentsmatch(::DenseArray, ::AbstractGPUArray) = false

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

function Serialization.serialize(s::AbstractSerializer, @nospecialize(t::AbstractGPUArray))
    serialize_type(s, typeof(t))
    serialize(s, Array(t))
end

function Serialization.deserialize(s::AbstractSerializer, ::Type{T}) where T <: AbstractGPUArray
    A = deserialize(s)
    T(A)
end

## showing

struct ToArray end
Adapt.adapt_storage(::ToArray, xs::AbstractGPUArray) = convert(Array, xs)

# display: show is called on the materialised CPU copy, so no need to
# specialize the forwarders per element type / wrapper.
Base.print_array(io::IO, @nospecialize(X::AnyGPUArray)) =
    Base.print_array(io, adapt(ToArray(), X))

# show
function Base._show_nonempty(io::IO, @nospecialize(X::AnyGPUArray), prefix::String)
    print(io, typeof(X).name.name, "(")
    Base._show_nonempty(io, adapt(ToArray(), X), prefix)
    print(io, ")")
end
function Base._show_empty(io::IO, @nospecialize(X::AnyGPUArray))
    print(io, typeof(X).name.name, "(")
    Base._show_empty(io, adapt(ToArray(), X))
    print(io, ")")
end
function Base.show_vector(io::IO, @nospecialize(v::AnyGPUArray), args...)
    print(io, typeof(v).name.name, "(")
    Base.show_vector(io, adapt(ToArray(), v), args...)
    print(io, ")")
end

## collect to CPU (discarding wrapper type)

collect_to_cpu(xs::AbstractArray) = collect(adapt(ToArray(), xs))
Base.collect(X::AnyGPUArray) = collect_to_cpu(X)


# memory copying

# expects the GPU array type to have linear `copyto!` methods (i.e. accepting an integer
# offset and length) from and to CPU arrays and between GPU arrays.

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

Base.deepcopy_internal(@nospecialize(x::AbstractGPUArray), ::IdDict) = copy(x)


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
function Base.append!(a::AbstractGPUVector, @nospecialize(items::Tuple))
    append!(a, collect(items))
    return a
end
