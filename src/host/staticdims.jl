# arrays with a mix of static and dynamic dimensions

export StaticDimsArray

"""
    StaticDimsArray(parent::AbstractGPUArray, dims::Tuple)

An `N`-dimensional GPU array that views the memory of `parent` with the dimensions `dims`, a
tuple of `Int`s and `StaticInt`s from Static.jl, e.g. `(static(4), static(4), n)`. The static
dimensions are part of the type, so in kernels the linear indices, strides and bounds checks
involving them are computed from constants, while the other dimensions can change at runtime.

Like other GPU arrays, a `StaticDimsArray` supports broadcasting, reductions and the other
operations of GPUArrays; out-of-place operations return arrays of the back-end without the
static dimensions. In kernels it becomes a [`StaticDimsDeviceArray`](@ref).

The size information is available through `static_size` and `known_size` of
StaticArrayInterface.jl, when it is loaded.

```julia
julia> a = StaticDimsArray(JLArray{Float32}(undef, 4 * 4 * 10), (static(4), static(4), 10));

julia> size(a)
(4, 4, 10)

julia> typeof(a.dims)
Tuple{StaticInt{4}, StaticInt{4}, Int64}
```
"""
struct StaticDimsArray{T, N, D <: MixedDims{N}, A <: AbstractGPUArray{T, 1}} <: AbstractGPUArray{T, N}
    parent::A
    dims::D

    function StaticDimsArray(parent::AbstractGPUArray{T, 1}, dims::MixedDims{N}) where {T, N}
        if prod(map(dynamic, dims)) != length(parent)
            throw(DimensionMismatch("parent of length $(length(parent)) cannot be viewed with dimensions $dims"))
        end
        new{T, N, typeof(dims), typeof(parent)}(parent, dims)
    end
end

StaticDimsArray(parent::AbstractGPUArray, dims::Tuple{Vararg{Union{Integer, StaticInt}}}) =
    StaticDimsArray(vec(parent), map(d -> d isa StaticInt ? d : Int(d), dims))

# keep the dimensions of `a`, use different memory
StaticDimsArray(a::StaticDimsArray, parent::AbstractGPUArray) = StaticDimsArray(parent, a.dims)

"""
    static_size(a)

The dimensions of a [`StaticDimsArray`](@ref) or [`StaticDimsDeviceArray`](@ref), as a tuple
of `StaticInt`s and `Int`s.
"""
static_size(a::Union{StaticDimsArray, StaticDimsDeviceArray}) = a.dims

"""
    known_size(::Type{<:StaticDimsArray})

The dimensions that are known from the type of a [`StaticDimsArray`](@ref) or
[`StaticDimsDeviceArray`](@ref), with `nothing` for the dynamic ones.
"""
known_size(::Type{<:Union{StaticDimsArray{T, N, D}, StaticDimsDeviceArray{T, N, D}}}) where {T, N, D} =
    known(D)
known_size(a::Union{StaticDimsArray, StaticDimsDeviceArray}) = known_size(typeof(a))

Base.parent(a::StaticDimsArray) = a.parent
Base.size(a::StaticDimsArray) = map(dynamic, a.dims)
Base.elsize(::Type{<:StaticDimsArray{T, N, D, A}}) where {T, N, D, A} = Base.elsize(A)

KernelAbstractions.get_backend(a::StaticDimsArray) = get_backend(parent(a))

# the memory is the one of the parent, so is the alias detection
storage(a::StaticDimsArray) = storage(parent(a))
memory_location(a::StaticDimsArray) = memory_location(parent(a))

# views, reshapes and reinterpretations derive from the parent, and lose the static dimensions
derive(::Type{T}, a::StaticDimsArray, dims::Dims, offset::Int) where {T} =
    derive(T, parent(a), dims, offset)

# broadcasting: the style of the parent, for `N` dimensions
function Broadcast.BroadcastStyle(::Type{<:StaticDimsArray{T, N, D, A}}) where {T, N, D, A}
    style = Broadcast.BroadcastStyle(A)
    typeof(style)(Val(N))
end

# `similar`: keep the static dimensions when the size does not change
Base.similar(a::StaticDimsArray, ::Type{S}) where {S} =
    StaticDimsArray(similar(parent(a), S), a.dims)
function Base.similar(a::StaticDimsArray, ::Type{S}, dims::Dims) where {S}
    if dims == size(a)
        similar(a, S)
    else
        similar(parent(a), S, dims)
    end
end

Base.copy(a::StaticDimsArray) = StaticDimsArray(copy(parent(a)), a.dims)

# copies to and from the CPU go through the parent
function Base.copyto!(dest::Array{T}, dstart::Integer, src::StaticDimsArray{T},
                      sstart::Integer, n::Integer) where {T}
    copyto!(dest, dstart, parent(src), sstart, n)
end
function Base.copyto!(dest::StaticDimsArray{T}, dstart::Integer, src::Array{T},
                      sstart::Integer, n::Integer) where {T}
    copyto!(parent(dest), dstart, src, sstart, n)
    return dest
end

# kernel arguments (and other conversions of the memory)
Adapt.adapt_structure(to, a::StaticDimsArray) = wrap_static_dims(adapt(to, parent(a)), a.dims)
Adapt.adapt_structure(::ToArray, a::StaticDimsArray) = convert(Array, a)

wrap_static_dims(parent::AbstractGPUArray, dims) = StaticDimsArray(parent, dims)
wrap_static_dims(parent::AbstractVector, dims) = StaticDimsDeviceArray(parent, dims)
