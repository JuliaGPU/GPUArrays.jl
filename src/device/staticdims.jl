# arrays with a mix of static and dynamic dimensions

export StaticDimsDeviceArray

# Each dimension is either a `StaticInt`, a compile-time constant that takes no space, or an
# `Int`. E.g. `Tuple{StaticInt{4}, Int}` for an array of size `(4, n)`.
const MixedDims{N} = NTuple{N, Union{Int, StaticInt}}

# Column-major linear index of the (in-bounds) indices `I`, as a sum of strided terms:
# `1 + (i1 - 1) + d1 * (i2 - 1) + (d1 * d2) * (i3 - 1)` for the dimensions `(d1, d2, d3)`.
# The strides are products of `StaticInt`s as long as the dimensions are, i.e. constants.
# (Unlike the Horner form `i1 + d1 * ((i2 - 1) + d2 * (i3 - 1))`, the terms of indices that
# don't change in a loop can be hoisted out of it when the dimensions are dynamic.)
# (Unrolled, as recursing over the dimensions with strides of growing types would not infer.)
@inline @generated function static_linear_index(dims::Tuple, I::NTuple{N, Int}) where {N}
    ex = quote
        stride_1 = StaticInt(1)
    end
    for k in 2:N
        push!(ex.args, :($(Symbol(:stride_, k)) = $(Symbol(:stride_, k - 1)) * dims[$(k - 1)]))
    end
    terms = [:(dynamic($(Symbol(:stride_, k))) * (I[$k] - 1)) for k in 1:N]
    push!(ex.args, :(+(1, $(terms...))))
    return ex
end

"""
    StaticDimsDeviceArray{T,N,D,A} <: AbstractDeviceArray{T,N}

The device-side counterpart of [`StaticDimsArray`](@ref): a vector `parent::A` (typically a
device vector of the back-end) viewed as an `N`-dimensional array with the dimensions
`dims::D`, a tuple of `Int`s and `StaticInt`s. If the parent is `isbits`, so is the array.
"""
struct StaticDimsDeviceArray{T, N, D <: MixedDims{N}, A <: AbstractVector{T}} <: AbstractDeviceArray{T, N}
    parent::A
    dims::D
end

Base.parent(a::StaticDimsDeviceArray) = a.parent

# adapt the parent, e.g. so that KernelAbstractions' `@Const` reaches the back-end's read-only
# loads (`ld.global.nc` with CUDA)
Adapt.adapt_structure(to, a::StaticDimsDeviceArray) =
    wrap_static_dims(adapt(to, parent(a)), a.dims)

Base.size(a::StaticDimsDeviceArray) = map(dynamic, a.dims)
Base.elsize(::Type{<:StaticDimsDeviceArray{T, N, D, A}}) where {T, N, D, A} = Base.elsize(A)

@inline function Base.getindex(a::StaticDimsDeviceArray, i::Int)
    @boundscheck checkbounds(a, i)
    @inbounds a.parent[i]
end
@inline function Base.getindex(a::StaticDimsDeviceArray{T, N}, I::Vararg{Int, N}) where {T, N}
    @boundscheck checkbounds(a, I...)
    @inbounds a.parent[static_linear_index(a.dims, I)]
end

@inline function Base.setindex!(a::StaticDimsDeviceArray, x, i::Int)
    @boundscheck checkbounds(a, i)
    @inbounds a.parent[i] = x
    return a
end
@inline function Base.setindex!(a::StaticDimsDeviceArray{T, N}, x, I::Vararg{Int, N}) where {T, N}
    @boundscheck checkbounds(a, I...)
    @inbounds a.parent[static_linear_index(a.dims, I)] = x
    return a
end
