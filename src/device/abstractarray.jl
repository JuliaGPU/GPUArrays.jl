# on-device array type

export AbstractDeviceArray


## device array

"""
    AbstractDeviceArray{T, N} <: DenseArray{T, N}

Supertype for `N`-dimensional GPU arrays (or array-like types) with elements of type `T`.
Instances of this type are expected to live on the device, see [`AbstractGPUArray`](@ref)
for host-side objects.
"""
abstract type AbstractDeviceArray{T, N} <: DenseArray{T, N} end

Base.IndexStyle(::AbstractDeviceArray) = IndexLinear()

@inline function Base.iterate(A::AbstractDeviceArray, i=1)
    if (i % UInt) - 1 < length(A)
        (@inbounds A[i], i + 1)
    else
        nothing
    end
end

function Base.sum(A::AbstractDeviceArray{T}) where T
    acc = zero(T)
    for elem in A
        acc += elem
    end
    acc
end
