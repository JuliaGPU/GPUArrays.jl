# on-device functionality

export AbstractDeviceArray


## device array

abstract type AbstractDeviceArray{T, N} <: AbstractArray{T, N} end

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
