# on-device functionality

export AbstractDeviceArray, @LocalMemory


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


## thread-local array

const shmem_counter = Ref{Int}(0)

"""
Creates a local static memory shared inside one block.
Equivalent to `__local` of OpenCL or `__shared__ (<variable>)` of CUDA.
"""
macro LocalMemory(state, T, N)
    id = (shmem_counter[] += 1)
    quote
        LocalMemory($(esc(state)), $(esc(T)), Val($(esc(N))), Val($id))
    end
end

"""
Creates a block local array pointer with `T` being the element type
and `N` the length. Both T and N need to be static! C is a counter for
approriately get the correct Local mem id in CUDAnative.
This is an internal method which needs to be overloaded by the GPU Array backends
"""
function LocalMemory(state, ::Type{T}, ::Val{dims}, ::Val{id}) where {T, dims, id}
    error("Not implemented") # COV_EXCL_LINE
end
