# functionality for vendor-agnostic kernels

## indexing

# thread indexing functions
for sym in (:x, :y, :z)
    for f in (:blockidx, :blockdim, :threadidx, :griddim)
        fname = Symbol(string(f, '_', sym))
        @eval $fname(state)::Int = error("Not implemented")
        @eval export $fname
    end
end

"""
    global_size(state)

Global size == blockdim * griddim == total number of kernel execution
"""
@inline function global_size(state)
    # TODO nd version
    griddim_x(state) * blockdim_x(state)
end

"""
    linear_index(state)

linear index corresponding to each kernel launch (in OpenCL equal to get_global_id).

"""
@inline function linear_index(state)
    (blockidx_x(state) - 1) * blockdim_x(state) + threadidx_x(state)
end

"""
    linearidx(A, statesym = :state)

Macro form of `linear_index`, which calls return when out of bounds.
So it can be used like this:

    ```julia
    function kernel(state, A)
        idx = @linear_index A state
        # from here on it's save to index into A with idx
        @inbounds begin
            A[idx] = ...
        end
    end
    ```
"""
macro linearidx(A, statesym = :state)
    quote
        x1 = $(esc(A))
        i1 = linear_index($(esc(statesym)))
        i1 > length(x1) && return
        i1
    end
end

"""
    cartesianidx(A, statesym = :state)

Like [`@linearidx(A, statesym = :state)`](@ref), but returns an N-dimensional `NTuple{ndim(A), Int}` as index
"""
macro cartesianidx(A, statesym = :state)
    quote
        x = $(esc(A))
        i2 = @linearidx(x, $(esc(statesym)))
        gpu_ind2sub(x, i2)
    end
end


## synchronization

"""
     synchronize_threads(state)

in CUDA terms `__synchronize`
in OpenCL terms: `barrier(CLK_LOCAL_MEM_FENCE)`
"""
function synchronize_threads(state)
    error("Not implemented")
end


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


## device memory

const shmem_counter = Ref{Int}(0)

"""
Creates a local static memory shared inside one block.
Equivalent to `__local` of OpenCL or `__shared__ (<variable>)` of CUDA.
"""
macro LocalMemory(state, T, N)
    id = (shmem_counter[] += 1)
    quote
        lémem = LocalMemory($(esc(state)), $(esc(T)), Val($(esc(N))), Val($id))
        AbstractDeviceArray(lémem, $(esc(N)))
    end
end

export @LocalMemory

"""
Creates a block local array pointer with `T` being the element type
and `N` the length. Both T and N need to be static! C is a counter for
approriately get the correct Local mem id in CUDAnative.
This is an internal method which needs to be overloaded by the GPU Array backends
"""
function LocalMemory(state, ::Type{T}, ::Val{N}, ::Val{C}) where {N, T, C}
    error("Not implemented")
end
