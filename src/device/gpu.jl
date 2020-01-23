# gpu-specific functionality

export global_size, synchronize_threads


## synchronization

"""
     synchronize_threads(state)

in CUDA terms `__synchronize`
in OpenCL terms: `barrier(CLK_LOCAL_MEM_FENCE)`
"""
function synchronize_threads(state)
    error("Not implemented") # COV_EXCL_LINE
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
    error("Not implemented") # COV_EXCL_LINE
end
