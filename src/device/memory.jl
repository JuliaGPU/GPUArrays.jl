# on-device memory management

export @LocalMemory


## thread-local array

"""
Creates a local static memory shared inside one block.
Equivalent to `__local` of OpenCL or `__shared__ (<variable>)` of CUDA.
"""
macro LocalMemory(ctx, T, N)
    id = gensym("local_memory")
    quote
        LocalMemory($(esc(ctx)), $(esc(T)), Val($(esc(N))), Val($(QuoteNode(id))))
    end
end

"""
Creates a block local array pointer with `T` being the element type
and `N` the length. Both T and N need to be static! C is a counter for
approriately get the correct Local mem id in CUDAnative.
This is an internal method which needs to be overloaded by the GPU Array backends
"""
function LocalMemory(ctx, ::Type{T}, ::Val{dims}, ::Val{id}) where {T, dims, id}
    error("Not implemented") # COV_EXCL_LINE
end
