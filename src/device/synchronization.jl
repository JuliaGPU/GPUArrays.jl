# synchronization

export synchronize_threads

"""
     synchronize_threads(ctx::AbstractKernelContext)

in CUDA terms `__synchronize`
in OpenCL terms: `barrier(CLK_LOCAL_MEM_FENCE)`
"""
function synchronize_threads(ctx::AbstractKernelContext)
    error("Not implemented") # COV_EXCL_LINE
end
