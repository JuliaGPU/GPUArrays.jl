# synchronization

export synchronize_threads

"""
     synchronize_threads(state)

in CUDA terms `__synchronize`
in OpenCL terms: `barrier(CLK_LOCAL_MEM_FENCE)`
"""
function synchronize_threads(state)
    error("Not implemented") # COV_EXCL_LINE
end
