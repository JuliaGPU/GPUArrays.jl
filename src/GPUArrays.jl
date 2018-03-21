__precompile__(true)
module GPUArrays

include("abstractarray.jl")
include("abstract_gpu_interface.jl")
include("ondevice.jl")
include("base.jl")
include("construction.jl")
include("blas.jl")
include("broadcast.jl")
include("devices.jl")
include("heuristics.jl")
include("indexing.jl")
include("linalg.jl")
include("mapreduce.jl")
include("vectors.jl")
include("convolution.jl")
include("testsuite/testsuite.jl")
include("jlbackend.jl")
include("random.jl")
include("pool.jl")

export GPUArray, gpu_call, thread_blocks_heuristic, global_size, synchronize_threads
export linear_index, @linearidx, @cartesianidx, convolution!, device, synchronize, maxpool2d
export JLArray

end # module
