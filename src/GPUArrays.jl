__precompile__(true)
module GPUArrays

using Serialization
using Random
using LinearAlgebra
using Printf
import Base: copyto!

import Random: rand, rand!
using LinearAlgebra.BLAS
using FFTW
import FFTW: *, plan_ifft!, plan_fft!, plan_fft, plan_ifft, size, plan_bfft, plan_bfft!
import Base: pointer, similar, size, convert
import LinearAlgebra: scale!, transpose!, permutedims!
using Base: @propagate_inbounds, @pure, RefValue
using Base.Cartesian
using Random

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

export GPUArray, gpu_call, thread_blocks_heuristic, global_size, synchronize_threads
export linear_index, @linearidx, @cartesianidx, convolution!, device, synchronize
export JLArray

end # module
