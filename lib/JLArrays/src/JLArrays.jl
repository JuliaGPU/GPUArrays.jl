# reference implementation on the CPU
# This acts as a wrapper around KernelAbstractions's parallel CPU
# functionality. It is useful for testing GPUArrays (and other packages)
# when no GPU is present.
# This file follows conventions from AMDGPU.jl

module JLArrays

export JLArray, JLVector, JLMatrix, jl, JLBackend, JLSparseVector, JLSparseMatrixCSC, JLSparseMatrixCSR

using GPUArrays

using Adapt
using SparseArrays, LinearAlgebra
using Random

import GPUArrays: dense_array_type

import KernelAbstractions

@static if isdefined(JLArrays.KernelAbstractions, :POCL) # KA v0.10
    import KernelAbstractions: POCL
end

module AS

const Generic  = 0

end

# device functionality
include("device/array.jl")

# array implementation
include("array.jl")
include("sparse.jl")
include("broadcast.jl")
include("mapreduce.jl")

# KernelAbstractions
include("JLKernels.jl")
import .JLKernels: JLBackend

end
