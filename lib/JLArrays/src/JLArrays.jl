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

import KernelAbstractions: KernelAbstractions, POCL

# the size of an element in an array, including padding up to the element's alignment
@static if VERSION < v"1.11"
    @generated aligned_sizeof(::Type{T}) where {T} = :($(Base.aligned_sizeof(T)))
else
    import Base: aligned_sizeof
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
include("sorting.jl")
include("accumulate.jl")
include("findall.jl")

# KernelAbstractions
include("JLKernels.jl")
import .JLKernels: JLBackend

# code reflection: JLArray kernels are compiled by POCL, for the device-side argument types
import .POCL: code_typed, code_warntype, code_llvm, code_native

end
