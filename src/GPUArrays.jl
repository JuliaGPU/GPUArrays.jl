module GPUArrays

using GPUToolbox
using KernelAbstractions
using Serialization
using Random
using LinearAlgebra
using Printf

using LinearAlgebra.BLAS
using Base.Cartesian

using Adapt
using LLVM.Interop

using Reexport
@reexport using GPUArraysCore

import KernelAbstractions as KA
import AcceleratedKernels as AK

# device functionality
include("device/abstractarray.jl")

# host abstractions
include("host/abstractarray.jl")
include("host/construction.jl")
## integrations and specialized methods
include("host/base.jl")
include("host/indexing.jl")
include("host/broadcast.jl")
include("host/mapreduce.jl")
include("host/linalg.jl")
include("host/math.jl")
include("host/random.jl")
include("host/sorting.jl")
include("host/quirks.jl")
include("host/uniformscaling.jl")
include("host/statistics.jl")
include("host/alloc_cache.jl")


end # module
