module GPUArrays

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

using KernelAbstractions

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
include("host/quirks.jl")
include("host/uniformscaling.jl")
include("host/statistics.jl")
include("host/allocations_cache.jl")


end # module
