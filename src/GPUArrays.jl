module GPUArrays

using KernelAbstractions
using Serialization
using Random
using LinearAlgebra
using SparseArrays
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
include("host/gemm.jl")
include("host/linalg.jl")
include("host/math.jl")
include("host/random.jl")
include("host/quirks.jl")
include("host/uniformscaling.jl")
include("host/statistics.jl")
include("host/sparse/types.jl")
include("host/sparse/primitives.jl")
include("host/sparse/conversions.jl")
include("host/sparse/assembly.jl")
include("host/sparse/show.jl")
include("host/sparse/indexing.jl")
include("host/sparse/broadcast.jl")
include("host/sparse/reductions.jl")
include("host/alloc_cache.jl")

include("deprecated.jl")

end # module
