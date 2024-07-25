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

## executed on-device
include("device/abstractarray.jl")

using KernelAbstractions
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

include("deprecated.jl")


end # module
