module GPUArrays

using Serialization
using Random
using LinearAlgebra
using Printf

using LinearAlgebra.BLAS
using Base.Cartesian

using AbstractFFTs

using Adapt

# device functionality
include("device/device.jl")
include("device/execution.jl")
## executed on-device
include("device/abstractarray.jl")
include("device/indexing.jl")
include("device/memory.jl")
include("device/synchronization.jl")

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

# CPU reference implementation
include("reference.jl")


end # module
