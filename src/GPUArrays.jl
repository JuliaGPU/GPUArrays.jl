module GPUArrays

using Serialization
using Random
using LinearAlgebra
using Printf

using LinearAlgebra.BLAS
using Base.Cartesian

using AbstractFFTs

using Adapt

# device array
include("device/abstractarray.jl")
include("device/indexing.jl")
include("device/synchronization.jl")

# host array
include("host/abstractarray.jl")
include("host/devices.jl")
include("host/execution.jl")
include("host/construction.jl")
## integrations and specialized functionality
include("host/base.jl")
include("host/indexing.jl")
include("host/broadcast.jl")
include("host/mapreduce.jl")
include("host/linalg.jl")
include("host/random.jl")
include("host/quirks.jl")

# CPU reference implementation
include("array.jl")


end # module
