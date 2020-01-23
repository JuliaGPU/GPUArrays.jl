module GPUArrays

using Serialization
using Random
using LinearAlgebra
using Printf

using LinearAlgebra.BLAS
using Base.Cartesian

using AbstractFFTs

using Adapt

# GPU interface
## core definition
include("abstractarray.jl")
include("devices.jl")
include("execution.jl")
include("ondevice.jl")
include("construction.jl")
## integrations and specialized functionality
include("base.jl")
include("indexing.jl")
include("broadcast.jl")
include("mapreduce.jl")
include("linalg.jl")
include("random.jl")

# CPU implementation
include("array.jl")

include("quirks.jl")

end # module
