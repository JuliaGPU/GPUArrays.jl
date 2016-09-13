module GPUArrays

abstract Context
include("arrays.jl")
include(joinpath("backends", "backends.jl"))

end # module
