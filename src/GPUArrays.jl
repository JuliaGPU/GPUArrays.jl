module GPUArrays

abstract Context

include("arrays.jl")
export buffer, context

include(joinpath("backends", "backends.jl"))

end # module
