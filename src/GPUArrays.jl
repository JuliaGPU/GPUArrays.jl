__precompile__(true)
module GPUArrays

abstract Context
using Sugar

include("arrays.jl")
export buffer, context, GPUArray

include(joinpath("backends", "backends.jl"))
export is_backend_supported, supported_backends


end # module
