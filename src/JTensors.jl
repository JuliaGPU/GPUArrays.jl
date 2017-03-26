__precompile__(true)
module JTensors

abstract Context
using Sugar

include("abstractarray.jl")
export buffer, context, JTensor

include(joinpath("backends", "backends.jl"))
export is_backend_supported, supported_backends


end # module
