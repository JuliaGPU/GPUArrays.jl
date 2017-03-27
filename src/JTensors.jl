__precompile__(true)
module JTensors

abstract Context
using Sugar

include("abstractarray.jl")
export JTensor, mapidx, linear_index

include(joinpath("backends", "backends.jl"))
export is_backend_supported, supported_backends


end # module
