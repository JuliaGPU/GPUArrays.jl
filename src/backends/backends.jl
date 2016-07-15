abstract Context
global compute_context

#interface
function create_buffer(::Context, ::Array) end

include("opencl.jl")
