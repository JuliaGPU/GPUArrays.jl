global current_context, make_current
function default_backend()
    if is_backend_supported(:cudanative)
        CUBackend
    elseif is_backend_supported(:opencl)
        CLBackend
    else
        JLBackend
    end
end
let compute_contexts = Context[]
    function current_context()
        if isempty(compute_contexts)
            default_backend().init()
        end
        last(compute_contexts)
    end
    all_contexts() = copy(compute_contexts)
    function make_current(ctx)
        idx = findfirst(compute_contexts, ctx)
        if idx != 0
            splice!(compute_contexts, idx) # remove
        end
        push!(compute_contexts, ctx)
    end
end
#interface
function create_buffer(ctx, array) end
# BLAS support
include("blas.jl")
include("supported_backends.jl")
