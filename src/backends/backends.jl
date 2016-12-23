global current_context, make_current
let compute_contexts = Context[]
    current_context() = last(compute_contexts)
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

include("supported_backends.jl")
