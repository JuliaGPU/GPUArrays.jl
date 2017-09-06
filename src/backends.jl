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


#interface
function create_buffer(ctx, array) end



"""
`A` must be a gpu Array and will help to dispatch to the correct GPU backend
and can supply queues and contexts.
Calls `f` on args on the GPU, falls back to a normal call if there is no backend.
"""
function gpu_call(f, A::AbstractArray, args::Tuple, worksize = length(A), localsize = nothing)
    f(args...)
end

free(x::AbstractArray) = nothing


function backend_module(sym::Symbol)
    if sym in supported_backends()
        if sym == :threaded
            JLBackend
        elseif sym == :cudanative
            CUBackend
        elseif sym == :opencl
            CLBackend
        elseif sym == :opengl
            GLBackend
        end
    else
        error("$sym not a supported backend. Try one of: $(supported_backends())")
    end
end
function init(sym::Symbol, args...; kw_args...)
    init(backend_module(sym), args...; kw_args...)
end
function init(mod::Module, args...; kw_args...)
    setbackend!(mod)
    init(args...; kw_args...)
end

function init(filterfuncs::Function...; kw_args...)
    devices = available_devices(filterfuncs...)
    devices = sort(devices, by = x-> !is_gpu(x)) # prioritize gpu devices
    if isempty(devices)
        error("No device found for: $(join(string.(filterfuncs), " "))")
    end
    init(first(devices))
end

# BLAS support
hasblas(x) = false
include("blas.jl")
include("supported_backends.jl")
include("shared.jl")



active_backends() = backend_module.(supported_backends())

const global_current_backend = Ref{Module}(default_backend())

current_backend() = global_current_backend[]
current_device() = current_backend().current_device()
current_context() = current_backend().current_context()

"""
Sets the current backend to be used globally. Accepts the symbols:
:cudanative, :opencl, :threaded.
"""
function setbackend!(backend::Symbol)
    setbackend!(backend_module(backend))
end

function setbackend!(backend::Module)
    global_current_backend[] = backend
    return
end


export is_cudanative, is_julia, is_opencl, on_device
export opencl, cudanative, threaded
