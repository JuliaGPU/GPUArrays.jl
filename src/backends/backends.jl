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
Blocks until all operations are finished on `A`
"""
function synchronize(A::AbstractArray)
    # fallback is a noop, for backends not needing synchronization. This
    # makes it easier to write generic code that also works for AbstractArrays
end

"""
`A` must be a gpu Array and will help to dispatch to the correct GPU backend
and can supply queues and contexts.
Calls `f` on args on the GPU, falls back to a normal call if there is no backend.
"""
function gpu_call(f, A::AbstractArray, args::Tuple, worksize = length(A), localsize = nothing)
    f(args...)
end

free(x::AbstractArray) = nothing

#=
Functions to select contexts
=#

threads(device) = 0
blocks(device) = 0
global_memory(device) = 0
free_global_memory(device) = NaN
local_memory(device) = 0
name(device) = "Undefined"

function device_summary(io::IO, device)
    println(io, "Device: ", name(device))
    for (n, f) in (:threads => threads, :blocks => blocks)
        @printf(io, "%19s: %s\n", string(n), string(f(device)))
    end
    for (n, f) in (:global_memory => global_memory, :free_global_memory => free_global_memory, :local_memory => local_memory)
        @printf(io, "%19s: %f mb\n", string(n), f(device) / 10^6)
    end
    return
end

################################
# Device selection functions for e.g. devices(filterfuncs)
is_gpu(device) = false
is_cpu(device) = false
has_atleast(device, attribute, value) = attribute(ctx_or_device) >= value

"""
Creates a new context from `device` without caching the resulting context.
"""
function new_context(device)
    error("Device $device not supported")
end

# BLAS support
hasblas(x) = false
include("blas.jl")
include("supported_backends.jl")
include("shared.jl")

function backend_module(sym::Symbol)
    if sym in supported_backends()
        if sym == :julia
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
    backend_module(sym).init(args...; kw_args...)
end

function init(filterfuncs::Function...; kw_args...)
    devices = available_devices(filterfuncs...)
    if isempty(devices)
        error("No device found for: $(join(string.(filterfuncs), " "))")
    end
    current_backend().init(first(devices))
end

active_backends() = backend_module.(supported_backends())

const global_current_backend = Ref{Module}(default_backend())

current_backend() = global_current_backend[]
current_device() = current_backend().current_device()
current_context() = current_backend().current_context()

"""
Sets the current backend to be used globally. Accepts the symbols:
:cudanative, :opencl, :julia.
"""
function setbackend!(backend::Symbol)
    setbackend!(backend_module(backend))
end

function setbackend!(backend::Module)
    global_current_backend[] = backend
    return
end

"""
Creates a temporary context for `device` and executes `f(context)` while this context is active.
Context gets destroyed afterwards. Note, that creating a temporary context is expensive.
"""
function on_device(f, device = current_device())
    ctx = new_context(device)
    f(ctx)
    destroy!(ctx)
    return
end

"""
Returns all devices for the current backend.
Can be filtered by passing `filter_funcs`, e.g. `is_gpu`, `is_cpu`, `(dev)-> has_atleast(dev, threads, 512)`
"""
function available_devices(filter_funcs...)
    result = []
    for device in current_backend().devices()
        if all(f-> f(device), filter_funcs)
            push!(result, device)
        end
    end
    result
end


"""
Returns all devices from `backends = active_backends()`.
Can be filtered by passing `filter_funcs`, e.g. `is_gpu`, `is_cpu`, `dev-> has_atleast(dev, threads, 512)`
"""
function all_devices(filter_funcs...; backends = active_backends())
    result = []
    for Module in backends
        for device in Module.devices()
            if all(f-> f(device), filter_funcs)
                push!(result, device)
            end
        end
    end
    result
end

"""
Iterates through all backends and calls `f` after initializing the current one!
"""
function perbackend(f)
    for backend in supported_backends()
        ctx = GPUArrays.init(backend)
        f(ctx)
    end
end

"""
Iterates through all available devices and calls `f(context)` after initializing the standard context for that device.
"""
function forall_devices(f, filterfuncs...)
    for device in all_devices(filterfunc...)
        ctx = init(device)
        f(ctx)
    end
end
