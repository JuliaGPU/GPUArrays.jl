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
"""
Hardware threads of device
"""
threads(device) = 0

"""
Blocks that group together hardware threads
"""
blocks(device) = 1
"""
Global memory, e.g. VRAM or RAM of device
"""
global_memory(device) = 0

"""
Free global memory. Isn't supported for AMD cards right now, in which case it returns NaN,
so don't rely on the output of this function.
"""
free_global_memory(device) = NaN

"""
Block local memory
"""
local_memory(device) = 0

"""
Hardware name of a device
"""
name(device) = "Undefined"

"""
Summarizes all features of a device and prints it to `io`
"""
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
"""
Returns true if `device` is a gpu
"""
is_gpu(device) = false

"""
Returns true if `device` is a cpu
"""
is_cpu(device) = false

"""
Checks a device for a certain attribute and returns true if it has at least `value`.
Can be used with e.g. `threads`, `blocks`, `global_memory`, `local_memory`
"""
has_atleast(device, attribute, value) = attribute(ctx_or_device) >= value


#################################
# Context filter functions
# Works for context objects as well but is overloaded in the backends
is_opencl(ctx::Symbol) = ctx == :opencl
is_cudanative(ctx::Symbol) =  ctx == :cudanative
is_julia(ctx::Symbol) =  ctx == :threaded
is_opengl(ctx::Symbol) =  ctx == :opengl

is_opencl(ctx) = false
is_cudanative(ctx) = false
is_julia(ctx) = false
is_opengl(ctx) = false

const filterfuncs = """
Device can be filtered by passing `filter_funcs`, e.g. :
`is_gpu`, `is_cpu`, `(dev)-> has_atleast(dev, threads, 512)`
"""

"""
Initializes the opencl backend with a default device.
$filterfuncs
"""
opencl(filterfuncs...) = init(:opencl, filterfuncs...)

"""
Initializes the cudanative backend with a default device.
$filterfuncs
"""
cudanative(filterfuncs...) = init(:cudanative, filterfuncs...)
"""
Initializes the threaded backend with a default device.
$filterfuncs
"""
threaded(filterfuncs...) = init(:threaded, filterfuncs...)



"""
Creates a new context from `device` without caching the resulting context.
"""
function new_context(device)
    error("Device $device not supported")
end

"""
Destroys context, freeing all it's resources.
"""
function destroy!(context)
    error("Device $context not supported")
end

"""
Resets a context freeing all resources and creating a new context.
"""
function reset!(context)
    error("Context $context not supported")
end

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

"""
Creates a temporary context for `device` and executes `f(context)` while this context is active.
Context gets destroyed afterwards. Note, that creating a temporary context is expensive.
"""
function on_device(f, device = current_device())
    ctx = new_context(device)
    try
        f(ctx)
    finally
        destroy!(ctx)
    end
    return
end

"""
Returns all devices for the current backend.
$filterfuncs
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
$filterfuncs
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
Iterates through all available devices and calls `f(context)` after initializing the standard context for that device.
"""
function forall_devices(func, filterfuncs...)
    for device in all_devices(filterfuncs...)
        ctx = init(device)
        func(ctx)
    end
end


export is_cudanative, is_julia, is_opencl, on_device
export opencl, cudanative, threaded
