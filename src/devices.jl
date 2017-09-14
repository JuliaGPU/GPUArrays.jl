

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

supports_double(ctx) = false 



#=
Functions to select contexts
=#

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

const filterfuncs = """
Device can be filtered by passing `filter_funcs`, e.g. :
`is_gpu`, `is_cpu`, `(dev)-> has_atleast(dev, threads, 512)`
"""

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
