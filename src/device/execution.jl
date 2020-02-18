# kernel execution

export AbstractGPUBackend, AbstractKernelContext, gpu_call, synchronize, thread_blocks_heuristic

abstract type AbstractGPUBackend end

abstract type AbstractKernelContext end

"""
    backend(T::Type)
    backend(x)

Gets the GPUArrays back-end responsible for managing arrays of type `T`.
"""
backend(::Type) = error("This object is not a GPU array") # COV_EXCL_LINE
backend(x) = backend(typeof(x))

"""
    gpu_call(kernel::Function, arg0, args...; kwargs...)

Executes `kernel` on the device that backs `arg` (see [`backend`](@ref)), passing along any
arguments `args`. Additionally, the kernel will be passed the kernel execution context (see
[`AbstractKernelContext`]), so its signature should be `(ctx::AbstractKernelContext, arg0,
args...)`.

The keyword arguments `kwargs` are not passed to the function, but are interpreted on the
host to influence how the kernel is executed. The following keyword arguments are supported:

- `target::AbstractArray`: specify which array object to use for determining execution
  properties (defaults to the first argument `arg0`).
- `total_threads::Int`: how many threads should be launched _in total_. The actual number of
  threads and blocks is determined using a heuristic. Defaults to the length of `arg0` if
  no other keyword arguments that influence the launch configuration are specified.
- `threads::Int` and `blocks::Int`: configure exactly how many threads and blocks are
  launched. This cannot be used in combination with the `total_threads` argument.
- `name::String`: inform the back end about the name of the kernel to be executed.
  This can be used to emit better diagnostics, and is useful with anonymous kernels.
"""
function gpu_call(kernel::Base.Callable, args...;
                  target::AbstractArray=first(args),
                  total_threads::Union{Int,Nothing}=nothing,
                  threads::Union{Int,Nothing}=nothing,
                  blocks::Union{Int,Nothing}=nothing,
                  name::Union{String,Nothing}=nothing)
    # non-trivial default values for launch configuration
    if total_threads===nothing && threads===nothing && blocks===nothing
        total_threads = length(target)
    elseif total_threads===nothing
        if threads === nothing
            threads = 1
        end
        if blocks === nothing
            blocks = 1
        end
    elseif threads!==nothing || blocks!==nothing
        error("Cannot specify both total_threads and threads/blocks configuration")
    end

    if total_threads !== nothing
        gpu_call(backend(target), kernel, args, total_threads; name=name)
    else
        gpu_call(backend(target), kernel, args, threads, blocks; name=name)
    end
end

# gpu_call method with a simple launch configuration heuristic.
# this can be specialised if more sophisticated heuristics are available.
function gpu_call(backend::AbstractGPUBackend, kernel, args, total_threads::Int; kwargs...)
    threads = clamp(total_threads, 1, 256)
    blocks = max(ceil(Int, total_threads / threads), 1)

    gpu_call(backend, kernel, args, threads, blocks; kwargs...)
end

# bottom-line gpu_call method that is expected to be implemented by the back end
gpu_call(backend::AbstractGPUBackend, kernel, args, threads::Int, blocks::Int; kwargs...) = error("Not implemented") # COV_EXCL_LINE

"""
    synchronize(A::AbstractArray)

Blocks until all operations are finished on `A`
"""
function synchronize(A::AbstractArray)
    # fallback is a noop, for backends not needing synchronization. This
    # makes it easier to write generic code that also works for AbstractArrays
end
