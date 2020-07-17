# kernel execution

export AbstractGPUBackend, AbstractKernelContext, gpu_call

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
        @assert total_threads > 0
        gpu_call(backend(target), kernel, args, total_threads; name=name)
    else
        @assert threads > 0
        @assert blocks > 0
        gpu_call(backend(target), kernel, args, threads, blocks; name=name)
    end
end

function gpu_call(backend::AbstractGPUBackend, kernel, args, total_threads::Int; kwargs...)
    heuristic = launch_heuristic(backend, kernel, args...)
    config = launch_configuration(backend, heuristic, total_threads)
    gpu_call(backend, kernel, args, config.threads, config.blocks; kwargs...)
end

# bottom-line gpu_call method that is expected to be implemented by the back end
gpu_call(backend::AbstractGPUBackend, kernel, args, threads::Int, blocks::Int; kwargs...) =
    error("Not implemented") # COV_EXCL_LINE

# how many threads and blocks this kernel need to fully saturate the GPU.
# this can be specialised if more sophisticated heuristics are available.
#
# the `maximize_blocksize` indicates whether the kernel benifits from a large block size
function launch_heuristic(backend::AbstractGPUBackend, kernel, args...;
                          maximize_blocksize=false)
    return (threads=256, blocks=32)
end

# determine how many threads and blocks to actually launch given upper limits.
# returns a tuple of blocks, threads, and elements_per_thread (which is always 1
# unless specified that the kernel can handle a number of elements per thread)
function launch_configuration(backend::AbstractGPUBackend, heuristic,
                              elements::Int, elements_per_thread::Int=1)
    threads = clamp(elements, 1, heuristic.threads)
    blocks = max(cld(elements, threads), 1)

    if elements_per_thread == 1 || blocks < config.blocks
        (threads=threads, blocks=blocks, elements_per_thread=1)
    else
        nelem = clamp(cld(blocks, heuristic.blocks), 1, elements_per_thread)
        blocks = cld(blocks, nelem)
        (threads=threads, blocks=blocks, elements_per_thread=nelem)
    end
end

