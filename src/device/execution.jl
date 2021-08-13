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
- `elements::Int`: how many elements will be processed by this kernel. In most
  circumstances, this will correspond to the total number of threads that needs to be
  launched, unless the kernel supports a variable number of elements to process per
  iteration. Defaults to the length of `arg0` if no other keyword arguments that influence
  the launch configuration are specified.
- `threads::Int` and `blocks::Int`: configure exactly how many threads and blocks are
  launched. This cannot be used in combination with the `elements` argument.
- `name::String`: inform the back end about the name of the kernel to be executed. This can
  be used to emit better diagnostics, and is useful with anonymous kernels.
"""
function gpu_call(kernel::F, args::Vararg{Any,N};
                  target::AbstractArray=first(args),
                  elements::Union{Int,Nothing}=nothing,
                  threads::Union{Int,Nothing}=nothing,
                  blocks::Union{Int,Nothing}=nothing,
                  name::Union{String,Nothing}=nothing) where {F,N}
    # non-trivial default values for launch configuration
    if elements===nothing && threads===nothing && blocks===nothing
        elements = length(target)
    elseif elements===nothing
        if threads === nothing
            threads = 1
        end
        if blocks === nothing
            blocks = 1
        end
    elseif threads!==nothing || blocks!==nothing
        error("Cannot specify both elements and threads/blocks configuration")
    end

    # the number of elements to process needs to be passed to the kernel somehow, so there's
    # no easy way to do this without passing additional arguments or changing the context.
    # both are expensive, so require manual use of `launch_heuristic` for those kernels.
    elements_per_thread = 1

    if elements !== nothing
        @assert elements > 0
        heuristic = launch_heuristic(backend(target), kernel, args...;
                                     elements, elements_per_thread)
        config = launch_configuration(backend(target), heuristic;
                                      elements, elements_per_thread)
        gpu_call(backend(target), kernel, args, config.threads, config.blocks; name=name)
    else
        @assert threads > 0
        @assert blocks > 0
        gpu_call(backend(target), kernel, args, threads, blocks; name=name)
    end
end

# how many threads and blocks `kernel` needs to be launched with, passing arguments `args`,
# to fully saturate the GPU. `elements` indicates the number of elements that needs to be
# processed, while `elements_per_threads` indicates the number of elements this kernel can
# process (i.e. if it's a grid-stride kernel, or 1 if otherwise).
#
# this heuristic should be specialized for the back-end, ideally using an API for maximizing
# the occupancy of the launch configuration (like CUDA's occupancy API).
function launch_heuristic(backend::AbstractGPUBackend, kernel, args...;
                          elements::Int, elements_per_thread::Int)
    return (threads=256, blocks=32)
end

# determine how many threads and blocks to actually launch given upper limits.
# returns a tuple of blocks, threads, and elements_per_thread (which is always 1
# unless specified that the kernel can handle a number of elements per thread)
function launch_configuration(backend::AbstractGPUBackend, heuristic;
                              elements::Int, elements_per_thread::Int)
    threads = clamp(elements, 1, heuristic.threads)
    blocks = max(cld(elements, threads), 1)

    if elements_per_thread > 1 && blocks > heuristic.blocks
        # we want to launch more blocks than required, so prefer a grid-stride loop instead
        ## try to stick to the number of blocks that the heuristic suggested
        blocks = heuristic.blocks
        nelem = cld(elements, blocks*threads)
        ## only bump the number of blocks if we really need to
        if nelem > elements_per_thread
            nelem = elements_per_thread
            blocks = cld(elements, nelem*threads)
        end
        (; threads, blocks, elements_per_thread=nelem)
    else
        (; threads, blocks, elements_per_thread=1)
    end
end

gpu_call(backend::AbstractGPUBackend, kernel, args, threads::Int, blocks::Int; kwargs...) =
    error("Not implemented") # COV_EXCL_LINE
