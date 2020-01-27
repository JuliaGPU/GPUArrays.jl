# kernel execution

export AbstractGPUBackend, AbstractKernelContext, gpu_call, synchronize, thread_blocks_heuristic

abstract type AbstractGPUBackend end

abstract type AbstractKernelContext end

"""
    backend(T::Type{<:AbstractArray})

Gets the GPUArrays back-end responsible for managing arrays of type `T`.
"""
backend(::Type{<:AbstractArray}) = error("This array is not a GPU array") # COV_EXCL_LINE

"""
    gpu_call(kernel::Function, A::AbstractGPUArray, args...; kwargs...)

Calls function `kernel` on the GPU device that backs array `A`, passing along arguments
`args`. The keyword arguments `kwargs` are not passed along, but are interpreted on the host
to influence how the kernel is executed. The following keyword arguments are supported:

- `total_threads::Int`: how many threads should be launched _in total_. The actual number of
   threads and blocks is determined using a heuristic. Defaults to the length of `A` if no
   other keyword arguments that influence the launch configuration are specified.
- `threads::Int` and `blocks::Int`: configure exactly how many threads and blocks are
   launched. This cannot be used in combination with the `total_threads` argument.
"""
function gpu_call(kernel::Base.Callable, A::AbstractArray, args...;
                  total_threads::Union{Int,Nothing}=nothing,
                  threads::Union{Int,Nothing}=nothing,
                  blocks::Union{Int,Nothing}=nothing,
                  kwargs...)
    # determine how many threads/blocks to launch
    if total_threads===nothing && threads===nothing && blocks===nothing
        total_threads = length(A)
    end
    if total_threads !== nothing
        if threads !== nothing || blocks !== nothing
            error("Cannot specify both total_threads and threads/blocks configuration")
        end
        threads, blocks = thread_blocks_heuristic(total_threads)
    else
        if threads === nothing
            threads = 1
        end
        if blocks === nothing
            blocks = 1
        end
    end

    gpu_call(backend(typeof(A)), kernel, args...; threads=threads, blocks=blocks, kwargs...)
end

gpu_call(backend::AbstractGPUBackend, kernel, args...; kwargs...) = error("Not implemented") # COV_EXCL_LINE

"""
    synchronize(A::AbstractArray)

Blocks until all operations are finished on `A`
"""
function synchronize(A::AbstractArray)
    # fallback is a noop, for backends not needing synchronization. This
    # makes it easier to write generic code that also works for AbstractArrays
end

function thread_blocks_heuristic(len::Integer)
    # TODO better threads default
    threads = clamp(len, 1, 256)
    blocks = max(ceil(Int, len / threads), 1)
    (blocks, threads)
end
