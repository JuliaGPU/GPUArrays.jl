# kernel execution

export AbstractGPUBackend, AbstractKernelContext, gpu_call, synchronize, thread_blocks_heuristic

abstract type AbstractGPUBackend end

abstract type AbstractKernelContext end

backend(::Type{T}) where T = error("Can't choose GPU backend for $T")

"""
    gpu_call(kernel::Function, A::AbstractGPUArray, args::Tuple, configuration = length(A))

Calls function `kernel` on the GPU.
`A` must be an AbstractGPUArray and will help to dispatch to the correct GPU backend
and supplies queues and contexts.
Calls the kernel function with `kernel(ctx, args...)`, where ctx is dependant on the backend
and can be used for getting an index into `A` with `linear_index(ctx)`.
Optionally, a launch configuration can be supplied in the following way:

    1) A single integer, indicating how many work items (total number of threads) you want to launch.
        in this case `linear_index(ctx)` will be a number in the range `1:configuration`
    2) Pass a tuple of integer tuples to define blocks and threads per blocks!

"""
function gpu_call(kernel, A::AbstractArray, args::Tuple, configuration = length(A))
    ITuple = NTuple{N, Integer} where N
    # If is a single integer, we assume it to be the global size / total number of threads one wants to launch
    thread_blocks = if isa(configuration, Integer)
        thread_blocks_heuristic(configuration)
    elseif isa(configuration, ITuple)
        @assert length(configuration) == 1
        configuration[1], 1
    elseif isa(configuration, Tuple{ITuple, ITuple})
        @assert length(configuration[1]) == 1
        @assert length(configuration[2]) == 1
        configuration[1][1], configuration[2][1]
    else
        error("""Please launch a gpu kernel with a valid configuration.
            Found: $configurations
            Configuration needs to be:
            1) A single integer, indicating how many work items (total number of threads) you want to launch.
                in this case `linear_index(ctx)` will be a number in the range 1:configuration
            2) Pass a tuple of integer tuples to define blocks and threads per blocks!
                `linear_index` will be inbetween 1:prod((blocks..., threads...))
        """)
    end
    _gpu_call(backend(typeof(A)), kernel, A, args, thread_blocks)
end

# Internal GPU call function, that needs to be overloaded by the backends.
_gpu_call(::Any, f, A, args, thread_blocks) = error("Not implemented") # COV_EXCL_LINE

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
