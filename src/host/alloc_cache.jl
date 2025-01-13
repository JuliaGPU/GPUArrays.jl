using ..GPUArrays

@static if VERSION < v"1.11"
    using ScopedValues
else
    using Base.ScopedValues
end

mutable struct AllocCache
    lock::ReentrantLock
    busy::Dict{UInt64, Vector{DataRef}}
    free::Dict{UInt64, Vector{DataRef}}

    function AllocCache()
        cache = new(
            ReentrantLock(),
            Dict{UInt64, Vector{Any}}(),
            Dict{UInt64, Vector{Any}}()
        )
        return finalizer(unsafe_free!, cache)
    end
end

function get_pool!(cache::AllocCache, pool::Symbol, uid::UInt64)
    pool = getproperty(cache, pool)
    uid_pool = get(pool, uid, nothing)
    if uid_pool === nothing
        uid_pool = pool[uid] = DataRef[]
    end
    return uid_pool
end

function cached_alloc(f, key)
    cache = ALLOC_CACHE[]
    if cache === nothing
        return f()::DataRef
    end

    ref = nothing
    uid = hash(key)

    Base.@lock cache.lock begin
        free_pool = get_pool!(cache, :free, uid)

        if !isempty(free_pool)
            ref = Base.@lock cache.lock pop!(free_pool)
        end
    end

    if ref === nothing
        ref = f()::DataRef
        ref.cached = true
    end

    Base.@lock cache.lock begin
        busy_pool = get_pool!(cache, :busy, uid)
        push!(busy_pool, ref)
    end

    return ref
end

function free_busy!(cache::AllocCache)
    Base.@lock cache.lock begin
        for uid in keys(cache.busy)
            busy_pool = get_pool!(cache, :busy, uid)
            isempty(busy_pool) && continue

            free_pool = get_pool!(cache, :free, uid)
            append!(free_pool, busy_pool)
            empty!(busy_pool)
        end
    end
    return
end

function unsafe_free!(cache::AllocCache)
    Base.@lock cache.lock begin
        for pool in values(cache.busy)
            isempty(pool) || error("Cannot invalidate a cache that's in active use")
        end
        for pool in values(cache.free), ref in pool
            # release the reference
            ref.cached = false
            unsafe_free!(ref)
        end
        empty!(cache.free)
    end
    return
end

function Base.sizeof(cache::AllocCache)
    sz = UInt64(0)
    Base.@lock cache.lock begin
        for kind in (cache.free, cache.busy), (_, pool) in kind
            sz += sum(sizeof, pool; init = UInt64(0))
        end
    end
    return sz
end

function Base.show(io::IO, cache::AllocCache)
    sz, n_free, n_busy = Base.@lock cache.lock begin
        sz = sizeof(cache)
        n_free = sum(p -> length(p[2]), cache.free; init = 0)
        n_busy = sum(p -> length(p[2]), cache.busy; init = 0)
        sz, n_free, n_busy
    end
    return print(io, "AllocCache(n_free=$n_free, n_busy=$n_busy, sizeof=$(Base.format_bytes(sz)))")
end

const ALLOC_CACHE = ScopedValue{Union{Nothing, AllocCache}}(nothing)

"""
    @cached cache expr

Evaluate `expr` using allocations cache `cache`.

When GPU memory is allocated during the execution of `expr`, `cache` will first be checked.
If no memory is available in the cache, a new allocation will be requested.

After the execution of `expr`, all allocations made under the scope of `@cached` will be
cached within `cache` for future use. This is useful to avoid relying on GC to free GPU
memory in time.

Once `cache` goes out scope, or when the user calls `unsafe_free!` on it, all cached
allocations will be freed.

# Example

In the following example, each iteration of the for-loop requires 8 GiB of GPU memory.
Without caching those allocations, significant pressure would be put on the GC, resulting
in high memory usage and latency. By using the allocator cache, the memory usage is stable:

```julia
cache = GPUArrays.AllocCache()
for i in 1:1000
    GPUArrays.@cached cache begin
        sin.(CUDA.rand(Float32, 1024^3))
    end
end

# optionally: free the memory now, instead of waiting for the GC to collect `cache`
GPUArrays.unsafe_free!(cache)
```

See [`@uncached`](@ref).
"""
macro cached(cache, expr)
    return quote
        cache = $(esc(cache))
        GC.@preserve cache begin
            res = @with $(esc(ALLOC_CACHE)) => cache $(esc(expr))
            free_busy!(cache)
            res
        end
    end
end

"""
    @uncached expr

Evaluate expression `expr` without using the allocation. This is useful to call from within
`@cached` to avoid caching some allocations, e.g., because they can be returned out of the
`@cached` scope.
"""
macro uncached(expr)
    return quote
        @with $(esc(ALLOC_CACHE)) => nothing $(esc(expr))
    end
end
