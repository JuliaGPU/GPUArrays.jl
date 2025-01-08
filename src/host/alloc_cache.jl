using ..GPUArrays

@static if VERSION < v"1.11"
    using ScopedValues
else
    using Base.ScopedValues
end

mutable struct AllocCache{T <: AbstractGPUArray}
    lock::ReentrantLock
    busy::Dict{UInt64, Vector{T}} # hash(key) => GPUArray[]
    free::Dict{UInt64, Vector{T}}

    function AllocCache(::Type{T}) where {T <: AbstractGPUArray}
        cache = new{T}(
            ReentrantLock(),
            Dict{UInt64, Vector{T}}(),
            Dict{UInt64, Vector{T}}()
        )
        return finalizer(unsafe_free!, cache)
    end
end

function get_pool!(cache::AllocCache{T}, pool::Symbol, uid::UInt64) where {T <: AbstractGPUArray}
    pool = getproperty(cache, pool)
    uid_pool = get(pool, uid, nothing)
    if uid_pool ≡ nothing
        uid_pool = Base.@lock cache.lock pool[uid] = T[]
    end
    return uid_pool
end

function alloc!(alloc_f, cache::AllocCache, key)
    x = nothing
    uid = hash(key)

    busy_pool = get_pool!(cache, :busy, uid)
    free_pool = get_pool!(cache, :free, uid)
    isempty(free_pool) && (x = alloc_f())

    while !isempty(free_pool) && x ≡ nothing
        tmp = Base.@lock cache.lock pop!(free_pool)
        # Array was manually freed via `unsafe_free!`.
        GPUArrays.storage(tmp).freed && continue
        x = tmp
    end

    x ≡ nothing && (x = alloc_f())
    Base.@lock cache.lock push!(busy_pool, x)
    return x
end

function free_busy!(cache::AllocCache)
    for uid in cache.busy.keys
        busy_pool = get_pool!(cache, :busy, uid)
        isempty(busy_pool) && continue

        Base.@lock cache.lock begin
            free_pool = get_pool!(cache, :free, uid)
            append!(free_pool, busy_pool)
            empty!(busy_pool)
        end
    end
    return
end

function unsafe_free!(cache::AllocCache)
    Base.@lock cache.lock begin
        for (_, pool) in cache.busy
            isempty(pool) || error(
                "Invalidating allocations cache that's currently in use. " *
                    "Invalidating inside `@enable` is not allowed."
            )
        end
        for (_, pool) in cache.free
            map(unsafe_free!, pool)
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

const ALLOC_CACHE = ScopedValue{Union{Nothing, AllocCache}}(nothing)

"""
    @enable(cache, expr)

Evaluate expression `expr` using allocations cache `cache`.

When gpu allocation is requested during execution of `expr`,
it will first check if there's "free" cache instead of performing an actual allocation.
If no "free" allocation exists, an actual allocation is performed.
Before returning allocation to the user, it is marked as busy and
will not be used by allocation in the scope defined by `@enable`.

**After** the execution of `expr` all "busy" allocations are marked as "free"
thus they can be re-used next time the program enters this scope.

This is useful to apply in a repeating block of code to avoid relying on
GC to free gpu memory in time.

# Example

In the following example, each iteration of the for-loop requires `8 GiB` of gpu memory.
Without caching allocator GC wouldn't be able to free arrays in time
resulting in higher memory usage.
With caching allocator, memory usage stays at exactly `8 GiB`.

```julia
cache = GPUArrays.AllocCache(CuArray)
n = 1024^3
for i in 1:1000
    GPUArrays.@enable cache begin
        sin.(CUDA.rand(Float32, n))
    end
end
# To free immediately.
# Otherwise, it will be freed when collected by GC.
GPUArrays.unsafe_free!(cache)
```

See [`@disable`](@ref).
"""
macro enable(cache, expr)
    return quote
        res = @with $(esc(ALLOC_CACHE)) => $(esc(cache)) $(esc(expr))
        free_busy!($(esc(cache)))
        res
    end
end

"""
    disable(expr)

Evaluate expression `expr` without using allocations cache.
This is useful to call from within `@enable` to avoid caching some allocations.
"""
macro disable(expr)
    return quote
        @with $(esc(ALLOC_CACHE)) => nothing $(esc(expr))
    end
end
