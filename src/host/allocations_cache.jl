module AllocCache

using ..GPUArrays
using KernelAbstractions

@static if VERSION < v"1.11"
    using ScopedValues
else
    using Base.ScopedValues
end

const CacheAllocatorName = ScopedValue{Union{Nothing, Symbol}}(nothing)

struct CacheAllocator{T <: AbstractGPUArray}
    lock::ReentrantLock
    busy::Dict{UInt64, Vector{T}} # hash(key) => GPUArray[]
    free::Dict{UInt64, Vector{T}}
end

CacheAllocator(::Type{T}) where T = CacheAllocator(
    ReentrantLock(),
    Dict{UInt64, Vector{T}}(),
    Dict{UInt64, Vector{T}}(),
)

function get_pool!(cache::CacheAllocator{T}, pool::Symbol, uid::UInt64) where T
    pool = getproperty(cache, pool)
    uid_pool = get(pool, uid, nothing)
    if uid_pool ≡ nothing
        uid_pool = Base.@lock cache.lock pool[uid] = T[]
    end
    return uid_pool
end

"""
    alloc!(alloc_f, cache::CacheAllocator, key; skip_free::Bool)

Attempt to retrieve cached allocation from `cache` using `key` for searching.
If no such allocation is found, execute `alloc_f` that does actual allocation,
store it in cache for future use and return it.

`skip_free::Bool` is used together with `PerDeviceCacheAllocator.free_immediately`.
When `true` arrays are bulk-freed instead of stored in cache.
In this case `alloc!` will avoid looking into "free" part of `cache`
and execute `alloc_f` immediately, storing allocation for future bulk-freeing.
"""
function alloc!(alloc_f, cache::CacheAllocator, key; skip_free::Bool)
    x = nothing
    uid = hash(key)
    busy_pool = get_pool!(cache, :busy, uid)

    if skip_free
        x = alloc_f()
    else
        free_pool = get_pool!(cache, :free, uid)
        isempty(free_pool) && (x = alloc_f())

        while !isempty(free_pool) && x ≡ nothing
            tmp = Base.@lock cache.lock pop!(free_pool)
            # Array was manually freed via `unsafe_free!`.
            GPUArrays.storage(tmp).freed && continue
            x = tmp
        end
    end

    x ≡ nothing && (x = alloc_f())
    Base.@lock cache.lock push!(busy_pool, x)
    return x
end

function free_busy!(cache::CacheAllocator; free_immediately::Bool)
    for uid in cache.busy.keys
        busy_pool = get_pool!(cache, :busy, uid)
        isempty(busy_pool) && continue

        Base.@lock cache.lock begin
            if free_immediately
                map(unsafe_free!, busy_pool)
            else
                free_pool = get_pool!(cache, :free, uid)
                append!(free_pool, busy_pool)
            end
            empty!(busy_pool)
        end
    end
end

mutable struct PerDeviceCacheAllocator{T <: AbstractGPUArray}
    lock::ReentrantLock
    caches::Dict{UInt64, Dict{Symbol, CacheAllocator{T}}}
    free_immediately::Bool
end

PerDeviceCacheAllocator(::Type{T}; free_immediately::Bool) where T <: AbstractGPUArray =
    PerDeviceCacheAllocator(ReentrantLock(), Dict{UInt64, Dict{Symbol, CacheAllocator{T}}}(), free_immediately)

function named_cache_allocator!(pdcache::PerDeviceCacheAllocator{T}, device, name::Symbol) where T
    h = hash(device)
    dev_cache = get(pdcache.caches, h, nothing)
    if dev_cache ≡ nothing
        Base.@lock pdcache.lock begin
            named_cache = CacheAllocator(T)
            pdcache.caches[h] = Dict{Symbol, CacheAllocator{T}}(name => named_cache)
            return named_cache
        end
    end

    named_cache = get(dev_cache, name, nothing)
    if named_cache ≡ nothing
        named_cache = CacheAllocator(T)
        Base.@lock pdcache.lock dev_cache[name] = named_cache
    end
    return named_cache
end

function alloc!(alloc_f, AT::Type{<: AbstractGPUArray}, name::Symbol, key)
    pdcache = cache_allocator(AT)
    cache = named_cache_allocator!(pdcache, device(AT), name)
    alloc!(alloc_f, cache, key; skip_free=pdcache.free_immediately)
end

function Base.sizeof(pdcache::PerDeviceCacheAllocator, device, name::Symbol)
    sz = UInt64(0)
    h = hash(device)

    dev_cache = get(pdcache.caches, h, nothing)
    dev_cache ≡ nothing && return sz

    named_cache = get(dev_cache, name, nothing)
    named_cache ≡ nothing && return sz

    Base.@lock named_cache.lock begin
        for (_, pool) in named_cache.free
            sz += sum(sizeof, pool; init=UInt64(0))
        end
        for (_, pool) in named_cache.busy
            sz += sum(sizeof, pool; init=UInt64(0))
        end
    end
    return sz
end

"""
    invalidate!(AT::Type{AbstractGPUArray}, name::Symbol)

Free all memory held by `name`d cached allocator given array type `AT`.
"""
invalidate!(AT::Type{<: AbstractGPUArray}, name::Symbol) =
    invalidate!(cache_allocator(AT), device(AT), name)

function invalidate!(pdcache::PerDeviceCacheAllocator, device, name::Symbol)
    h = hash(device)
    dev_cache = get(pdcache.caches, h, nothing)
    dev_cache ≡ nothing && return

    named_cache = get(dev_cache, name, nothing)
    named_cache ≡ nothing && return

    Base.@lock named_cache.lock begin
        for (_, pool) in named_cache.free
            map(unsafe_free!, pool)
        end
        # TODO error when trying to invalidate busy cache?
        for (_, pool) in named_cache.busy
            map(unsafe_free!, pool)
        end
        empty!(named_cache.busy)
        empty!(named_cache.free)
    end
    return
end

function free_busy!(AT::Type{<: AbstractGPUArray}, name::Symbol)
    pdcache = cache_allocator(AT)
    free_busy!(named_cache_allocator!(pdcache, device(AT), name); pdcache.free_immediately)
end

"""
    @enable AT name expr

Evaluate expression `expr` using `name`d caching allocator for the given array type `AT`.

When gpu allocation is requested during execution of `expr`,
allocator will try to use its "free" cache instead of doing an actual allocation.
If no "free" allocation exists, an actual allocation is performed.
Before returning allocation to the user, it is marked as busy and
will not be used by allocation in the scope defined by `@enable`.

**After** the execution of `expr` all "busy" allocations are marked as "free"
thus they can be re-used next time the program enters this scope.

This is useful to apply in a repeating block of code to avoid relying on
GC to free gpu memory in time.

`name` is a `Symbol` that defines which allocator to use
(`nothing`, which is a default, disables it).

# Example

In the following example, each iteration of the for-loop requires `2 GiB` of gpu memory.
Without caching allocator GC wouldn't be able to free arrays in time
resulting in higher memory usage.
With caching allocator, memory usage stays at exactly `2 GiB`.

See [`@disable`](@ref), [`invalidate!`](@ref).
```julia
n = 1024^3
for i in 1:1000
    CUDA.AllocCache.@enable CuArray :loop begin
        sin.(CUDA.rand(Float32, n))
    end
end
CUDA.AllocCache.invalidate!(CuArray, :loop)
```
"""
macro enable(AT, name, expr)
    quote
        res = @with $(esc(CacheAllocatorName)) => $(esc(name)) $(esc(expr))
        free_busy!($(esc(AT)), $(esc(name)))
        res
    end
end

"""
    @disable expr

Evaluate expression `expr` without using caching allocator.
This is useful to call from within `@enable` to avoid caching arrays.
"""
macro disable(expr)
    quote
        @with $(esc(CacheAllocatorName)) => nothing $(esc(expr))
    end
end

# Interface API.

"""
    cache_allocator(::Type{AbstractGPUArray})

Given array type, return corresponding `PerDeviceCacheAllocator` for it.
Each GPU backend must implement this.
"""
cache_allocator(::Type{AbstractGPUArray}) = error("Not implemented.")

"""
    device(::Type{AbstractGPUArray})

Given array type, return current device.
Each GPU backend must implement this.
"""
device(::Type{AbstractGPUArray}) = error("Not implemented.")

end
