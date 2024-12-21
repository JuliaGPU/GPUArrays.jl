@static if VERSION < v"1.11"
    using ScopedValues
else
    using Base.ScopedValues
end

const CacheAllocatorName = ScopedValue(:none)

struct CacheAllocator{T <: AbstractGPUArray}
    lock::ReentrantLock
    busy::Dict{UInt64, Vector{T}} # hash((T, dims)) => GPUArray[]
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
    alloc!(alloc_f, cache::CacheAllocator, ::Type{T}, dims::Dims{N}; skip_free::Bool) where {T, N}

Attempt to retrieve cached allocation from `cache` using eltype `T` and `dims`
as keys for searching.
If no such allocation is found, execute `alloc_f` that does actual allocation,
store it in cache for future use and return it.

`skip_free::Bool` is used together with `PerDeviceCacheAllocator.free_immediately`.
When `true` arrays are bulk-freed instead of stored in cache.
In this case `alloc!` will avoid looking into "free" part of `cache`
and execute `alloc_f` immediately, storing allocation for future bulk-freeing.
"""
function alloc!(alloc_f, cache::CacheAllocator, ::Type{T}, dims::Dims{N}; skip_free::Bool) where {T, N}
    x = nothing
    uid = hash((T, dims))
    busy_pool = get_pool!(cache, :busy, uid)

    if skip_free
        x = alloc_f()
    else
        free_pool = get_pool!(cache, :free, uid)
        isempty(free_pool) && (x = alloc_f())

        while !isempty(free_pool) && x ≡ nothing
            tmp = Base.@lock cache.lock pop!(free_pool)
            # Array was manually freed via `unsafe_free!`.
            storage(tmp).freed && continue
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

function alloc!(alloc_f, kab::Backend, name::Symbol, ::Type{T}, dims::Dims{N}) where {T, N}
    pdcache = cache_allocator(kab)
    cache = named_cache_allocator!(pdcache, device(kab), name)
    alloc!(alloc_f, cache, T, dims; skip_free=pdcache.free_immediately)
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
    invalidate_cache_allocator!(kab::Backend, name::Symbol)

Free all memory held by `name`d cached allocator given KernelAbstractions `backend`.
"""
invalidate_cache_allocator!(kab::Backend, name::Symbol) =
    invalidate_cache_allocator!(cache_allocator(kab), device(kab), name)

function invalidate_cache_allocator!(pdcache::PerDeviceCacheAllocator, device, name::Symbol)
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

function free_busy!(kab::Backend, name::Symbol)
    pdcache = cache_allocator(kab)
    free_busy!(named_cache_allocator!(pdcache, device(kab), name); pdcache.free_immediately)
end

"""
    @cache_scope backend name expr

Evaluate expression `expr` using `name`d caching allocator
for the given KernelAbstractions `backend`.

When gpu allocation is requested during execution of `expr`,
allocator will try to use its "free" cache instead of doing an actual allocation.
If no "free" allocation exists, an actual allocation is performed.
Before returning allocation to the user, it is marked as busy and
will not be used by allocation in the scope defined by `@cache_scope`.

**After** the execution of `expr` all "busy" allocations are marked as "free"
thus they can be re-used next time the program enters this scope.

This is useful to apply in a repeating block of code to avoid relying on
GC to free gpu memory in time.

`name` is a `Symbol` that defines which allocator to use
(`:none` is reserved and means no allocator).

# Example

In the following example, each iteration of the for-loop requires `2 GiB`
of gpu memory.
Without caching allocator GC wouldn't be able to free arrays in time
resulting in higher memory usage.
With caching allocator, memory usage stays at exactly `2 GiB`.

See [`@no_cache_scope`](@ref), [`invalidate_cache_allocator!`](@ref).
```julia
kab = CUDABackend()
n = 1024^3
for i in 1:1000
    @cache_scope kab :loop begin
        sin.(CUDA.rand(Float32, n))
    end
end
invalidate_cache_allocator!(kab, :loop)
```
"""
macro cache_scope(backend, name, expr)
    quote
        res = @with $(esc(CacheAllocatorName)) => $(esc(name)) $(esc(expr))
        free_busy!($(esc(backend)), $(esc(name)))
        res
    end
end

"""
    @no_cache_scope expr

Evaluate expression `expr` without using caching allocator.
This is useful to call from within `@cache_scope` to avoid caching arrays.
"""
macro no_cache_scope(expr)
    quote
        @with $(esc(CacheAllocatorName)) => :none $(esc(expr))
    end
end

# Interface API.

"""
    cache_allocator(::Backend)

Given KernelAbstractions `backend`, return corresponding `PerDeviceCacheAllocator` for it.
Each GPU backend must implement this.
"""
cache_allocator(::Backend) = error("Not implemented.")

"""
    device(::Backend)

Given KernelAbstractions `backend`, return current device.
Each GPU backend must implement this.
"""
device(::Backend) = error("Not implemented.")
