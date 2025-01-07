@testsuite "Caching Allocator" (AT, eltypes) -> begin
    if AT <: AbstractGPUArray
        device = GPUArrays.AllocCache.device(AT)
        pdcache = GPUArrays.AllocCache.cache_allocator(AT)
        named_cache = GPUArrays.AllocCache.named_cache_allocator!(pdcache, device, :cache)

        T = Float32
        dims = (1, 2, 3)

        x1 = GPUArrays.AllocCache.@enable AT :cache begin
            AT(zeros(T, dims))
        end
        @test sizeof(pdcache, device, :cache) == sizeof(Float32) * prod(dims)
        @test length(named_cache.free) == 1

        key = first(keys(named_cache.free))
        @test length(named_cache.free[key]) == 1
        @test length(named_cache.busy[key]) == 0
        @test x1 === named_cache.free[key][1]

        # Second allocation does not allocate - cache stays the same in size.

        x2, x_free = GPUArrays.AllocCache.@enable AT :cache begin
            x2 = AT(zeros(T, dims))

            # Does not go to cache.
            GPUArrays.AllocCache.@disable begin
                x_free = AT(zeros(T, dims))
            end
            x2, x_free
        end
        @test sizeof(pdcache, device, :cache) == sizeof(Float32) * prod(dims)
        @test length(named_cache.free[key]) == 1
        @test length(named_cache.busy[key]) == 0
        @test x2 === x1
        @test x2 === named_cache.free[key][1]
        @test x_free !== x2

        # Third allocation of different type - cache grows.

        T2 = Int32
        key2 = hash((T2, dims))
        x3 = GPUArrays.AllocCache.@enable AT :cache begin
            AT(zeros(T2, dims))
        end
        @test sizeof(pdcache, device, :cache) == (sizeof(Float32) + sizeof(Int32)) * prod(dims)

        _keys = collect(keys(named_cache.free))
        key2 = _keys[findfirst(i -> i != key, _keys)]
        @test length(named_cache.free[key]) == 1
        @test length(named_cache.free[key2]) == 1
        @test x3 === named_cache.free[key2][1]

        # Freeing all memory held by cache.

        GPUArrays.AllocCache.invalidate!(AT, :cache)
        @test sizeof(pdcache, device, :cache) == 0
    end
end
