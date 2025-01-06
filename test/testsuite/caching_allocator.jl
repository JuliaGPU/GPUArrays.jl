@testsuite "Caching Allocator" (AT, eltypes) -> begin
    device = GPUArrays.AllocCache.device(AT)

    @testset "free_immediately=false" begin
        pdcache = GPUArrays.AllocCache.cache_allocator(AT)
        pdcache.free_immediately = false
        named_cache = GPUArrays.AllocCache.named_cache_allocator!(pdcache, device, :cache)

        T = Float32
        dims = (1, 2, 3)
        key = hash((T, dims))

        GPUArrays.AllocCache.@enable AT :cache begin
            x1 = AT(zeros(T, dims))
        end
        @test sizeof(pdcache, device, :cache) == sizeof(Float32) * prod(dims)
        @test length(named_cache.free[key]) == 1
        @test length(named_cache.busy[key]) == 0
        @test x1 === named_cache.free[key][1]

        # Second allocation does not allocate - cache stays the same in size.

        GPUArrays.AllocCache.@enable AT :cache begin
            x2 = AT(zeros(T, dims))

            # Does not go to cache.
            GPUArrays.AllocCache.@disable begin
                x_free = AT(zeros(T, dims))
            end
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
        GPUArrays.AllocCache.@enable AT :cache begin
            x3 = AT(zeros(T2, dims))
        end
        @test sizeof(pdcache, device, :cache) == (sizeof(Float32) + sizeof(Int32)) * prod(dims)
        @test length(named_cache.free[key]) == 1
        @test length(named_cache.free[key2]) == 1
        @test x3 === named_cache.free[key2][1]

        # Freeing all memory held by cache.

        GPUArrays.AllocCache.invalidate!(AT, :cache)
        @test sizeof(pdcache, device, :cache) == 0
    end

    @testset "free_immediately=true" begin
        pdcache = GPUArrays.AllocCache.cache_allocator(AT)
        pdcache.free_immediately = true
        named_cache = GPUArrays.AllocCache.named_cache_allocator!(pdcache, device, :cache2)

        T = Float32
        dims = (1, 2, 3)
        key = hash((T, dims))

        @test sizeof(pdcache, device, :cache2) == 0

        GPUArrays.AllocCache.@enable AT :cache2 begin
            x1 = AT(zeros(T, dims))

            @test !haskey(named_cache.free, key)
            @test length(named_cache.busy[key]) == 1
            @test sizeof(pdcache, device, :cache2) == sizeof(Float32) * prod(dims)
        end

        # `free` was never even used with `free_immediately=true`.
        @test !haskey(named_cache.free, key)
        @test length(named_cache.busy[key]) == 0
        @test sizeof(pdcache, device, :cache2) == 0
    end
end
