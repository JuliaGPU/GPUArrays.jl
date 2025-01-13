@testsuite "alloc cache" (AT, eltypes) -> begin
    if AT <: AbstractGPUArray
        cache = GPUArrays.AllocCache()

        # first allocation populates the cache
        T, dims = Float32, (1, 2, 3)
        GPUArrays.@cached cache begin
            cached1 = AT(zeros(T, dims))
        end
        @test sizeof(cache) == sizeof(cached1)
        key = first(keys(cache.free))
        @test length(cache.free[key]) == 1
        @test length(cache.busy[key]) == 0
        @test cache.free[key][1] === GPUArrays.storage(cached1)

        # second allocation hits the cache
        GPUArrays.@cached cache begin
            cached2 = AT(zeros(T, dims))

            # explicitly uncached ones don't
            GPUArrays.@uncached uncached = AT(zeros(T, dims))
        end
        @test sizeof(cache) == sizeof(cached2)
        key = first(keys(cache.free))
        @test length(cache.free[key]) == 1
        @test length(cache.busy[key]) == 0
        @test cache.free[key][1] === GPUArrays.storage(cached2)
        @test uncached !== cached2

        # compatible shapes should also hit the cache
        dims = (3, 2, 1)
        GPUArrays.@cached cache begin
            cached3 = AT(zeros(T, dims))
        end
        @test sizeof(cache) == sizeof(cached3)
        key = first(keys(cache.free))
        @test length(cache.free[key]) == 1
        @test length(cache.busy[key]) == 0
        @test cache.free[key][1] === GPUArrays.storage(cached3)

        # as should compatible eltypes
        T = Int32
        GPUArrays.@cached cache begin
            cached4 = AT(zeros(T, dims))
        end
        @test sizeof(cache) == sizeof(cached4)
        key = first(keys(cache.free))
        @test length(cache.free[key]) == 1
        @test length(cache.busy[key]) == 0
        @test cache.free[key][1] === GPUArrays.storage(cached4)

        # different shapes should trigger a new allocation
        dims = (2, 2)
        GPUArrays.@cached cache begin
            cached5 = AT(zeros(T, dims))

            # we're allowed to early free arrays, which should be a no-op for cached data
            GPUArrays.unsafe_free!(cached5)
        end
        @test sizeof(cache) == sizeof(cached4) + sizeof(cached5)
        _keys = collect(keys(cache.free))
        key2 = _keys[findfirst(i -> i != key, _keys)]
        @test length(cache.free[key]) == 1
        @test length(cache.free[key2]) == 1
        @test cache.free[key2][1] === GPUArrays.storage(cached5)

        # we should be able to re-use the early-freed
        GPUArrays.@cached cache begin
            cached5 = AT(zeros(T, dims))
        end

        # freeing all memory held by cache should free all allocations
        @test !GPUArrays.storage(cached1).freed
        @test GPUArrays.storage(cached1).cached
        @test !GPUArrays.storage(cached5).freed
        @test GPUArrays.storage(cached5).cached
        @test !GPUArrays.storage(uncached).freed
        @test !GPUArrays.storage(uncached).cached
        GPUArrays.unsafe_free!(cache)
        @test sizeof(cache) == 0
        @test GPUArrays.storage(cached1).freed
        @test !GPUArrays.storage(cached1).cached
        @test GPUArrays.storage(cached5).freed
        @test !GPUArrays.storage(cached5).cached
        @test !GPUArrays.storage(uncached).freed
        ## test that the underlying data was freed as well
        @test GPUArrays.storage(cached1).rc.count[] == 0
        @test GPUArrays.storage(cached5).rc.count[] == 0
        @test GPUArrays.storage(uncached).rc.count[] == 1
    end
end
