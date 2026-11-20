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

        # arrays derived from cached ones share the cache's ownership of their memory,
        # irrespective of the cache scope or task that queries it
        @test GPUArrays.is_cached_storage(GPUArrays.storage(cached2))
        @test !GPUArrays.is_cached_storage(GPUArrays.storage(uncached))
        cached2_view = view(cached2, :, :, 2:3)
        cached2_reshape = reshape(cached2, :)
        cached2_reinterpret = reinterpret(Int32, cached2)
        for derived in (cached2_view, cached2_reshape, cached2_reinterpret)
            @test !GPUArrays.storage(derived).cached
            @test GPUArrays.is_cached_storage(GPUArrays.storage(derived))
        end
        inner = GPUArrays.AllocCache()
        GPUArrays.@cached inner begin
            inner_array = AT(zeros(T, dims))
            @test GPUArrays.is_cached_storage(GPUArrays.storage(cached2_view))
            GPUArrays.@uncached begin
                @test GPUArrays.is_cached_storage(GPUArrays.storage(inner_array))
                @test fetch(Threads.@spawn GPUArrays.is_cached_storage(GPUArrays.storage(cached2_view)))
            end
        end
        GPUArrays.unsafe_free!(inner)
        @test !GPUArrays.is_cached_storage(GPUArrays.storage(inner_array))
        @test GPUArrays.is_cached_storage(GPUArrays.storage(cached2_view))
        foreach(GPUArrays.unsafe_free!, (cached2_view, cached2_reshape, cached2_reinterpret))

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
        key2 = _keys[findfirst(!=(key), _keys)]
        @test length(cache.free[key]) == 1
        @test length(cache.free[key2]) == 1
        @test cache.free[key2][1] === GPUArrays.storage(cached5)

        # we should be able to re-use the early-freed
        GPUArrays.@cached cache begin
            cached5 = AT(zeros(T, dims))
        end

        # exceptions shouldn't cause issues
        @test_throws "Allowed exception" GPUArrays.@cached cache begin
            AT(zeros(T, dims))
            error("Allowed exception")
        end
        # NOTE: this should remaint the last test before calling `unsafe_free!` below,
        #       as it caught an erroneous assertion in the original code.

        # freeing all memory held by cache should free all allocations
        @test !GPUArrays.storage(cached1).freed
        @test GPUArrays.storage(cached1).cached
        @test !GPUArrays.storage(cached5).freed
        @test GPUArrays.storage(cached5).cached
        @test !GPUArrays.storage(uncached).freed
        @test !GPUArrays.storage(uncached).cached
        GPUArrays.unsafe_free!(cache)
        @test sizeof(cache) == 0
        @test !GPUArrays.is_cached_storage(GPUArrays.storage(cached1))
        @test GPUArrays.storage(cached1).freed
        @test !GPUArrays.storage(cached1).cached
        @test GPUArrays.storage(cached5).freed
        @test !GPUArrays.storage(cached5).cached
        @test !GPUArrays.storage(uncached).freed
        ## test that the underlying data was freed as well
        @test GPUArrays.storage(cached1).rc.count[] == 0
        @test GPUArrays.storage(cached5).rc.count[] == 0
        @test GPUArrays.storage(uncached).rc.count[] == 1

        # a derived array keeps the memory alive past invalidation, but it isn't
        # owned by the cache anymore
        GPUArrays.@cached cache begin
            cached6 = AT(zeros(T, dims))
            cached6_derived = reshape(cached6, :)
        end
        @test GPUArrays.is_cached_storage(GPUArrays.storage(cached6_derived))
        GPUArrays.unsafe_free!(cache)
        @test !GPUArrays.is_cached_storage(GPUArrays.storage(cached6_derived))
        @test GPUArrays.storage(cached6_derived).rc.count[] == 1
        @test Array(cached6_derived) == zeros(T, prod(dims))
        GPUArrays.unsafe_free!(cached6_derived)
        @test GPUArrays.storage(cached6_derived).rc.count[] == 0
    end
end
