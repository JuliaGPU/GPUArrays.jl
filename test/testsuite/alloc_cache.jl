@testsuite "alloc cache" (AT, eltypes) -> begin
    if AT <: AbstractGPUArray
        cache = GPUArrays.AllocCache()

        # first allocation populates the cache
        T, dims = Float32, (1, 2, 3)
        GPUArrays.@cached cache begin
            x1 = AT(zeros(T, dims))
        end
        @test sizeof(cache) == sizeof(T) * prod(dims)
        key = first(keys(cache.free))
        @test length(cache.free[key]) == 1
        @test length(cache.busy[key]) == 0
        @test cache.free[key][1] === GPUArrays.storage(x1)

        # second allocation hits the cache
        GPUArrays.@cached cache begin
            x2 = AT(zeros(T, dims))

            # explicitly uncached ones don't
            GPUArrays.@uncached x_free = AT(zeros(T, dims))
        end
        @test sizeof(cache) == sizeof(T) * prod(dims)
        key = first(keys(cache.free))
        @test length(cache.free[key]) == 1
        @test length(cache.busy[key]) == 0
        @test cache.free[key][1] === GPUArrays.storage(x2)
        @test x_free !== x2

        # compatible shapes should also hit the cache
        dims = (3, 2, 1)
        GPUArrays.@cached cache begin
            x3 = AT(zeros(T, dims))
        end
        @test sizeof(cache) == sizeof(T) * prod(dims)
        key = first(keys(cache.free))
        @test length(cache.free[key]) == 1
        @test length(cache.busy[key]) == 0
        @test cache.free[key][1] === GPUArrays.storage(x3)

        # as should compatible eltypes
        T = Int32
        GPUArrays.@cached cache begin
            x4 = AT(zeros(T, dims))
        end
        @test sizeof(cache) == sizeof(T) * prod(dims)
        key = first(keys(cache.free))
        @test length(cache.free[key]) == 1
        @test length(cache.busy[key]) == 0
        @test cache.free[key][1] === GPUArrays.storage(x4)

        # different shapes should trigger a new allocation
        dims = (2, 2)
        GPUArrays.@cached cache begin
            x5 = AT(zeros(T, dims))

            # we're allowed to early free arrays, which shouldn't release the underlying data
            GPUArrays.unsafe_free!(x5)
        end
        _keys = collect(keys(cache.free))
        key2 = _keys[findfirst(i -> i != key, _keys)]
        @test length(cache.free[key]) == 1
        @test length(cache.free[key2]) == 1
        @test cache.free[key2][1] === GPUArrays.storage(x5)

        # freeing all memory held by cache should free all allocations
        @test !GPUArrays.storage(x1).freed
        @test GPUArrays.storage(x5).freed
        @test GPUArrays.storage(x5).rc.count[] == 1 # the ref appears freed, but the data isn't
        @test !GPUArrays.storage(x_free).freed
        GPUArrays.unsafe_free!(cache)
        @test sizeof(cache) == 0
        @test GPUArrays.storage(x1).freed
        @test GPUArrays.storage(x1).rc.count[] == 0
        @test GPUArrays.storage(x5).freed
        @test GPUArrays.storage(x5).rc.count[] == 0
        @test !GPUArrays.storage(x_free).freed
    end
end
