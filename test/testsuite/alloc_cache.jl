@testsuite "alloc cache" (AT, eltypes) -> begin
    if AT <: AbstractGPUArray
        cache = GPUArrays.AllocCache(AT)

        T, dims = Float32, (1, 2, 3)
        GPUArrays.@cached cache begin
            x1 = AT(zeros(T, dims))
        end
        @test sizeof(cache) == sizeof(T) * prod(dims)
        key = first(keys(cache.free))
        @test length(cache.free[key]) == 1
        @test length(cache.busy[key]) == 0
        @test x1 === cache.free[key][1]

        # Second allocation hits cache.
        GPUArrays.@cached cache begin
            x2 = AT(zeros(T, dims))
            # Does not hit the cache.
            GPUArrays.@uncached x_free = AT(zeros(T, dims))
        end
        @test sizeof(cache) == sizeof(T) * prod(dims)
        key = first(keys(cache.free))
        @test length(cache.free[key]) == 1
        @test length(cache.busy[key]) == 0
        @test x2 === cache.free[key][1]
        @test x_free !== x2

        # Third allocation is of different shape - allocates.
        dims = (2, 2)
        GPUArrays.@cached cache begin
            x3 = AT(zeros(T, dims))
        end
        _keys = collect(keys(cache.free))
        key2 = _keys[findfirst(i -> i != key, _keys)]
        @test length(cache.free[key]) == 1
        @test length(cache.free[key2]) == 1
        @test x3 === cache.free[key2][1]

        # Freeing all memory held by cache.
        GPUArrays.unsafe_free!(cache)
        @test sizeof(cache) == 0
    end
end
