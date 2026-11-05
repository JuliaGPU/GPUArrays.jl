# random elements with many duplicates
dups(ET, dims...) = ET <: Complex ? ET.(rand(1:3, dims...), rand(1:3, dims...)) :
                                     ET.(rand(1:5, dims...))

@testsuite "set/unique" (AT, eltypes)->begin
    @testset "$ET" for ET in eltypes
        # Base returns elements in the order they first occur
        for n in (0, 1, 2, 10, 10000)
            @test compare_exact(unique, AT, dups(ET, n))
            @test compare_exact(allunique, AT, dups(ET, n))
        end
        @test compare_exact(unique, AT, dups(ET, 10, 20))
        @test compare_exact(allunique, AT, ET.(collect(1:100)))
        @test compare_exact(A -> unique(abs2, A), AT, dups(ET, 1000))
        @test compare_exact(A -> unique(abs2, A), AT, ET[-2, 2, 1, -1])
        @test compare_exact(A -> unique!(A), AT, dups(ET, 1000))
        @test compare_exact(A -> unique!(abs2, A), AT, dups(ET, 1000))
        if hasmethod(allunique, Tuple{Any, Any})
            @test compare_exact(A -> allunique(abs2, A), AT, dups(ET, 3))
        end

        # unique slices
        for dims in (1, 2, 3)
            @test compare_exact(A -> unique(A; dims), AT, dups(ET, 100, 2))
            @test compare_exact(A -> unique(A; dims), AT, dups(ET, 2, 3, 100))
        end
        @test compare_exact(A -> unique(A; dims=2), AT, dups(ET, 2, 0))
        @test compare_exact(A -> unique(A; dims=1), AT, dups(ET, 1, 3))
        # Base returns a copy when `dims` is out of range
        for dims in (-1, 0, 4)
            @test compare_exact(A -> unique(A; dims), AT, dups(ET, 2, 3, 4))
        end

        # views and reshaped arrays
        @test compare_exact(A -> unique(view(A, 11:90)), AT, dups(ET, 100))
        @test compare_exact(A -> unique(reshape(A, 10, 10); dims=2), AT, dups(ET, 100))
    end

    # with `isequal`, signed zeros differ and NaNs are equal
    for ET in filter(isrealfloattype, eltypes)
        x = ET[0, -0.0, NaN, 1, -NaN, NaN, -0.0, 1]
        @test compare_exact(unique, AT, x)
        @test compare_exact(allunique, AT, x)
        @test compare_exact(A -> unique(A; dims=2), AT, [x x x]')
    end

    @test compare_exact(unique, AT, rand(Bool, 100))

    # isbits unions, including ones with `missing`
    for x in (Union{Int32, Float32}[Int32(1), 1.0f0, Int32(2), 2.0f0, Int32(1)],
              Union{Missing, Float32}[1, missing, 2, missing, 1],
              Union{Missing, ComplexF32}[1, missing, 1im, missing, 1])
        @test compare_exact(unique, AT, x)
        @test compare_exact(allunique, AT, x)
        @test compare_exact(A -> unique(A; dims=1), AT, x)
    end

    # elements without `isless` are not supported (Base handles them by hashing)
    if AT <: AbstractGPUArray
        @test_throws Exception unique(AT([Some(Int32(1)), Some(Int32(2))]))
    end
end
