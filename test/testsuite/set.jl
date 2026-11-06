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

@testsuite "set/operations" (AT, eltypes)->begin
    @testset "$ET" for ET in eltypes
        for (m, n) in ((0, 0), (0, 10), (10, 0), (1, 1), (100, 200))
            a, b, c = dups(ET, m), dups(ET, n), dups(ET, 50)
            for f in (union, intersect, setdiff, symdiff, issubset, issetequal, isdisjoint)
                @test compare_exact(f, AT, a, b)
            end
            for f in (union, intersect, setdiff, symdiff)
                @test compare_exact(f, AT, a, b, c)
                @test compare_exact(f, AT, a)
            end
        end
        @test compare_exact(intersect, AT, dups(ET, 4, 5), dups(ET, 3))
        @test compare_exact(issetequal, AT, ET[1, 2, 2], ET[2, 1])
    end

    # element types are promoted as in Base, but elements are compared without converting them
    @test compare_exact(union, AT, Int32[3, 1], Float32[2, 3])
    @test compare_exact(intersect, AT, Int32[3, 1, 2, 1], Float32[2, 1])
    @test compare_exact(setdiff, AT, Int32[3, 1, 2, 1], Float32[2.5, 1])
    @test compare_exact(symdiff, AT, Int16[1, 2], Float32[2, 4])
    big, rounded = Int32[2^24 + 1, 1], Float32[2^24, 1]
    for f in (union, intersect, setdiff, symdiff, issubset, isdisjoint)
        # Base throws when it would convert an element inexactly, which depends on the order
        @test compare_exact(f, AT, big, rounded)
        @test compare_exact(f, AT, rounded, big)
    end
    @test compare_exact(issubset, AT, [-1], UInt[1])
    @test compare_exact(intersect, AT, ComplexF32[1, 2im], Float32[1, 2])
    # isbits unions
    m = Union{Missing, Float32}[1, missing, 3]
    mc = Union{Missing, ComplexF32}[1, missing, 2im]
    for (x, y) in ((Float32[1, 2], m), (m, Float32[1, 2]), (ComplexF32[1, 2im], mc), (mc, Float32[1]))
        for f in (union, intersect, setdiff, symdiff)
            @test compare_exact(f, AT, x, y)
        end
    end

    # with `isequal`, signed zeros differ and NaNs are equal
    x, y = Float32[0, NaN, 1, -0.0], Float32[-0.0, NaN, 2]
    for f in (union, intersect, setdiff, symdiff)
        @test compare_exact(f, AT, x, y)
    end
    # The predicates use `isequal` too. Base's `issubset` and `isdisjoint` use `==` for arrays
    # of fewer than 70 elements, so these are longer.
    r = Float32.(1:100)
    @test compare_exact(issubset, AT, [NaN32; r], [NaN32; r; 200])
    @test compare_exact(issubset, AT, [0.0f0; r], [-0.0f0; r])
    @test compare_exact(isdisjoint, AT, [NaN32; r], [NaN32; r .+ 200])
    @test compare_exact(isdisjoint, AT, [0.0f0; r], [-0.0f0; r .+ 200])
    mr = Union{Missing, Float32}[missing; r]
    @test compare_exact(issubset, AT, mr, [mr; 200])
    @test compare_exact(isdisjoint, AT, mr, [missing; r .+ 200])
end
