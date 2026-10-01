@testsuite "accumulate" (AT, eltypes)->begin
    @testset "$ET" for ET in eltypes
        range = ET <: Real ? (ET(1):ET(10)) : ET
        sizes = ET in (Float16, ComplexF16) ? (0, 1, 10, 100) : (0, 1, 10, 1000, 100_000)
        for n in sizes
            @test compare(A -> cumsum(A), AT, rand(range, n))
            @test compare(A -> accumulate(+, A), AT, rand(range, n))
            @test compare(A -> accumulate(+, A; init=one(ET)), AT, rand(range, n))
            @test compare((B, A) -> accumulate!(+, B, A), AT, zeros(ET, n), rand(range, n))
            # (into the element type, which small integers overflow)
            n <= 1000 && @test compare((B, A) -> cumsum!(B, A), AT, zeros(ET, n), rand(range, n))
        end
        @test compare(A -> cumprod(A), AT, rand(range, 10))
        # (into the element type, whose small integers overflow a product of larger values)
        @test compare((B, A) -> cumprod!(B, A), AT, zeros(ET, 10), rand(ET <: Real ? (ET(1):ET(2)) : ET, 10))

        # along dimensions, including one beyond the array's
        for dims in (1, 2, 3)
            @test compare(A -> cumsum(A; dims), AT, rand(range, 10, 20))
            @test compare(A -> accumulate(+, A; dims, init=one(ET)), AT, rand(range, 10, 20))
        end

        # views and reshaped arrays
        @test compare(A -> cumsum(view(A, 2:9)), AT, rand(range, 10))
        @test compare(A -> cumsum(reshape(A, 4, 5); dims=2), AT, rand(range, 20))
    end

    # Base's shape rules: an allocating scan without `dims` runs in linear order and keeps the
    # shape; an in-place one needs `dims` for arrays other than vectors
    M = rand(Float32, 4, 5)
    @test compare(A -> accumulate(+, A), AT, M)
    @test_throws ArgumentError accumulate!(+, AT(similar(M)), AT(M))
    @test_throws ArgumentError accumulate(+, AT(M); dims=0)
    @test_throws TypeError accumulate(+, AT(M); dims=:)
    @test_throws DimensionMismatch accumulate!(+, AT(zeros(Float32, 5, 4)), AT(M); dims=1)
    @test_throws DimensionMismatch accumulate!(+, AT(zeros(Float32, 5, 4)), AT(M); dims=3)
    @test_throws DimensionMismatch accumulate!(+, AT(zeros(Float32, 5, 4)), AT(M); dims=3, init=1f0)

    # An explicit `init=nothing` is an initial value
    something_add(a, b) = something(a, 0) + something(b, 0)
    @test compare(A -> accumulate(something_add, A; init=nothing), AT, rand(1:10, 100))

    # Base's rules, which AcceleratedKernels leaves to GPUArrays: element types (which differ
    # between Julia versions), a `dims` beyond the array's (which copies, ignoring `init`), and
    # the running type Base's front-end chooses through the destination
    for (f, x) in ((A -> accumulate(+, A), Int8[1, 2, 100]), (A -> accumulate(+, A; init=0), Int8[1, 2, 100]),
                   (cumsum, Int8[1, 2, 100]), (cumsum, Bool[1, 1, 0]), (cumprod, Int8[2, 3]),
                   (A -> accumulate(*, A), UInt8[16, 16, 16]),
                   (A -> accumulate(+, A; dims=3, init=10), rand(1:10, 3, 4)),
                   (A -> accumulate(+, A; dims=3), rand(1:10, 3, 4)),
                   (A -> cumsum(A; dims=2), rand(Int8(1):Int8(50), 3, 40)))
        @test compare_exact(f, AT, x)
    end
    @test compare_exact((B, A) -> accumulate!(+, B, A; init=0.5f0), AT, zeros(Int32, 3),
                        Float32[0.5, 1.0, 2.0])
    @test compare_exact((B, A) -> accumulate!(+, B, A; init=0.5f0, dims=2), AT,
                        zeros(Int32, 1, 3), Float32[0.5 1.0 2.0])
    @test compare_exact((B, A) -> accumulate!(+, B, A; dims=3, init=10), AT, zeros(Int, 3, 4),
                        rand(1:10, 3, 4))

    # Associative operators need not be commutative
    @test compare(A -> accumulate((a, b) -> a, A), AT, rand(1:10, 1000))
    @test compare(A -> accumulate((a, b) -> a, A; dims=2), AT, rand(1:10, 10, 100))
end
