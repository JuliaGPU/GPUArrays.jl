using Statistics

@testsuite "statistics" (AT, eltypes)->begin
    for ET in eltypes
        if !(ET in [Float16, Float32, Float64])
            continue
        end
        # Larger range for Float16 to avoid numerical instability
        range = ET == Float16 ? (ET(1):ET(100)) : ET

        @testset "std" begin
            @test compare(std, AT, rand(range, 10))
            @test compare(std, AT, rand(range, 10, 1, 2))
            @test compare(A -> std(A; corrected = false), AT, rand(range, 10, 1, 2))
            @test compare(A -> std(A; dims = 1), AT, rand(range, 10, 1, 2))
        end

        @testset "var" begin
            @test compare(var, AT, rand(range, 10))
            @test compare(var, AT, rand(range, 10, 1, 2))
            @test compare(A -> var(A; corrected = false), AT, rand(range, 10, 1, 2))
            @test compare(A -> var(A; dims = 1), AT, rand(range, 10, 1, 2))
            @test compare(A -> var(A; dims = [1]), AT, rand(range, 10, 1, 2))
            @test compare(A -> var(A; dims = (1,)), AT, rand(range, 10, 1, 2))
            @test compare(A -> var(A; dims = [2, 3]), AT, rand(range, 10, 1, 2))
            @test compare(A -> var(A; dims = (2, 3)), AT, rand(range, 10, 1, 2))
        end

        @testset "mean" begin
            @test compare(mean, AT, rand(range, 2, 2))
            @test compare(A -> mean(A; dims = 2), AT, rand(range, 2, 2))
            @test compare(A -> mean(A; dims = [1, 3]), AT, rand(range, 2, 2, 2))
            # sin of `range` can cancel to near zero, where the summation order decides the
            # result; values in [0, 1) keep every term positive
            @test compare(A -> mean(sin, A), AT, rand(ET, 2, 2))
            @test compare(A -> mean(sin, A; dims = 2), AT, rand(ET, 2, 2))
            @test compare(A -> mean(sin, A; dims = [1, 3]), AT, rand(ET, 2, 2, 2))
        end
    end

    @testset "var and std" begin
        for ET in (Float32, ComplexF32, Int64)
            ET in eltypes || continue
            # (integers have a Float64 variance)
            ET <: Integer && !(Float64 in eltypes) && continue
            x = rand(ET, 10, 3, 2)
            # complex data has a real variance
            @test compare(A -> var(A; dims = 2), AT, x)
            @test compare(A -> std(A; dims = 1), AT, x)
            @test compare(A -> var(A; dims = (1, 3)), AT, x)
            @test compare(A -> std(A; dims = 2, corrected = false), AT, x)
            @test compare((A, m) -> varm(A, m; dims = 1), AT, x, mean(x; dims = 1))
            @test compare((A, m) -> stdm(A, m; dims = 3), AT, x, mean(x; dims = 3))
            @test compare_exact(A -> eltype(var(A; dims = 1)), AT, x)
            # an empty slice has a NaN variance, a single element one with `corrected`
            @test compare(A -> var(A; dims = 1), AT, zeros(ET, 0, 3)) nans = true
            @test compare(A -> var(A; dims = 2), AT, zeros(ET, 3, 0)) nans = true
            @test compare(A -> var(A; dims = 1), AT, rand(ET, 1, 3)) nans = true
        end
        if Float32 in eltypes
            x = rand(Float32, 10, 3)
            # the means must have the reduced shape, up to trailing singleton dimensions
            @test compare((A, m) -> varm(A, m; dims = 1), AT, x, mean(x; dims = 1)[:, :, :])
            @test_throws DimensionMismatch varm(AT(x), AT(mean(x; dims = 2)); dims = 1)
            # (Statistics.jl throws a MethodError, from subtracting the means from each element)
            AT <: AbstractGPUArray && @test_throws ArgumentError varm(AT(x), AT(mean(x; dims = 1)))
            # the result has the type of the data, not of the means, and without data the means
            # are not checked
            @test compare((A, m) -> var(A; dims = 1, mean = m), AT, zeros(Float32, 0, 2), zeros(Float32, 1, 5)) nans = true
            if Float64 in eltypes
                @test compare_exact((A, m) -> eltype(var(A; dims = 1, mean = m)), AT, x, rand(1, 3))
            end
        end
    end

    if Float16 in eltypes
        # the deviations are summed before scaling, so that small ones don't vanish
        @test compare(A -> var(A; dims = 1), AT, repeat(Float16[0.0007, -0.0007], 512))
        # in the type of the deviations, so that wider means keep the sum from overflowing
        if Float32 in eltypes
            @test compare((A, m) -> varm(A, m; dims = 1), AT, zeros(Float16, 1023, 1), Float32[8.0025;;])
        end
    end

    if Int64 in eltypes && Float64 in eltypes
        # integer deviations are summed in floating point, so they don't overflow
        @test compare((A, m) -> varm(A, m; dims = 1, corrected = false), AT,
                      fill(2_000_000_000, 3, 1), zeros(Int64, 1, 1))
    end

    @testset "mean accumulation" begin
        # integers are summed in floating point, so they don't overflow
        if Int64 in eltypes && Float64 in eltypes
            x = fill(typemax(Int64) ÷ 2, 4, 2)
            @test compare(mean, AT, x)
            @test compare(A -> mean(A; dims = 1), AT, x)
            @test compare(A -> mean(identity, A), AT, x)
            @test compare(A -> mean(identity, A; dims = 2), AT, x)
            @test compare(mean, AT, zeros(Int64, 0, 2)) nans = true
            @test compare(A -> mean(A; dims = 1), AT, zeros(Int64, 0, 2)) nans = true
        end
        if Int32 in eltypes && Float32 in eltypes
            x = rand(Int32, 10, 3)
            @test compare(A -> mean(Float32, A), AT, x)
            @test compare(A -> mean(Float32, A; dims = 1), AT, x)
        end

        # dividing after summing is exact here, scaling every term first is not
        if Float32 in eltypes
            @test all(isone, Array(mean(AT(ones(Float32, 640, 64)); dims = 1)))
        end
    end

    @testset "cov and cor of integers" begin
        if Int64 in eltypes && Float64 in eltypes
            x, y = rand(1:9, 50), rand(1:9, 50)
            X, Y = rand(1:9, 50, 2), rand(1:9, 50, 3)
            @test compare(cov, AT, x, y)
            @test compare(cor, AT, x, y)
            @test compare(cov, AT, X, Y)
            @test compare(cor, AT, X, Y)
            @test compare(cor, AT, x, Y)
        end
        # the accumulators widen to the type of the centred data
        if Float32 in eltypes && Float64 in eltypes
            # (summed in Float32, in another order than on the CPU)
            @test compare(cor, AT, rand(Float32, 50), rand(50); rtol = 1e-5)
        end
        if Float32 in eltypes
            @test_throws ArgumentError cor(AT(Float32[]), AT(Float32[]))
            @test_throws DimensionMismatch cor(AT(rand(Float32, 3)), AT(rand(Float32, 4)))
        end
    end

    for ET in eltypes
        # Doesn't work with ComplexF32 in oneAPI for some reason.
        if !(ET in [Float32, Float64, Float16, ComplexF64])
            continue
        end
        # Integers for Float16, so that `cor` of two-element rows have no subnormal variances,
        # but small ones, so that `cov` of 100 samples does not overflow
        range = ET == Float16 ? (ET(1):ET(10)) : ET

        @testset "cov" begin
            s = 100
            @test compare(cov, AT, rand(range, s))
            @test compare(cov, AT, rand(range, s, 2))
            @test compare(A -> cov(A; dims = 2), AT, rand(range, s, 2))
            # of two sets of variables
            @test compare(cov, AT, rand(range, s), rand(range, s))
            @test compare((x, y) -> cov(x, y; corrected = false), AT, rand(range, s), rand(range, s))
            @test compare(cov, AT, rand(range, s), rand(range, s, 2))
            @test compare(cov, AT, rand(range, s, 2), rand(range, s))
            @test compare(cov, AT, rand(range, s, 2), rand(range, s, 3))
            @test compare((x, y) -> cov(x, y; dims = 2), AT, rand(range, 2, s), rand(range, 3, s))
            @test compare_exact((x, y) -> typeof(cov(x, y)), AT, rand(range, s), rand(range, s))
        end

        @testset "cor" begin
            s = 100
            @test compare(cor, AT, rand(range, s)) nans = true
            @test compare(cor, AT, rand(range, s, 2)) nans = true
            @test compare(A -> cor(A; dims = 2), AT, rand(range, s, 2)) nans = true
            # of two sets of variables
            @test compare(cor, AT, rand(range, s), rand(range, s))
            @test compare(cor, AT, rand(range, s), rand(range, s, 2))
            @test compare(cor, AT, rand(range, s, 2), rand(range, s))
            @test compare(cor, AT, rand(range, s, 2), rand(range, s, 3))
            @test compare((x, y) -> cor(x, y; dims = 2), AT, rand(range, 2, s), rand(range, 3, s))
            @test compare_exact((x, y) -> typeof(cor(x, y)), AT, rand(range, s), rand(range, s))
        end
    end

    @testset "median" begin
        for ET in (Int16, Int64, Float16, Float32, Float64)
            ET in eltypes || continue
            # integers have a floating-point median; keep Float16 data exact
            range = ET == Float16 ? (ET(1):ET(100)) : ET <: Integer ? (ET(-50):ET(50)) : ET
            for n in (1, 2, 101, 1000)
                @test compare_exact(median, AT, rand(range, n))
                @test compare_exact(median!, AT, rand(range, n))
            end
            x = rand(range, 6, 5, 4)
            @test compare_exact(median, AT, x)
            @test compare_exact(median!, AT, x)
            # (integers have a Float64 median, which along `dims` is computed on the device)
            ET <: Integer && !(Float64 in eltypes) && continue
            for dims in (1, 2, 3, (1, 2), (1, 3), (2, 3), (1, 2, 3), [3, 1, 1])
                @test compare_exact(A -> median(A; dims), AT, x)
            end
            @test compare_exact(A -> median(A; dims = 1), AT, rand(range, 7, 3))
            @test compare_exact(A -> median(A; dims = 2), AT, rand(range, 3, 1))
            @test compare_exact(A -> median(abs, A), AT, rand(range, 10))
            # empty data, or slices
            @test compare_exact(median, AT, ET[])
            @test compare_exact(A -> median(A; dims = 1), AT, zeros(ET, 0, 3))
            @test compare_exact(A -> median(A; dims = 0), AT, x)
            @test compare_exact(A -> median(A; dims = 4), AT, x)
            @test compare_exact(A -> median(A; dims = 1.0), AT, x)
            @test compare_exact(A -> median(A; dims = Set([1, 3])), AT, x)
            @test compare_exact(A -> median(A; dims = (d for d in 2:3)), AT, x)
            # (Statistics.jl throws a MethodError, from calling `median!` on each element)
            @test_throws Exception median(AT(x); dims = ())
            # (Statistics.jl's `mapslices` fails on an empty result)
            AT <: AbstractGPUArray && @test size(median(AT(zeros(ET, 3, 0)); dims = 1)) == (1, 0)
        end
        if Float32 in eltypes
            # NaN propagates
            x = rand(Float32, 101)
            x[17] = NaN32
            @test compare_exact(median, AT, x)
            @test compare_exact(median, AT, x[1:100])
            X = rand(Float32, 7, 6)
            X[3, 4] = NaN32
            X[7, 1] = NaN32
            @test compare_exact(A -> median(A; dims = 1), AT, X)
            @test compare_exact(A -> median(A; dims = 2), AT, X)
            # signed zeros are ordered
            @test compare_exact(median, AT, Float32[0.0, -0.0, -0.0, 1.0])
            # missing takes precedence over NaN, and a slice without it has a median
            y = Union{Missing, Float32}[1, NaN, 3]
            @test compare_exact(median, AT, y)
            y[1] = missing
            @test compare_exact(median, AT, y)
            Y = Union{Missing, Float32}[1 2; 4 3; 5 6]
            @test compare_exact(A -> median(A; dims = 1), AT, Y)
            Y[2, 1] = missing
            # (Statistics.jl's `mapslices` cannot store a `missing` median)
            AT <: AbstractGPUArray && @test isequal(Array(median(AT(Y); dims = 1)), [missing 3.0f0])
            @test compare_exact(A -> median(A; dims = 1), AT, fill(missing, 2, 2))
            @test compare_exact(A -> median(A; dims = 1), AT, fill(missing, 0, 2))
        end
    end
end
