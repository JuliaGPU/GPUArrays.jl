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
            @test compare(A -> std(A; corrected = true), AT, rand(range, 10, 1, 2))
            @test compare(A -> std(A; dims = 1), AT, rand(range, 10, 1, 2))
        end

        @testset "var" begin
            @test compare(var, AT, rand(range, 10))
            @test compare(var, AT, rand(range, 10, 1, 2))
            @test compare(A -> var(A; corrected = true), AT, rand(range, 10, 1, 2))
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
        end

        @testset "cor" begin
            s = 100
            @test compare(cor, AT, rand(range, s)) nans = true
            @test compare(cor, AT, rand(range, s, 2)) nans = true
            @test compare(A -> cor(A; dims = 2), AT, rand(range, s, 2)) nans = true
        end
    end
end
