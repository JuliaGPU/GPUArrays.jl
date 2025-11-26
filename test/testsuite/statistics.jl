using Statistics

@testsuite "statistics" (AT, eltypes)->begin
    for ET in eltypes
        if !(ET in [Float16, Float32, Float64])
            continue
        end
        range = ET <: Real ? (ET(1):ET(100)) : ET

        @testset "std" begin
            @test compare(std, AT, rand(range, 10))
            @test compare(std, AT, rand(range, 10,1,2))
            @test compare(A->std(A; corrected=true), AT, rand(range, 10,1,2))
            @test compare(A->std(A; dims=1), AT, rand(range, 10,1,2))
        end

        @testset "var" begin
            @test compare(var, AT, rand(range, 10))
            @test compare(var, AT, rand(range, 10,1,2))
            @test compare(A->var(A; corrected=true), AT, rand(range, 10,1,2))
            @test compare(A->var(A; dims=1), AT, rand(range, 10,1,2))
            @test compare(A->var(A; dims=[1]), AT, rand(range, 10,1,2))
            @test compare(A->var(A; dims=(1,)), AT, rand(range, 10,1,2))
            @test compare(A->var(A; dims=[2,3]), AT, rand(range, 10,1,2))
            @test compare(A->var(A; dims=(2,3)), AT, rand(range, 10,1,2))
        end

        @testset "mean" begin
            @test compare(mean, AT, rand(range, 2, 2))
            @test compare(A->mean(A; dims=2), AT, rand(range, 2, 2))
            @test compare(A->mean(A; dims=[1,3]), AT, rand(range, 2, 2, 2))
            @test compare(A->mean(sin, A), AT, rand(range, 2,2))
            @test compare(A->mean(sin, A; dims=2), AT, rand(range, 2,2))
            @test compare(A->mean(sin, A; dims=[1,3]), AT, rand(range, 2,2,2))
        end
    end

    for ET in eltypes
        if !(ET in [Float32, Float64, Float16, ComplexF32, ComplexF64])
            continue
        end
        range = ET <: Real ? (ET(1):ET(100)) : ET

        @testset "cov" begin
            s = 100
            @test compare(cov, AT, rand(range, s))
            @test compare(cov, AT, rand(range, s, 2))
            @test compare(A->cov(A; dims=2), AT, rand(range, s, 2))
        end

        @testset "cor" begin
            s = 100
            @test compare(cor, AT, rand(range, s)) nans=true
            @test compare(cor, AT, rand(range, s, 2)) nans=true
            @test compare(A->cor(A; dims=2), AT, rand(range, s, 2)) nans=true
        end
    end
end
