using Statistics

@testsuite "statistics" (AT, eltypes)->begin
    for ET in eltypes
        if !(ET in [Float16, Float32, Float64])
            continue
        end
        @testset "std" begin
            @test compare(std, AT, rand(ET, 10))
            @test compare(std, AT, rand(ET, 10,1,2))
            @test compare(A->std(A; corrected=true), AT, rand(ET, 10,1,2))
            @test compare(A->std(A; dims=1), AT, rand(ET, 10,1,2))
        end

        @testset "var" begin
            @test compare(var, AT, rand(ET, 10))
            @test compare(var, AT, rand(ET, 10,1,2))
            @test compare(A->var(A; corrected=true), AT, rand(ET, 10,1,2))
            @test compare(A->var(A; dims=1), AT, rand(ET, 10,1,2))
            @test compare(A->var(A; dims=[1]), AT, rand(ET, 10,1,2))
            @test compare(A->var(A; dims=(1,)), AT, rand(ET, 10,1,2))
            @test compare(A->var(A; dims=[2,3]), AT, rand(ET, 10,1,2))
            @test compare(A->var(A; dims=(2,3)), AT, rand(ET, 10,1,2))
        end

        @testset "mean" begin
            @test compare(mean, AT, rand(ET, 2, 2))
            @test compare(A->mean(A; dims=2), AT, rand(ET, 2, 2))
            @test compare(A->mean(A; dims=[1,3]), AT, rand(ET, 2, 2, 2))
            @test compare(A->mean(sin, A), AT, rand(ET, 2,2))
            @test compare(A->mean(sin, A; dims=2), AT, rand(ET, 2,2))
            @test compare(A->mean(sin, A; dims=[1,3]), AT, rand(ET, 2,2,2))
        end
    end

    for ET in eltypes
        # Doesn't work with ComplexF32 in oneAPI for some reason.
        if !(ET in [Float32, Float64, Float16, ComplexF64])
            continue
        end
        @testset "cov" begin
            s = 100
            @test compare(cov, AT, rand(ET, s))
            @test compare(cov, AT, rand(ET, s, 2))
            @test compare(A->cov(A; dims=2), AT, rand(ET, s, 2))
            if ET <: Real
                @test compare(cov, AT, rand(ET(1):ET(100), s))
            end
        end

        @testset "cor" begin
            s = 100
            @test compare(cor, AT, rand(ET, s)) nans=true
            @test compare(cor, AT, rand(ET, s, 2)) nans=true
            @test compare(A->cor(A; dims=2), AT, rand(ET, s, 2)) nans=true
            if ET <: Real
                @test compare(cor, AT, rand(ET(1):ET(100), s)) nans=true
            end
        end
    end
end
