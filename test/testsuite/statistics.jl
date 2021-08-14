using Statistics

@testsuite "statistics" (AT, eltypes)->begin
    for ET in eltypes
        if !(ET in [Float16, Float32, Float64])
            continue
        end
        @testset "std" begin
            @test compare(std, AT, rand(ET, 10))
            @test compare(std, AT, rand(ET, 10,1,2))
            @test compare(std, AT, rand(ET, 10,1,2); corrected=true)
            @test compare(std, AT, rand(ET, 10,1,2); dims=1)
        end

        @testset "var" begin
            @test compare(var, AT, rand(ET, 10))
            @test compare(var, AT, rand(ET, 10,1,2))
            @test compare(var, AT, rand(ET, 10,1,2); corrected=true)
            @test compare(var, AT, rand(ET, 10,1,2); dims=1)
            @test compare(var, AT, rand(ET, 10,1,2); dims=[1])
            @test compare(var, AT, rand(ET, 10,1,2); dims=(1,))
            @test compare(var, AT, rand(ET, 10,1,2); dims=[2,3])
            @test compare(var, AT, rand(ET, 10,1,2); dims=(2,3))
        end

        @testset "mean" begin
            @test compare(mean, AT, rand(ET, 2, 2))
            @test compare(mean, AT, rand(ET, 2, 2); dims=2)
            @test compare(mean, AT, rand(ET, 2, 2, 2); dims=[1,3])
            @test compare(x->mean(sin, x), AT, rand(ET, 2,2))
            @test compare(x->mean(sin, x; dims=2), AT, rand(ET, 2,2))
            @test compare(x->mean(sin, x; dims=[1,3]), AT, rand(ET, 2,2,2))
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
            @test compare(cov, AT, rand(ET, s, 2); dims=2)
            if ET <: Real
                @test compare(cov, AT, rand(ET(1):ET(100), s))
            end
        end

        @testset "cor" begin
            s = 100
            @test compare(cor, AT, rand(ET, s))
            @test compare(cor, AT, rand(ET, s, 2))
            @test compare(cor, AT, rand(ET, s, 2); dims=2)
            if ET <: Real
                @test compare(cor, AT, rand(ET(1):ET(100), s))
            end
        end
    end
end
