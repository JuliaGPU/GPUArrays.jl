using Statistics

@testsuite "statistics" AT->begin
    @testset "std" begin
        @test compare(std, AT, rand(10))
        @test compare(std, AT, rand(10,1,2))
        @test compare(std, AT, rand(10,1,2); corrected=true)
        @test compare(std, AT, rand(10,1,2); dims=1)
    end

    @testset "var" begin
        @test compare(var, AT, rand(10))
        @test compare(var, AT, rand(10,1,2))
        @test compare(var, AT, rand(10,1,2); corrected=true)
        @test compare(var, AT, rand(10,1,2); dims=1)
        @test compare(var, AT, rand(10,1,2); dims=[1])
        @test compare(var, AT, rand(10,1,2); dims=(1,))
        @test compare(var, AT, rand(10,1,2); dims=[2,3])
        @test compare(var, AT, rand(10,1,2); dims=(2,3))
    end

    @testset "mean" begin
        @test compare(mean, AT, rand(2,2))
        @test compare(mean, AT, rand(2,2); dims=2)
        @test compare(mean, AT, rand(2,2,2); dims=[1,3])
        @test compare(x->mean(sin, x), AT, rand(2,2))
        @test compare(x->mean(sin, x; dims=2), AT, rand(2,2))
        @test compare(x->mean(sin, x; dims=[1,3]), AT, rand(2,2,2))
    end

    @testset "cov" begin
        s = 100
        @test compare(cov, AT, rand(s))
        @test compare(cov, AT, rand(Complex{Float64}, s))
        @test compare(cov, AT, rand(s, 2))
        @test compare(cov, AT, rand(Complex{Float64}, s, 2))
        @test compare(cov, AT, rand(s, 2); dims=2)
        @test compare(cov, AT, rand(Complex{Float64}, s, 2); dims=2)
        @test compare(cov, AT, rand(1:100, s))
    end

    @testset "cor" begin
        s = 100
        @test compare(cor, AT, rand(s))
        @test compare(cor, AT, rand(Complex{Float64}, s))
        @test compare(cor, AT, rand(s, 2))
        @test compare(cor, AT, rand(Complex{Float64}, s, 2))
        @test compare(cor, AT, rand(s, 2); dims=2)
        @test compare(cor, AT, rand(Complex{Float64}, s, 2); dims=2)
        @test compare(cor, AT, rand(1:100, s))
    end
end
