using GPUArrays
using Base.Test, GPUArrays.TestSuite

function run_random(Typ)
    @testset "Random" begin
        @testset "rand" begin  # uniform
            for T in (Float32, Float64, Int64, Int32)
                @test length(rand(Typ{T,1}, (4,))) == 4
                @test length(rand(Typ{T}, (4,))) == 4
                @test length(rand(Typ{T}, 4)) == 4
                @test eltype(rand(Typ, 4)) == Float32
                @test length(rand(Typ, T, 4)) == 4
                @test length(rand(Typ{T,2}, (4,5))) == 20
                @test length(rand(Typ, T, 4, 5)) == 20
                A = rand(Typ{T,2}, (2,2))
                B = copy(A)
                @test all(A .== B)
                rand!(B)
                @test !any(A .== B)
            end
        end
    end
end
