function test_random(AT)
    @testset "Random" begin
        @testset "rand" begin  # uniform
            for T in (Float32, Float64, Int64, Int32)
                @test length(rand(AT{T,1}, (4,))) == 4
                @test length(rand(AT{T}, (4,))) == 4
                @test length(rand(AT{T}, 4)) == 4
                @test eltype(rand(AT, 4)) == Float32
                @test length(rand(AT, T, 4)) == 4
                @test length(rand(AT{T,2}, (4,5))) == 20
                @test length(rand(AT, T, 4, 5)) == 20
                A = rand(AT{T,2}, (2,2))
                B = copy(A)
                @test all(A .== B)
                rand!(B)
                @test !any(A .== B)
            end
        end
    end
end
