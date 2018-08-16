function test_linalg(AT)
    @testset "linear algebra" begin
        @testset "transpose" begin
            @test compare(adjoint, AT, rand(Float32, 32, 32))
        end

        @testset "permutedims" begin
            @test compare(x -> permutedims(x, (2, 1)), AT, rand(Float32, 2, 3))
            @test compare(x -> permutedims(x, (2, 1, 3)), AT, rand(Float32, 4, 5, 6))
            @test compare(x -> permutedims(x, (3, 1, 2)), AT, rand(Float32, 4, 5, 6))
        end
    end
end
