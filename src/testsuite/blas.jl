function test_blas(AT)
    @testset "BLAS" begin
        @testset "matmul" begin
            @test compare(*, AT, rand(Float32, 5, 5), rand(Float32, 5, 5))
            @test compare(*, AT, rand(Float32, 5, 5), rand(Float32, 5))
            @test compare((a, b)-> a * transpose(b), AT, rand(Float32, 5, 5), rand(Float32, 5, 5))
            @test compare((c, a, b)-> mul!(c, a, transpose(b)), AT, rand(Float32, 10, 32), rand(Float32, 10, 60), rand(Float32, 32, 60))
            @test compare((a, b)-> transpose(a) * b, AT, rand(Float32, 5, 5), rand(Float32, 5, 5))
            @test compare((a, b)-> transpose(a) * transpose(b), AT, rand(Float32, 10, 15), rand(Float32, 1, 10))
            @test compare((a, b)-> transpose(a) * b, AT, rand(Float32, 10, 15), rand(Float32, 10))
            @test compare(mul!, AT, rand(Float32, 15), rand(Float32, 15, 10), rand(Float32, 10))
        end

        for T in (ComplexF32, Float32)
            @testset "rmul! $T" begin
                @test compare(rmul!, AT, rand(T, 13, 23), Ref(77f0))
            end
        end

        @testset "gbmv" begin
            m, n = 10, 11
            A, x, y = randn(Float32, 3, n), randn(Float32, n), fill(0f0, m)

            Ag, xg, yg = AT(A), AT(x), AT(y)

            BLAS.gbmv!('N', m, 1, 1, 1f0, A, x, 0f0, y)
            BLAS.gbmv!('N', m, 1, 1, 1f0, Ag, xg, 0f0, yg)
            @test y ≈ Array(yg)
        end
    end
end
