function test_blas(AT)
    @testset "BLAS" begin
        @testset "matmul with element type $elty" for elty in (Float32, Float64, ComplexF32, ComplexF64)
            A = rand(elty, 5, 6)
            B = rand(elty, 6, 5)
            C = rand(elty, 5, 5)
            x = rand(elty, 5)
            y = rand(elty, 6)

            @test compare(*, AT, A, y)
            @test compare(*, AT, transpose(A), x)
            @test compare(*, AT, adjoint(A), x)

            @test compare(*, AT, A, B)
            @test compare(*, AT, transpose(A), C)
            @test compare(*, AT, C, transpose(B))
            @test compare(*, AT, transpose(A), transpose(B))
            @test compare(*, AT, adjoint(A), C)
            @test compare(*, AT, C, adjoint(B))
            @test compare(*, AT, adjoint(A), adjoint(B))
            @test compare(*, AT, transpose(A), adjoint(B))
            @test compare(*, AT, adjoint(A), transpose(B))
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
