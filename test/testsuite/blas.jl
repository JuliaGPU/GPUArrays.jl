function test_blas(AT)
    @testset "BLAS" begin
        @testset "matmul with element type $elty" for elty in (Float32, Float64, ComplexF32, ComplexF64)
            A = rand(elty, 5, 6)
            B = rand(elty, 6, 5)
            C = rand(elty, 5, 5)
            x = rand(elty, 5)
            y = rand(elty, 6)

            @test compare(*, AT, A, y)
            @test compare((t,s) -> transpose(t)*s, AT, A, x)
            @test compare((t,s) -> adjoint(t)*s  , AT, A, x)

            @test compare(*, AT, A, B)
            @test compare((t,s) -> transpose(t)*s           , AT, A, C)
            @test compare((t,s) -> t*transpose(s)           , AT, C, B)
            @test compare((t,s) -> transpose(t)*transpose(s), AT, A, B)
            @test compare((t,s) -> adjoint(t)*s             , AT, A, C)
            @test compare((t,s) -> t*adjoint(s)             , AT, C, B)
            @test compare((t,s) -> adjoint(t)*adjoint(s)    , AT, A, B)
            @test compare((t,s) -> transpose(t)*adjoint(s)  , AT, A, B)
            @test compare((t,s) -> adjoint(t)*transpose(s)  , AT, A, B)
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
