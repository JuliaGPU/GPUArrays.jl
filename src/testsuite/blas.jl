using GPUArrays
using GPUArrays.TestSuite
using Base.Test

function run_blas(Typ)
    @testset "BLAS" begin
        T = Typ{Float32}
        @testset "matmul" begin
            against_base(*, T, (5, 5), (5, 5))
            against_base(*, T, (5, 5), (5,))
            against_base(A_mul_Bt, T, (5, 5), (5, 5))
            against_base(A_mul_Bt!, T, (10, 32), (10, 60), (32, 60))
            against_base(At_mul_B, T, (5, 5), (5, 5))
            against_base(At_mul_Bt, T, (10, 15), (1, 10))
            against_base(At_mul_B, T, (10, 15), (10,))
            against_base(A_mul_B!, T, (15,), (15, 10), (10,))
        end
        for T in (Complex64, Float32)
            @testset "scale! $T" begin
                against_base(scale!, Typ{T}, (13, 23), 77f0)
            end
        end
        @testset "gbmv" begin
            m, n = 10, 11
            A, x, y = randn(Float32, 3, n), randn(Float32, n), zeros(Float32, m)

            Ag, xg, yg = Typ(A), Typ(x), Typ(y)

            BLAS.gbmv!('N', m, 1, 1, 1f0, A, x, 0f0, y)
            BLAS.gbmv!('N', m, 1, 1, 1f0, Ag, xg, 0f0, yg)
            @test y ≈ Array(yg)
        end
    end
end
