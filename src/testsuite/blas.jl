using GPUArrays
using GPUArrays.TestSuite
using Base.Test

function run_blas(Typ)
    @testset "BLAS" begin
        T = Typ{Float32}
        @testset "matmul" begin
            against_base(*, T, (5, 5), (5, 5))
            against_base(*, T, (5, 5), (5))
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
    end
end
