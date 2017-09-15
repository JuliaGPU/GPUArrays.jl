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
            against_base(At_mul_B, T, (5, 5), (5, 5))
        end
        # for T in (Complex64, Float32)
        #     @testset "scale!" begin
        #         against_base(scale!, Typ{T}, (13, 23), 77f0)
        #     end
        # end
    end
end
