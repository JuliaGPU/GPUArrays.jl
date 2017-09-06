using GPUArrays
using Base.Test

@allbackends "BLAS" ctx begin
    if GPUArrays.hasblas(ctx)
        @testset "matmul" begin
            for T in (Float32,) # TODO complex float64
                A, B = rand(T, 15, 15), rand(T, 15, 15)
                X = rand(T, 15)
                C = A * B
                D = A * X
                Agpu, Bgpu, Xgpu = GPUArray(A), GPUArray(B), GPUArray(X)
                Cgpu = Agpu * Bgpu
                Dgpu = Agpu * Xgpu
                @test Array(Cgpu) ≈ C
                @test Array(Dgpu) ≈ D
            end
        end
    end
    for T in (Complex64, Float32)
        @testset "scale!" begin
            x = rand(T, 13, 23)
            A = GPUArray(x)

            scale!(A, 77f0)
            scale!(x, 77f0)
            @test x ≈ Array(A)
        end
    end
end


@testset "BLAS" begin
  testf(*, rand(5, 5), rand(5, 5))
  testf(*, rand(5, 5), rand(5))
  testf(A_mul_Bt, rand(5, 5), rand(5, 5))
  testf(At_mul_B, rand(5, 5), rand(5, 5))
end
