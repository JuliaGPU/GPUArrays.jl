using GPUArrays
using Base.Test

@testset "matmul" begin
    for T in (Float32, Float64) # TODO complex
        A, B = rand(T, 33, 33), rand(T, 33, 33)
        X = rand(T, 33)
        C = A * B
        D = A * X
        for backend in supported_backends()
            ctx = GPUArrays.init(backend)
            if GPUArrays.hasblas(ctx)
                @testset "$backend" begin
                    Agpu, Bgpu, Xgpu = GPUArray(A), GPUArray(B), GPUArray(X)
                    Cgpu = Agpu * Bgpu
                    Dgpu = Agpu * Xgpu
                    @test Array(Cgpu) ≈ C
                    @test Array(Dgpu) ≈ D
                end
            end
        end
    end
end
