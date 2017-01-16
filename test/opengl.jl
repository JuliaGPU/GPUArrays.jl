using GPUArrays
using Base.Test
import GPUArrays: GLBackend
import GLBackend: GLArray
glctx = GLBackend.init()

# more complex function for broadcast
function test(a, b)
    x = sqrt(sin(a) * b) / 10.0
    y = 33.0x + cos(b)
    y*10.0
end
A = GLArray(rand(Float32, 40, 40))
B = test.(A, 10.0)
GPUArrays.GLBackend._module_cache

@testset "broadcast Float32" begin
    A = GLArray(rand(Float32, 40, 40))
    A .= identity.(10.0)
    all(x-> x == 10, Array(A))
    A .= identity.(0.5)
    B = test.(A, 10.0)
    @test all(x-> isapprox(x, test(0.5f0, 10f0)), Array(B))
    A .= identity.(2.0)
    C = (*).(A, 10.0)
    @test all(x-> x == 20, Array(C))

    C = A .* 10f0
    @test all(x-> x == 20, Array(C))
    D = A .* B
    @test all(x-> x == jltest(0.5f0, 10f0) * 2, Array(D))
    D .= A .* B .+ 10f0
    @test all(x-> x == jltest(0.5f0, 10f0) * 2 + 10f0, Array(D))
end
#
# function cu_angle(z)
#     atan2(imag(z), real(z))
# end
# @testset "broadcast Complex64" begin
#     A = GLArray(fill(10f0*im, 40, 40))
#
#     A .= identity.(10f0*im)
#     @test all(x-> x == 10f0*im, Array(A))
#
#     B = cu_angle.(A)
#     @test all(x-> x == angle(10f0*im), Array(B))
#     A .= identity.(2f0*im)
#     C = A .* (2f0*im)
#     @test all(x-> x == 2f0*im * 2f0*im, Array(C))
#     D = A .* B
#     @test all(x-> x == angle(10f0*im) * 2f0*im, Array(D))
#     D .= A .* B .+ (0.5f0*im)
#     @test all(x-> x == (2f0*im * angle(10f0*im) + (0.5f0*im)), Array(D))
# end

# @testset "fft Complex64" begin
#     A = rand(Float32, 7,6)
#     # Move data to GPU
#     B = GLArray(A)
#     # Allocate space for the output (transformed array)
#     # Compute the FFT
#     fft!(B)
#     # Copy the result to main memory
#     # Compare against Julia's rfft
#     @test_approx_eq rfft(A) Array(B)
#     # Now compute the inverse transform
#     ifft!(B)
#     @test_approx_eq A Array(B)
# end

@testset "mapreduce" begin
    @testset "inbuilds using mapreduce (sum maximum minimum prod)" begin
        for dims in ((4048,), (1024,1024), (77,), (1923,209))
            for T in (Float32, Int32)
                A = GLArray(rand(T, dims))
                @test sum(A) ≈ sum(Array(A))
                @test maximum(A) ≈ maximum(Array(A))
                @test minimum(A) ≈ minimum(Array(A))
                @test prod(A) ≈ prod(Array(A))
            end
        end
    end
    # @testset "mapreduce with clojures" begin
    #     for dims in ((4048,), (1024,1024), (77,), (1923,209))
    #         for T in (Float32, Float64)
    #             A = GLArray(rand(T, dims))
    #             @test mapreduce(f1, op1, T(0), A) ≈ mapreduce(f1, op1, T(0), Array(A))
    #         end
    #     end
    # end
end
