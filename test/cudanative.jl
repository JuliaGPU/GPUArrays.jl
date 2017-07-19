using GPUArrays
using GPUArrays: free
using CUDAnative, Base.Test
cuctx = CUBackend.init()
const cu = CUDAnative

# more complex function for broadcast
function cutest(a, b)
    x = cu.sqrt(cu.sin(a) * b) / 10f0
    y = 33f0 * x + cu.cos(b)
    y*10f0
end
function test2(b)
    x = cu.sqrt(cu.sin(b*2f0) * b) / 10f0
    y = 33f0*x + cu.cos(b)
    y*87f0
end


@testset "broadcast Float32" begin
    A = GPUArray(rand(Float32, 40, 40))
    A .= identity.(10f0)
    @test all(x-> x == 10, Array(A))
    A .= identity.(0.5f0)
    B = cutest.(A, 10f0)
    @test all(x-> x  ≈ jltest(0.5f0, 10f0), Array(B))
    broadcast!(identity, A, 2f0)
    C = A .* 10f0
    @test all(x-> x ≈ 20f0, Array(C))
    D = A .* B
    @test all(x-> x ≈ jltest(0.5f0, 10f0) * 2, Array(D))
    D .= A .* B .+ 10f0
    @test all(x-> x ≈ jltest(0.5f0, 10f0) * 2f0 + 10f0, Array(D))
    free(D); free(C); free(A); free(B)
end




function cu_angle(z)
    cu.atan2(imag(z), real(z))
end
@testset "broadcast Complex64" begin
    A = GPUArray(fill(10f0*im, 40, 40))
    A .= identity.(10f0*im)
    @test all(x-> x ≈ 10f0*im, Array(A))
    B = cu_angle.(A)
    @test all(x-> x ≈ angle(10f0*im), Array(B))
    A .= identity.(2f0*im)
    Array(B)
    C = A .* (2f0*im)
    @test all(x-> x ≈ 2f0*im * 2f0*im, Array(C))
    testval = Array(A)[1] * Array(B)[1]
    D = A .* B
    @test all(x-> x ≈ testval, Array(D))
    D .= A .* B .+ (0.5f0*im)
    @test all(x-> x ≈ (2f0*im * angle(10f0*im) + (0.5f0*im)), Array(D))
    free(D); free(C); free(A); free(B)
end

# @testset "fft Complex64" begin
#     A = rand(Float32, 7,6)
#     # Move data to GPU
#     B = GPUArray(A)
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
                range = T <: Integer ? (T(-10):T(10)) : T
                A = GPUArray(rand(range, dims))
                @test sum(A) ≈ sum(Array(A))
                @test maximum(A) ≈ maximum(Array(A))
                @test minimum(A) ≈ minimum(Array(A))
                @test prod(A) ≈ prod(Array(A))
            end
        end
    end
end


function cumap!(state, f, out, b)
    i = linear_index(out, state) # get the kernel index it gets scheduled on
    out[i] = f(b[i])
    return
end

@testset "Custom kernel from Julia function" begin
    x = GPUArray(rand(Float32, 100))
    y = GPUArray(rand(Float32, 100))
    gpu_call(cumap!, x, (cu.sin, x, y))
    jy = Array(y)
    @test map!(sin, jy, jy) ≈ Array(x)
end

if CUBackend.hasnvcc()
    @testset "Custom kernel from string function" begin
        x = GPUArray(rand(Float32, 100))
        y = GPUArray(rand(Float32, 100))
        source = """
        __global__ void copy(const float *input, float *output)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            output[i] = input[i];
        }
        """
        f = (source, :copy)
        gpu_call(f, x, (x, y))
        @test Array(x) == Array(y)
    end
end
