using JTensors.CUBackend
using CUDAnative, Base.Test
cuctx = CUBackend.init()
const cu = CUDAnative

# more complex function for broadcast
function cutest(a, b)
    x = cu.sqrt(cu.sin(a) * b) / 10
    y = 33x + cu.cos(b)
    y*10
end
function test2(b)
    x = cu.sqrt(cu.sin(b*2.0) * b) / 10.0
    y = 33.0*x + cu.cos(b)
    y*87.0
end


@testset "broadcast Float32" begin
    A = JTensor(rand(Float32, 40, 40))

    A .= identity.(10f0)
    @test all(x-> x == 10, Array(A))

    A .= identity.(0.5f0)
    B = cutest.(A, 10f0)
    @test all(x-> x == jltest(0.5f0, 10f0), Array(B))
    A .= identity.(2f0)
    C = A .* 10f0
    @test all(x-> x == 20, Array(C))
    D = A .* B
    @test all(x-> x == jltest(0.5f0, 10f0) * 2, Array(D))
    D .= A .* B .+ 10f0
    @test all(x-> x == jltest(0.5f0, 10f0) * 2 + 10f0, Array(D))
end


function cu_angle(z)
    cu.atan2(imag(z), real(z))
end
@testset "broadcast Complex64" begin
    A = JTensor(fill(10f0*im, 40, 40))

    A .= identity.(10f0*im)
    @test all(x-> x == 10f0*im, Array(A))

    B = cu_angle.(A)
    @test all(x-> x == angle(10f0*im), Array(B))
    A .= identity.(2f0*im)
    C = A .* (2f0*im)
    @test all(x-> x == 2f0*im * 2f0*im, Array(C))
    D = A .* B
    @test all(x-> x == angle(10f0*im) * 2f0*im, Array(D))
    D .= A .* B .+ (0.5f0*im)
    @test all(x-> x == (2f0*im * angle(10f0*im) + (0.5f0*im)), Array(D))
end

# @testset "fft Complex64" begin
#     A = rand(Float32, 7,6)
#     # Move data to GPU
#     B = JTensor(A)
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
                A = JTensor(rand(range, dims))
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
    #             A = JTensor(rand(T, dims))
    #             @test mapreduce(f1, op1, T(0), A) ≈ mapreduce(f1, op1, T(0), Array(A))
    #         end
    #     end
    # end
end
@testset "mapidx" begin
    a = rand(Complex64, 1024)
    b = rand(Complex64, 1024)
    A = JTensor(a)
    B = JTensor(b)
    off = 1
    mapidx(A, (B, off, length(A))) do i, a, b, off, len
        x = b[i]
        x2 = b[min(i+off, len)]
        a[i] = x * x2
    end
    foreach(1:length(a)) do i
        x = b[i]
        x2 = b[min(i+off, length(a))]
        a[i] = x * x2
    end
    @test Array(A) ≈ a
end


function cumap!(f, out, b)
    i = linear_index(out) # get the kernel index it gets scheduled on
    out[i] = f(b[i])
    return
end

@testset "Custom kernel from Julia function" begin
    x = JTensor(rand(Float32, 100))
    y = JTensor(rand(Float32, 100))
    func = CUFunction(x, cumap!, cu.sin, x, y)
    # same here, x is just passed to supply a kernel size!
    func(x, cu.sin, x, y)
    jy = Array(y)
    @test map!(sin, jy, jy) ≈ Array(x)
end

@testset "Custom kernel from string function" begin
    x = JTensor(rand(Float32, 100))
    y = JTensor(rand(Float32, 100))
    source = """
    __global__ void copy(const float *input, float *output)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        output[i] = input[i];
    }
    """
    cucopy = CUFunction(x, (source, :copy), x, y)
    cucopy(x, x, y)
    @test Array(x) == Array(y)
end
