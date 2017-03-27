using GPUArrays
using Base.Test
JLBackend.init()

A = GPUArray(rand(33, 33))
B = convert(GPUArray, rand(33, 33))
C = A * B
c = Array(C)
@test c == Array(A) * Array(B)

@testset "mapreduce" begin
    @testset "inbuilds using mapreduce (sum maximum minimum prod)" begin
        for dims in ((4048,), (1024,1024), (77,), (1923,209))
            for T in (Float32, Int32)
                A = GPUArray(rand(T, dims))
                @test sum(A) ≈ sum(Array(A))
                @test maximum(A) ≈ maximum(Array(A))
                @test minimum(A) ≈ minimum(Array(A))
                @test prod(A) ≈ prod(Array(A))
            end
        end
    end
end

@testset "broadcast Float32" begin
    A = GPUArray(rand(Float32, 40, 40))
    A .= identity.(10f0)
    @test all(x-> x == 10, Array(A))

    A .= identity.(0.5f0)
    B = jltest.(A, 10f0)
    @test all(x-> x == jltest(0.5f0, 10f0), Array(B))
    A .= identity.(2f0)
    C = A .* 10f0
    @test all(x-> x == 20, Array(C))
    D = A .* B
    @test all(x-> x == jltest(0.5f0, 10f0) * 2, Array(D))
    D .= A .* B .+ 10f0
    @test all(x-> x == jltest(0.5f0, 10f0) * 2 + 10f0, Array(D))
end

@testset "broadcast Complex64" begin
    A = GPUArray(fill(10f0*im, 40, 40))

    A .= identity.(10f0*im)
    @test all(x-> x == 10f0*im, Array(A))

    B = angle.(A)
    @test all(x-> x == angle(10f0*im), Array(B))
    A .= identity.(2f0*im)
    C = A .* (2f0*im)
    @test all(x-> x == 2f0*im * 2f0*im, Array(C))
    D = A .* B
    @test all(x-> x == angle(10f0*im) * 2f0*im, Array(D))
    D .= A .* B .+ (0.5f0*im)
    @test all(x-> x == (2f0*im * angle(10f0*im) + (0.5f0*im)), Array(D))
end

@testset "fft Complex64" begin
    for n = 1:3
        @testset "N $n" begin
            a = rand(Complex64, ntuple(i-> 40, n))
            A = GPUArray(a)
            fft!(A)
            fft!(a)
            @test all(isapprox.(Array(A), a))
            ifft!(A)
            ifft!(a)
            @test all(isapprox.(Array(A), a))
        end
    end
end



@testset "mapidx" begin
    a = rand(Complex64, 1024)
    b = rand(Complex64, 1024)
    A = GPUArray(a)
    B = GPUArray(b)
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
    @test Array(A) == a
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
