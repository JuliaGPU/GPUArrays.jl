using Base.Test
using JTensors
ctx = CLBackend.init()


@testset "CLBLAS Float32" begin
    A = JTensor(rand(Float32, 33, 33));
    B = JTensor(rand(Float32, 33, 33));
    C = A * B;
    c = Array(C);
    @test c ≈ Array(A) * Array(B)
end
# more complex function for broadcast
function test{T}(a::T, b)
    x = sqrt(sin(a) * b) / T(10.0)
    y = T(33.0)x + cos(b)
    y * T(10.0)
end

@testset "broadcast Float32" begin
    A = JTensor(rand(Float32, 40, 40))

    A .= identity.(10f0)
    @test all(x-> x == 10f0, Array(A))

    A .= identity.(0.5f0)
    B = test.(A, 10f0)
    @test all(x-> x ≈ test(0.5f0, 10f0), Array(B))
    A .= identity.(2f0)
    C = (*).(A, 10f0)
    @test all(x-> x == 20f0, Array(C))
    D = (*).(A, B)
    @test all(x-> x ≈ test(0.5f0, 10f0) * 2, Array(D))
    D .= (+).((*).(A, B), 10f0)
    @test all(x-> x ≈ test(0.5f0, 10f0) * 2 + 10f0, Array(D))
end

@testset "broadcast Complex64" begin
    A = JTensor(fill(10f0*im, 40, 40))
    A .= identity.(10f0*im)
    @test all(x-> x == 10f0*im, Array(A))

    B = angle.(A)
    @test all(x-> x == angle(10f0*im), Array(B))
    A .= identity.(2f0*im)
    C = (*).(A, (2f0*im))
    @test all(x-> x ≈ 2f0*im * 2f0*im, Array(C))
    D = (*).(A, B)
    @test all(x-> x ≈ angle(10f0*im) * 2f0*im, Array(D))
    D .= (+).((*).(A, B), (0.5f0*im))
    @test all(x-> x ≈ (2f0*im * angle(10f0*im) + (0.5f0*im)), Array(D))
end


@testset "fft Complex64" begin
    for n = 1:3
        @testset "N $n" begin
            a = rand(Complex64, ntuple(i-> 40, n))
            A = JTensor(a)
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
    A = JTensor(a)
    B = JTensor(b)
    off = 1
    mapidx(A, (B, Int32(off), Int32(length(A)))) do i, a, b, off, len
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
