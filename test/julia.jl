using GPUArrays
using Base.Test
JLBackend.init()

@testset "mapreduce" begin
    @testset "inbuilds using mapreduce (sum maximum minimum prod)" begin
        for dims in ((4048,), (1024,1024), (77,), (1923,209))
            for T in (Float32, Int32)
                range = T <: Integer ? (T(-2):T(2)) : T
                A = GPUArray(rand(range, dims))
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
    @test all(x-> x ≈ jltest(0.5f0, 10f0), Array(B))
    A .= identity.(2f0)
    C = A .* 10f0
    @test all(x-> x ≈ 20, Array(C))
    D = A .* B
    @test all(x-> x ≈ jltest(0.5f0, 10f0) * 2, Array(D))
    D .= A .* B .+ 10f0
    @test all(x-> x ≈ jltest(0.5f0, 10f0) * 2 + 10f0, Array(D))
end

@testset "broadcast Complex64" begin
    A = GPUArray(fill(10f0*im, 40, 40))

    A .= identity.(10f0*im)
    @test all(x-> x ≈ 10f0*im, Array(A))

    B = angle.(A)
    @test all(x-> x ≈ angle(10f0*im), Array(B))
    A .= identity.(2f0*im)
    C = A .* (2f0*im)
    @test all(x-> x ≈ 2f0*im * 2f0*im, Array(C))
    D = A .* B
    @test all(x-> x ≈ angle(10f0*im) * 2f0*im, Array(D))
    D .= A .* B .+ (0.5f0*im)
    @test all(x-> x ≈ (2f0*im * angle(10f0*im) + (0.5f0*im)), Array(D))
end
