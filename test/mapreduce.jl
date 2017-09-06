
@allbackends "mapreduce" backend begin
    N = 18
    y = rand(Float32, N, N)
    x = GPUArray(y)
    @test sum(y, 2) ≈ Array(sum(x, 2))
    @test sum(y, 1) ≈ Array(sum(x, 1))

    y = rand(Float32, N, 10)
    x = GPUArray(y)
    @test sum(y, 2) ≈ Array(sum(x, 2))
    @test sum(y, 1) ≈ Array(sum(x, 1))

    y = rand(Float32, 10, N)
    x = GPUArray(y)
    @test sum(y, 2) ≈ Array(sum(x, 2))
    @test sum(y, 1) ≈ Array(sum(x, 1))

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
