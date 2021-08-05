@testsuite "random" AT->begin
    rng = if AT <: AbstractGPUArray
        GPUArrays.default_rng(AT)
    else
        Random.default_rng()
    end

    @testset "rand" begin  # uniform
        for T in (Int16, Int32, Int64,
                  Float16, Float32, Float64,
                  Complex{Float16}, Complex{Float32}, Complex{Float64},
                  Complex{Int32}, Complex{Int64}), d in (10, (10,10))
            A = AT{T}(undef, d)
            B = copy(A)
            rand!(rng, A)
            rand!(rng, B)
            @test Array(A) != Array(B)

            Random.seed!(rng)
            Random.seed!(rng, 1)
            rand!(rng, A)
            Random.seed!(rng, 1)
            rand!(rng, B)
            @test all(Array(A) .== Array(B))
        end

        A = AT{Bool}(undef, 1024)
        fill!(A, false)
        rand!(rng, A)
        @test true in Array(A)
        fill!(A, true)
        rand!(rng, A)
        @test false in Array(A)
    end

    @testset "randn" begin  # normally-distributed
        for T in (Float16, Float32, Float64), d in (2, (2,2))
            A = AT{T}(undef, d)
            B = copy(A)
            randn!(rng, A)
            randn!(rng, B)
            @test !any(Array(A) .== Array(B))

            Random.seed!(rng)
            Random.seed!(rng, 1)
            randn!(rng, A)
            Random.seed!(rng, 1)
            randn!(rng, B)
            @test all(Array(A) .== Array(B))
        end
    end
end
