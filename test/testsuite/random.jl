@testsuite "random" AT->begin
    @testcase "rand" begin  # uniform
        for T in (Int8, Float32, Float64, Int64, Int32,
                  Complex{Float32}, Complex{Float64},
                  Complex{Int64}, Complex{Int32}), d in (10, (10,10))
            A = AT{T}(undef, d)
            B = copy(A)
            rand!(A)
            rand!(B)
            @test Array(A) != Array(B)

            rng = if AT <: AbstractGPUArray
                GPUArrays.default_rng(AT)
            else
                Random.default_rng()
            end
            Random.seed!(rng)
            Random.seed!(rng, 1)
            rand!(rng, A)
            Random.seed!(rng, 1)
            rand!(rng, B)
            @test all(Array(A) .== Array(B))
        end

        A = AT{Bool}(undef, 5)
        rand!(A)
        @test true in Array(A)
        @test false in Array(A)
    end

    @testcase "randn" begin  # uniform
        for T in (Float32, Float64), d in (2, (2,2))
            A = AT{T}(undef, d)
            B = copy(A)
            randn!(A)
            randn!(B)
            @test !any(Array(A) .== Array(B))

            rng = if AT <: AbstractGPUArray
                GPUArrays.default_rng(AT)
            else
                Random.default_rng()
            end
            Random.seed!(rng)
            Random.seed!(rng, 1)
            randn!(rng, A)
            Random.seed!(rng, 1)
            randn!(rng, B)
            @test all(Array(A) .== Array(B))
        end
    end
end
