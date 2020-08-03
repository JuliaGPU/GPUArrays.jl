@testsuite "random" AT->begin
    @testset "rand" begin  # uniform
        for T in (Int8, Float32, Float64, Int64, Int32,
                    Complex{Float32}, Complex{Float64},
                    Complex{Int64}, Complex{Int32}), d in (2, (2,2))
            A = AT{T}(undef, d)
            B = copy(A)
            rand!(A)
            rand!(B)
            @test !any(A .== B)

            rng = GPUArrays.default_rng(AT)
            Random.seed!(rng)
            Random.seed!(rng, 1)
            rand!(rng, A)
            Random.seed!(rng, 1)
            rand!(rng, B)
            @test all(A .== B)
        end
    end

    @testset "randn" begin  # uniform
        for T in (Float32, Float64), d in (2, (2,2))
            A = AT{T}(undef, d)
            B = copy(A)
            randn!(A)
            randn!(B)
            @test !any(A .== B)

            rng = GPUArrays.default_rng(AT)
            Random.seed!(rng)
            Random.seed!(rng, 1)
            randn!(rng, A)
            Random.seed!(rng, 1)
            randn!(rng, B)
            @test all(A .== B)
        end
    end
end
