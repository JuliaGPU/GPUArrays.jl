@testsuite "random" (AT, eltypes)->begin
    rng = if AT <: AbstractGPUArray
        GPUArrays.default_rng(AT)
    else
        Random.default_rng()
    end
    cpu_rng = Random.default_rng()

    @testset "rand" begin  # uniform
        for T in eltypes, d in (10, (10,10))
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

            if rng != cpu_rng
                rand!(cpu_rng, A)
            end
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
        # XXX: randn calls sqrt, and Base's sqrt(::Complex) performs
        #      checked type conversions that throw boxed numbers.
        for T in filter(isrealfloattype, eltypes), d in (2, (2,2))
            A = AT{T}(undef, d)
            B = copy(A)
            randn!(rng, A)
            randn!(rng, B)
            @test Array(A) != Array(B)

            Random.seed!(rng)
            Random.seed!(rng, 1)
            randn!(rng, A)
            Random.seed!(rng, 1)
            randn!(rng, B)
            @test Array(A) == Array(B)

            if rng != cpu_rng
                randn!(cpu_rng, A)
            end
        end
    end
end
