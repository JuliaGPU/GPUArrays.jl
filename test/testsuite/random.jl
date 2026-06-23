# A non-`Number` isbits eltype with a custom sampler, mirroring how ColorTypes
# hooks colorants into the Random API. Exercises the generic element-wise rand!
# path; regression cover for JuliaGPU/CUDA.jl#3179.
struct RGBTriplet
    r::Float32
    g::Float32
    b::Float32
end
Random.rand(rng::AbstractRNG, ::Random.SamplerType{RGBTriplet}) =
    RGBTriplet(rand(rng, Float32), rand(rng, Float32), rand(rng, Float32))

@testsuite "random" (AT, eltypes)->begin
    rng = if AT <: AbstractGPUArray
        GPUArrays.RNG{AT}()
    else
        Random.default_rng()
    end

    @testset "rand" begin  # uniform
        @testset "$T $d" for T in eltypes, d in (2, (2,2), (2,2,2))
            A = AT{T}(undef, d)
            B = copy(A)
            rand!(rng, A)
            rand!(rng, B)
            @test Array(A) != Array(B)
        end

        # empty arrays
        B = AT{Float32}(undef, 0)
        rand!(rng, B)
        @test isempty(Array(B))

        # Bool coverage
        A = AT{Bool}(undef, 1024)
        fill!(A, false)
        rand!(rng, A)
        @test true in Array(A)
        fill!(A, true)
        rand!(rng, A)
        @test false in Array(A)
    end

    @testset "randn" begin  # normally-distributed
        @testset "$T $d" for T in filter(isrealfloattype, eltypes),
                              d in (2, (2,2), (2,2,2))
            A = AT{T}(undef, d)
            B = copy(A)
            randn!(rng, A)
            randn!(rng, B)
            @test Array(A) != Array(B)
        end

        # complex randn
        for T in filter(t -> t <: Complex && isrealfloattype(real(t)), eltypes)
            A = AT{T}(undef, 8)
            randn!(rng, A)
            @test !any(isnan, Array(A))
        end

        # empty arrays
        A = AT{Float32}(undef, 0)
        randn!(rng, A)
        @test isempty(Array(A))

        # Box-Muller should not produce infinities
        if Float32 in eltypes
            @test isfinite(maximum(randn!(rng, AT{Float32}(undef, 2^20))))
        end
    end

    @testset "seeding" begin
        if AT <: AbstractGPUArray
            # seeding should produce reproducible results
            for T in (Float32, Float64)
                if T in eltypes
                    Random.seed!(rng, 1)
                    A = rand!(rng, AT{T}(undef, 100))
                    Random.seed!(rng, 1)
                    B = rand!(rng, AT{T}(undef, 100))
                    @test Array(A) == Array(B)

                    Random.seed!(rng, 1)
                    A = randn!(rng, AT{T}(undef, 100))
                    Random.seed!(rng, 1)
                    B = randn!(rng, AT{T}(undef, 100))
                    @test Array(A) == Array(B)
                end
            end
        end
    end

    # a CPU RNG into a GPU array should route through a CPU generate + copyto!,
    # not the stdlib scalar setindex! fallback
    if AT <: AbstractGPUArray && Float32 in eltypes
        @testset "CPU RNG → GPU array" begin
            cpu_rng = MersenneTwister(1)
            A = AT{Float32}(undef, 16)
            @test (rand!(cpu_rng, A); true)
            @test (randn!(cpu_rng, A); true)
        end
    end

    # non-`Number` eltypes with a custom sampler must generate element-wise
    # rather than hit an ambiguous fallback (JuliaGPU/CUDA.jl#3179)
    if AT <: AbstractGPUArray
        @testset "non-Number eltype" begin
            A = AT{RGBTriplet}(undef, 1024)
            rand!(rng, A)
            h = Array(A)
            @test all(t -> 0f0 <= t.r <= 1f0 && 0f0 <= t.g <= 1f0 && 0f0 <= t.b <= 1f0, h)
            @test any(t -> t.r != h[1].r, h)
        end
    end
end
