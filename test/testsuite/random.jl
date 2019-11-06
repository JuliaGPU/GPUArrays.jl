function test_random(AT)
    @testset "Random" begin
        @testset "rand" begin  # uniform
            for T in (Float32, Float64, Int64, Int32,
                      Complex{Float32}, Complex{Float64},
                      Complex{Int64}, Complex{Int32}), d in (2, (2,2))
                A = AT{T}(undef, d)
                B = copy(A)
                rand!(A)
                rand!(B)
                @test !any(A .== B)
            end
        end
    end
end
