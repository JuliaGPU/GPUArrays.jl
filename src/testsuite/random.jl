function test_random(AT)
    @testset "Random" begin
        @testset "rand" begin  # uniform
            for T in (Float32, Float64, Int64, Int32), d in (2, (2,2))
                A = AT{T}(d)
                B = copy(A)
                rand!(A)
                rand!(B)
                @test !any(A .== B)
            end
        end
    end
end
