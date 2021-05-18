@testsuite "math/intrinsics" AT->begin
    for ET in supported_eltypes()
        # Skip complex numbers
        ET in (Complex, ComplexF32, ComplexF64) && continue

        T = AT{ET}
        @testset "$ET" begin
            range = ET <: Integer ? (ET(-2):ET(2)) : ET
            low = ET(-1)
            high = ET(1)
            @testset "clamp!" begin
                for N in (2, 10)
                    @test compare(x -> clamp!(x, low, high), AT, rand(range, N, N))
                end
            end
        end
    end
end

@testsuite "math/power" AT->begin
    for ET in supported_eltypes()
        for p in 0:5
            compare(x->x^p, AT, rand(ET, 2,2))
        end
    end
end
