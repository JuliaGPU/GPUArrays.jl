@testsuite "math" (AT, eltypes)->begin
    @testset "$ET" for ET in eltypes
        @testset "power" begin
            for p in [0,1,2,5]
                @test compare(x->x^p, AT, rand(ET, 2,2))
            end
        end

        iscomplextype(ET) && continue
        @testset "clamp!" begin
            range = ET <: Integer ? (ET(-2):ET(2)) : ET
            low = ET(-1)
            high = ET(1)
            for N in (2, 10)
                @test compare(x -> clamp!(x, low, high), AT, rand(range, N, N))
            end
        end
    end
end
