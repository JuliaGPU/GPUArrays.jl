@testsuite "math" (AT, eltypes)->begin
    # clamp
    for ET in filter(!iscomplextype, eltypes)
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

    # power
    for ET in eltypes
        for p in 0:5
            @test compare(x->x^p, AT, rand(ET, 2,2))
        end
    end
end
