@testsuite "math" AT->begin
    @testcase for ET in supported_eltypes()
        # Skip complex numbers
        ET in (Complex, ComplexF32, ComplexF64) && return

        range = ET <: Integer ? (ET(-2):ET(2)) : ET
        low = ET(-1)
        high = ET(1)
        @testcase "clamp!" begin
            for N in (2, 10)
                @test compare(x -> clamp!(x, low, high), AT, rand(range, N, N))
            end
        end
    end
end
