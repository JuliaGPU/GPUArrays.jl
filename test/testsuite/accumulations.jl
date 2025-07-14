@testsuite "accumulations" (AT, eltypes)->begin
    @testset "$ET" for ET in eltypes
        range = ET <: Real ? (ET(1):ET(10)) : ET

        # 1d arrays
        for num_elems in 1:256
            @test compare(A->accumulate(+, A; init=zero(ET)), AT, rand(range, num_elems))
        end

        for num_elems = rand(1:100, 10)
            @test compare(A->accumulate(+, A; init=zero(ET)), AT, rand(range, num_elems))
        end

        for _ in 1:10 # nd arrays reduced as 1d
            n1 = rand(1:10)
            n2 = rand(1:10)
            n3 = rand(1:10)
            @test compare(A->accumulate(+, A; init=zero(ET)), AT, rand(range, n1, n2, n3))
        end

        for num_elems = rand(1:100, 10) # init value
            init = rand(range)
            @test compare(A->accumulate(+, A; init), AT, rand(range, num_elems))
        end


        # nd arrays
        for dims in 1:4 # corner cases
            for isize in 1:3
                for jsize in 1:3
                    for ksize in 1:3
                        @test compare(A->accumulate(+, A; dims, init=zero(ET)), AT, rand(range, isize, jsize, ksize))
                    end
                end
            end
        end

        for _ in 1:10
            for dims in 1:3
                n1 = rand(1:10)
                n2 = rand(1:10)
                n3 = rand(1:10)
                @test compare(A->accumulate(+, A; dims, init=zero(ET)), AT, rand(range, n1, n2, n3))
            end
        end

        for _ in 1:10 # init value
            for dims in 1:3
                n1 = rand(1:10)
                n2 = rand(1:10)
                n3 = rand(1:10)
                init = rand(range)
                @test compare(A->accumulate(+, A; init, dims), AT, rand(range, n1, n2, n3))
            end
        end

        # Larger containers to try and detect weird bugs
        for n in (0, 1, 2, 3, 10, 10_000, 16384, 16384+1) # small, large, odd & even, pow2 and not
            # Skip large tests on small datatypes
            n >= 10000 && sizeof(real(ET)) <= 2 && continue

            @test compare(x->accumulate(+, x), AT, rand(range, n))
            @test compare(x->accumulate(+, x), AT, rand(range, n, 2))
            @test compare(Base.Fix2((x,y)->accumulate(+, x; init=y), rand(range)), AT, rand(range, n))
        end

        # in place
        @test compare(x->(accumulate!(+, x, copy(x)); x), AT, rand(range, 2))

        @test_throws ArgumentError("accumulate does not support the keyword arguments [:bad_kwarg]") accumulate(+, AT(rand(ET, 10)); bad_kwarg="bad")
    end
end

@testsuite "accumulations/cumsum & cumprod" (AT, eltypes)->begin
    @test compare(cumsum, AT, rand(Bool, 16))

    @testset "$ET" for ET in eltypes
        range = ET <: Real ? (ET(1):ET(10)) : ET

        # cumsum
        for num_elems in rand(1:100, 10)
            @test compare(A->cumsum(A; dims=1), AT, rand(range, num_elems))
        end

        for _ in 1:10
            for dims in 1:3
                n1 = rand(1:10)
                n2 = rand(1:10)
                n3 = rand(1:10)
                @test compare(A->cumsum(A; dims), AT, rand(range, n1, n2, n3))
            end
        end


        # cumprod
        range = ET <: Real ? (ET(1):ET(10)) : ET
        @test compare(A->cumprod(A; dims=1), AT, ones(ET, 100_000))

        for _ in 1:10
            for dims in 1:3
                n1 = rand(1:10)
                n2 = rand(1:10)
                n3 = rand(1:10)
                @test compare(A->cumprod(A; dims), AT, rand(range, n1, n2, n3))
            end
        end
    end
end
