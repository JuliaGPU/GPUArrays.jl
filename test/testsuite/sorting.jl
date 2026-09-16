@testsuite "sorting/sort" (AT, eltypes)->begin
    @testset "$ET" for ET in eltypes
        ET <: Real || continue      # only orderable element types

        range = ET <: AbstractFloat ? ET : (ET(1):ET(100))

        # flat 1-D sort, in- and out-of-place, forward and reverse
        for n in (0, 1, 2, 10, 1000, 10000)
            @test compare(A -> sort(A), AT, rand(range, n))
            @test compare(A -> sort(A; rev=true), AT, rand(range, n))
            @test compare(A -> sort!(copy(A)), AT, rand(range, n))
        end

        # by / order
        @test compare(A -> sort(A; by=abs), AT, rand(range, 1000))
        @test compare(A -> sort(A; order=Base.Order.Reverse), AT, rand(range, 1000))

        # heavy ties
        @test compare(A -> sort(A), AT, rand(ET(1):ET(3), 1000))

        # per-slice sort along a dimension
        for dims in (1, 2)
            @test compare(A -> sort(A; dims), AT, rand(range, 100, 50))
            @test compare(A -> sort(A; dims, rev=true), AT, rand(range, 100, 50))
        end
        @test compare(A -> sort(A; dims=2), AT, rand(range, 8, 16, 4))
    end
end

@testsuite "sorting/sortperm" (AT, eltypes)->begin
    @testset "$ET" for ET in eltypes
        ET <: Real || continue

        range = ET <: AbstractFloat ? ET : (ET(1):ET(100))

        for n in (1, 2, 10, 1000)
            @test compare(A -> sortperm(A), AT, rand(range, n))
            @test compare(A -> sortperm(A; rev=true), AT, rand(range, n))
        end
    end
end
