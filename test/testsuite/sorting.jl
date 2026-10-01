@testsuite "sorting/sort" (AT, eltypes)->begin
    @testset "$ET" for ET in eltypes
        ET <: Real || continue      # only orderable element types

        range = ET <: AbstractFloat ? ET : (ET(1):ET(100))

        # flat 1-D sort, in- and out-of-place, forward and reverse
        for n in (0, 1, 2, 10, 1000, 100000)
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

        # views and reshaped arrays
        @test compare(A -> (sort!(view(A, 11:90)); A), AT, rand(range, 100))
        @test compare(A -> (sort!(view(A, 2:9, :); dims=1); A), AT, rand(range, 10, 10))
        @test compare(A -> sort(reshape(A, 10, 10); dims=2), AT, rand(range, 100))
    end
end

@testsuite "sorting/algorithms" (AT, eltypes)->begin
    x = rand(Float32, 1000)

    # Base's algorithms are requirements on the algorithm the implementation picks
    for alg in (QuickSort, MergeSort, InsertionSort, Base.Sort.DEFAULT_STABLE,
                Base.Sort.DEFAULT_UNSTABLE)
        @test compare(A -> sort(A; alg), AT, x)
    end
    @test compare(A -> Array(sort(A; alg=PartialQuickSort(1:10)))[1:10], AT, x)
    if AT <: AbstractGPUArray   # (Base supports every algorithm of its own, and no others)
        @test_throws ArgumentError sort(AT(x); alg=Base.Sort.ScratchQuickSort())
        @test_throws ArgumentError sortperm(AT(x); alg=Base.Sort.ScratchQuickSort())
        # ... or AcceleratedKernels' own
        @test Array(sort(AT(x); alg=GPUArrays.AK.MergeSort())) == sort(x)
        @test Array(sortperm(AT(x); alg=GPUArrays.AK.MergeSort())) == sortperm(x)
    end
    # `scratch` is accepted
    @test compare(A -> sort(A; scratch=nothing), AT, x)

    # Stable by default, as Base: tagged ties keep their order
    tagged = [(rand(1:3), i) for i in 1:1000]
    @test compare(A -> sort(A; by=first), AT, tagged)
    @test compare(A -> sort(A; by=first, alg=MergeSort), AT, tagged)
    @test compare(A -> sortperm(A; by=first), AT, tagged)

    # Arrays other than vectors need `dims`, and vectors take none, as in Base
    @test_throws MethodError sort(AT(x); dims=1)
    @test_throws MethodError sort!(AT(x); dims=1)
    @test_throws UndefKeywordError sort(AT(rand(Float32, 4, 4)))
    @test_throws UndefKeywordError sort!(AT(rand(Float32, 4, 4)))
    @test_throws UndefKeywordError sortperm(AT(rand(Float32, 4, 4)))
end

@testsuite "sorting/sortperm" (AT, eltypes)->begin
    @testset "$ET" for ET in eltypes
        ET <: Real || continue

        range = ET <: AbstractFloat ? ET : (ET(1):ET(100))

        for n in (1, 2, 10, 1000)
            @test compare(A -> sortperm(A), AT, rand(range, n))
            @test compare(A -> sortperm(A; rev=true), AT, rand(range, n))
        end

        # along a dimension, with linear indices as in Base
        for dims in (1, 2)
            @test compare(A -> sortperm(A; dims), AT, rand(range, 20, 30))
            @test compare((ix, A) -> sortperm!(ix, A; dims), AT, zeros(Int, 20, 30), rand(range, 20, 30))
        end
        # into an existing index array, and through views
        @test compare((ix, A) -> sortperm!(ix, A), AT, zeros(Int, 100), rand(range, 100))
        @test compare((ix, A) -> sortperm!(ix, view(A, 1:50)), AT, zeros(Int, 50), rand(range, 100))
    end

    # Base's rules: vectors take no `dims`, and the index array must match
    x = AT(rand(Float32, 10))
    @test_throws ArgumentError sortperm!(similar(x, Int), x; dims=1)
    @test_throws ArgumentError sortperm!(similar(x, Int, 11), x)
    @test_throws ArgumentError sortperm!(similar(x, Int, 4, 5), AT(rand(Float32, 5, 4)); dims=1)
end

@testsuite "sorting/partialsort" (AT, eltypes)->begin
    N = 10000
    @testset "$ET" for ET in eltypes
        ET <: Real || continue
        range = ET <: AbstractFloat ? ET : (ET(1):ET(100))

        @test compare(A -> partialsort!(A, 1), AT, rand(range, N))
        @test compare(A -> partialsort!(A, N), AT, rand(range, N))
        @test compare(A -> partialsort!(A, N ÷ 2; rev=true), AT, rand(range, N))
        @test compare(A -> partialsort!(A, (N ÷ 10):(2N ÷ 10)), AT, rand(range, N))
        @test compare(A -> partialsort(A, N ÷ 2), AT, rand(range, N))
    end

    # As Base: an element for an integer, a view for a range
    x = AT(rand(Float32, 100))
    @test partialsort!(copy(x), 3) isa Float32
    y = copy(x)
    r = partialsort!(y, 3:5)
    fill!(r, 0)
    @test all(iszero, Array(y)[3:5])
    @test Array(partialsort(x, 3:5)) == partialsort(Array(x), 3:5)
    # ... also with a positional ordering
    @test compare(A -> partialsort!(A, 3, Base.Order.Reverse), AT, rand(Float32, 100))
end

@testsuite "reverse" (AT, eltypes)->begin
    @testset "$ET" for ET in eltypes
        # vectors: whole, and ranged
        for n in (0, 1, 2, 7, 1000)
            @test compare(A -> reverse(A), AT, rand(ET, n))
            @test compare(A -> reverse!(A), AT, rand(ET, n))
        end
        @test compare(A -> reverse(A; dims=1), AT, rand(ET, 10))
        @test compare(A -> reverse!(A, 3), AT, rand(ET, 10))
        @test compare(A -> reverse!(A, 3, 8), AT, rand(ET, 10))
        @test compare(A -> reverse(A, 3, 8), AT, rand(ET, 10))

        # along dimensions
        for dims in (1, 2, (1, 2), :)
            @test compare(A -> reverse(A; dims), AT, rand(ET, 5, 6))
            @test compare(A -> reverse!(A; dims), AT, rand(ET, 5, 6))
        end
        @test compare(A -> reverse!(A; dims=(1, 3)), AT, rand(ET, 3, 4, 5))

        # views and reshaped arrays
        @test compare(A -> (reverse!(view(A, 2:9)); A), AT, rand(ET, 10))
        @test compare(A -> reverse(reshape(A, 4, 5); dims=2), AT, rand(ET, 20))
    end

    # As Base: a trivial interval is a no-op, even out of bounds; others are checked
    x = AT(collect(1:10))
    @test Array(reverse!(copy(x), 7, 6)) == 1:10
    @test Array(reverse!(copy(x), 12, 11)) == 1:10
    @test_throws BoundsError reverse!(copy(x), 0, 3)
    @test_throws BoundsError reverse!(copy(x), 5, 11)
    @test_throws ArgumentError reverse(x; dims=2)
    @test_throws ArgumentError reverse(AT(rand(Float32, 3, 3)); dims=3)
end
