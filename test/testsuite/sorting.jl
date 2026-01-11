@testsuite "sorting/sort" (AT, eltypes)->begin
    # Fuzzy correctness testing
    @testset "$ET" for ET in filter(x -> x <: Real, eltypes)
        for _ in 1:10
            num_elems = rand(1:100_000)
            @test compare((A)->Base.sort!(A), AT, rand(ET, num_elems))
        end
        # Not yet implemented
        # for _ in 1:5
        #     size = rand(1:100, 2)
        #     @test compare((A)->Base.sort!(A; dims=1), AT, rand(ET, size...))
        #     @test compare((A)->Base.sort!(A; dims=2), AT, rand(ET, size...))
        # end
    end
end

@testsuite "sorting/sortperm" (AT, eltypes)->begin
    # Fuzzy correctness testing
    @testset "$ET" for ET in filter(x -> x <: Real, eltypes)
        for _ in 1:10
            num_elems = rand(1:100_000)
            @test compare((ix, A)->Base.sortperm!(ix, A), AT, zeros(Int32, num_elems), rand(ET, num_elems))
        end
        # Not yet implemented
        # for _ in 1:5
        #     size = rand(1:100, 2)
        #     @test compare((A)->Base.sort!(A; dims=1), AT, zeros(Int32, size...), rand(ET, size...))
        #     @test compare((A)->Base.sort!(A; dims=2), AT, zeros(Int32, size...), rand(ET, size...))
        # end
    end
end

@testsuite "sorting/partialsort" (AT, eltypes)->begin
    local N = 10000
    @testset "$ET" for ET in filter(x -> x <: Real, eltypes)
        @test compare((A)->Base.partialsort!(A, 1), AT, rand(ET, N))
        @test compare((A)->Base.partialsort!(A, 1; rev=true), AT, rand(ET, N))

        @test compare((A)->Base.partialsort!(A, N), AT, rand(ET, N))
        @test compare((A)->Base.partialsort!(A, N; rev=true), AT, rand(ET, N))

        @test compare((A)->Base.partialsort!(A, N÷2), AT, rand(ET, N))
        @test compare((A)->Base.partialsort!(A, N÷2; rev=true), AT, rand(ET, N))

        @test compare((A)->Base.partialsort!(A, (N÷10):(2N÷10)), AT, rand(ET, N))
        @test compare((A)->Base.partialsort!(A, (N÷10):(2N÷10); rev=true), AT, rand(ET, N))

        @test compare((A)->Base.partialsort!(A, 1:N), AT, rand(ET, N))
        @test compare((A)->Base.partialsort!(A, 1:N; rev=true), AT, rand(ET, N))
    end
end
