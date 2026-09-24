@testsuite "findall" (AT, eltypes)->begin
    # indices as in Base: `Int` for vectors, `CartesianIndex` otherwise
    # (`Array`, since `CartesianIndex` results are compared elementwise)
    for sz in ((0,), (1,), (1000,), (10, 20), (4, 5, 6))
        @test compare(A -> Array(findall(A)), AT, rand(Bool, sz))
        @test compare(A -> Array(findall(x -> x > 0.5f0, A)), AT, rand(Float32, sz))
    end
    # ... also for 0-d arrays, where only the predicate form gives linear indices
    @test compare(A -> Array(findall(A)), AT, fill(true))
    @test compare(A -> Array(findall(identity, A)), AT, fill(true))
    @test compare(A -> Array(findall(A)), AT, fill(false))

    # views and reshaped arrays
    @test compare(A -> findall(view(A, 3:90)), AT, rand(Bool, 100))
    @test compare(A -> Array(findall(isodd, reshape(A, 10, 10))), AT, rand(1:10, 100))

    @test compare(A -> findall(in((2, 3)), A), AT, rand(1:5, 100))

    # the predicate must return a Bool, as in Base
    @test_throws Union{TypeError, ArgumentError} findall(x -> 1, AT([1, 2]))

    # Base's index types exactly (AcceleratedKernels selects the items GPUArrays passes)
    for x in (rand(Bool, 100), rand(Bool, 10, 20), fill(true), fill(false), Bool[])
        @test compare_exact(findall, AT, x)
        @test compare_exact(A -> findall(!, A), AT, x)
    end
    @test compare_exact(A -> findall(isodd, view(A, 2:2:10, :)), AT, rand(1:9, 10, 3))
    for x in (fill(2), fill(5), rand(1:5, 100), rand(1:5, 4, 5))
        @test compare_exact(A -> findall(in((2, 3)), A), AT, x)
    end
end

@testsuite "indexing logical" (AT, eltypes)->begin
    # a mask on the device or on the host
    @test compare((A, m) -> A[m], AT, rand(Float32, 100), rand(Bool, 100))
    @test compare(A -> A[Array(A) .> 0.5f0], AT, rand(Float32, 100))
    @test compare((A, m) -> A[m], AT, rand(Float32, 10, 10), rand(Bool, 10, 10))
    # mixed with other indices
    @test compare((A, m) -> A[m, :], AT, rand(Float32, 10, 5), rand(Bool, 10))
    @test compare((A, m) -> A[:, m], AT, rand(Float32, 5, 10), rand(Bool, 10))
    # assignment
    @test compare((A, m) -> (A[m] .= 0; A), AT, rand(Float32, 100), rand(Bool, 100))
    # views and reshaped arrays
    @test compare((A, m) -> view(A, 1:50)[m], AT, rand(Float32, 100), rand(Bool, 50))
    @test compare((A, m) -> reshape(A, 10, 10)[m], AT, rand(Float32, 100), rand(Bool, 10, 10))
    # the selected values and their type, from masks of the array's shape or of another one
    for (a, m) in ((rand(Float32, 100), rand(Bool, 100)), (rand(Int8, 10, 10), rand(Bool, 10, 10)),
                   (rand(Float32, 10, 10), rand(Bool, 100)), (rand(Float32, 0), Bool[]),
                   (rand(Float32, 4, 5), falses(4, 5)))
        @test compare_exact((A, m) -> A[m], AT, a, m)
    end
    @test compare_exact((A, m) -> view(A, 1:5, :)[m], AT, rand(Float32, 10, 4), rand(Bool, 5, 4))
    # ... and a mask that does not fit the array
    for (a, m) in ((rand(Float32, 3), rand(Bool, 2)), (rand(Float32, 2, 3), rand(Bool, 3, 2)))
        @test compare_exact((A, m) -> A[m], AT, a, m)
        @test compare_exact(A -> A[m], AT, a)                   # (a host mask)
        @test compare_exact(A -> A[view(m, :)], AT, a)          # (a host view as the mask)
    end
    @test compare_exact((A, m) -> view(A, 1:2:5)[m], AT, rand(Float32, 5), rand(Bool, 2))
end

@testsuite "reductions/any all predicates" (AT, eltypes)->begin
    for sz in ((0,), (1,), (1000,), (10, 20))
        @test compare(A -> any(A), AT, rand(Bool, sz))
        @test compare(A -> all(A), AT, rand(Bool, sz))
        @test compare(A -> any(x -> x > 0.9f0, A), AT, rand(Float32, sz))
        @test compare(A -> all(x -> x > 0.1f0, A), AT, rand(Float32, sz))
    end
    @test compare(A -> all(A), AT, trues(1000))
    @test compare(A -> any(A), AT, falses(1000))

    # along dimensions
    for dims in (1, 2, (1, 2))
        @test compare(A -> any(A; dims), AT, rand(Bool, 10, 20))
        @test compare(A -> all(x -> x > 0.1f0, A; dims), AT, rand(Float32, 10, 20))
    end

    # views and reshaped arrays
    @test compare(A -> any(x -> x > 0.9f0, view(A, 2:90)), AT, rand(Float32, 100))
    @test compare(A -> all(x -> x > 0.1f0, reshape(A, 10, 10); dims=2), AT, rand(Float32, 100))

    # the predicate must return a Bool (or missing, without `dims`), as in Base, but is never
    # called on an empty array
    @test_throws Union{TypeError, ArgumentError} any(x -> 1, AT([1, 2]))
    @test_throws Union{TypeError, ArgumentError} all(x -> 1, AT([1, 2]))
    if AT <: AbstractGPUArray   # (Julia 1.10's Base accepts this)
        @test_throws Union{TypeError, ArgumentError} any(x -> 1, AT([1 2; 3 4]); dims=1)
    end
    @test any(x -> 1, AT(Int[])) === false
    @test all(x -> 1, AT(Int[])) === true
    @test compare(A -> any(x -> 1, A; dims=1), AT, zeros(Int, 0, 3))
    for dims in (0, -1, 1.5, (1, 0)), f in (any, all)   # Base's checks of `dims`, also when empty
        @test compare_exact(A -> f(A; dims), AT, zeros(Bool, 0, 3))
    end
    @test compare(A -> all(x -> missing, A; dims=1), AT, zeros(Int, 0, 3))

    # three-valued logic with missing values
    for (f, xs) in ((x -> x > 1 ? missing : false, [1, 2]), (x -> x > 1 ? missing : true, [1, 2]),
                    (x -> x > 1 ? missing : x == 1, [1, 2, 3]), (x -> missing, [1, 2]))
        @test isequal(any(f, AT(xs)), any(f, xs))
        @test isequal(all(f, AT(xs)), all(f, xs))
    end
    # ... also stored in the array, where the back-end supports isbits unions
    supports_unions = try
        Array(AT(Union{Missing,Bool}[missing, true]) .| false)
        true
    catch
        false
    end
    if supports_unions
        for (a, b) in ((missing, true), (missing, false), (true, true))
            @test isequal(any(AT(Union{Missing,Bool}[a, b])), any([a, b]))
            @test isequal(all(AT(Union{Missing,Bool}[a, b])), all([a, b]))
        end
    end
end
