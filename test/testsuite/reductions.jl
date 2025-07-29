@testsuite "reductions/mapreducedim!" (AT, eltypes)->begin
    @testset "$ET" for ET in eltypes
        range = ET <: Real ? (ET(1):ET(10)) : ET
        for (sz,red) in [(10,)=>(1,), (10,10)=>(1,1), (10,10,10)=>(1,1,1), (10,10,10)=>(10,10,10),
                         (10,10,10)=>(1,10,10), (10,10,10)=>(10,1,10), (10,10,10)=>(10,10,1),
                         (0,)=>(1,)]
            @test compare((A,R)->Base.mapreducedim!(identity, +, R, A), AT, rand(range, sz), zeros(ET, red))
            @test compare((A,R)->Base.mapreducedim!(identity, *, R, A), AT, rand(range, sz), ones(ET, red))
            @test compare((A,R)->Base.mapreducedim!(x->x+x, +, R, A), AT, rand(range, sz), zeros(ET, red))
        end

        # implicit singleton dimensions
        @test compare((A,R)->Base.mapreducedim!(identity, +, R, A), AT, rand(range, (2,2)), zeros(ET, (2,)))
        @test compare((A,R)->Base.mapreducedim!(identity, +, R, A), AT, rand(range, (2,3)), zeros(ET, (2,)))
    end
end

@testsuite "reductions/mapreducedim!_large" (AT, eltypes)->begin
    @testset "$ET" for ET in eltypes
        # Skip smaller floating types due to precision issues
        if ET in (Float16, ComplexF16)
            continue
        end

        range = ET <: Real ? (ET(1):ET(10)) : ET
        # Reduce larger array sizes to test multiple-element reading in certain implementations
        for (sz,red) in [(1000000,)=>(1,), (5000,500)=>(1,1), (500,5000)=>(1,1),
                         (500,5000)=>(500,1), (5000,500)=>(1,500)]
            @test compare((A,R)->Base.mapreducedim!(identity, +, R, A), AT, rand(range, sz), zeros(ET, red))
        end
    end
end

@testsuite "reductions/reducedim!" (AT, eltypes)->begin
    @testset "$ET" for ET in eltypes
        range = ET <: Real ? (ET(1):ET(10)) : ET
        for (sz,red) in [(10,)=>(1,), (10,10)=>(1,1), (10,10,10)=>(1,1,1), (10,10,10)=>(10,10,10),
                         (10,10,10)=>(1,10,10), (10,10,10)=>(10,1,10), (10,10,10)=>(10,10,1),
                         (0,)=>(1,)]
            @test compare((A,R)->Base.reducedim!(+, R, A), AT, rand(range, sz), zeros(ET, red))
            @test compare((A,R)->Base.reducedim!(*, R, A), AT, rand(range, sz), ones(ET, red))
        end
    end
end

@testsuite "reductions/mapreduce" (AT, eltypes)->begin
    @testset "$ET" for ET in eltypes
        range = ET <: Real ? (ET(1):ET(10)) : ET
        for (sz,dims) in [(10,)=>[1], (10,10)=>[1,2], (10,10,10)=>[1,2,3], (10,10,10)=>[],
                          (10,)=>:, (10,10)=>:, (10,10,10)=>:,
                          (10,10,10)=>[1], (10,10,10)=>[2], (10,10,10)=>[3],
                          (0,)=>[1]]
            @test compare(A->mapreduce(identity, +, A; dims=dims, init=zero(ET)), AT, rand(range, sz))
            @test compare(A->mapreduce(identity, *, A; dims=dims, init=one(ET)), AT, rand(range, sz))
            @test compare(A->mapreduce(x->x+x, +, A; dims=dims, init=zero(ET)), AT, rand(range, sz))
        end
    end
    # Test more corner cases. Tests from AcceleraterKernels.jl
    for dims in [1,2,3,4,[1,2],[1,3],[1,4],[2,3],[2,4],[3,4],[1,2,3],[1,2,4],[1,3,4],[2,3,4],[1,2,3,4]]
        for isize in 0:3
            for jsize in 0:3
                for ksize in 0:3
                    @test compare(A->mapreduce(x->x+x, +, A; init=zero(Int32), dims), AT, rand(Int32(1):Int32(10), isize, jsize, ksize))
                end
            end
        end
    end
end

@testsuite "reductions/reduce" (AT, eltypes)->begin
    @testset "$ET" for ET in eltypes
        range = ET <: Real ? (ET(1):ET(10)) : ET
        for (sz,dims) in [(10,)=>[1], (10,10)=>[1,2], (10,10,10)=>[1,2,3], (10,10,10)=>[],
                          (10,)=>:, (10,10)=>:, (10,10,10)=>:,
                          (10,10,10)=>[1], (10,10,10)=>[2], (10,10,10)=>[3],
                          (0,)=>[1]]
            @test compare(A->reduce(+, A; dims=dims, init=zero(ET)), AT, rand(range, sz))
            @test compare(A->reduce(*, A; dims=dims, init=one(ET)), AT, rand(range, sz))
            if ET <: Integer
                @test compare(A->reduce(&, A; dims=dims, init=~zero(ET)), AT, rand(range, sz))
                @test compare(A->reduce(|, A; dims=dims, init=zero(ET)), AT, rand(range, sz))
                @test compare(A->reduce(⊻, A; dims=dims, init=zero(ET)), AT, rand(range, sz))
            end
        end
    end
    # Test more corner cases. Tests from AcceleraterKernels.jl
    for dims in [1,2,3,4,[1,2],[1,3],[1,4],[2,3],[2,4],[3,4],[1,2,3],[1,2,4],[1,3,4],[2,3,4],[1,2,3,4]]
        for isize in 0:3
            for jsize in 0:3
                for ksize in 0:3
                    @test compare(A->reduce(+, A; init=zero(Int32), dims), AT, rand(Int32(1):Int32(10), isize, jsize, ksize))
                end
            end
        end
    end
end

@testsuite "reductions/sum prod" (AT, eltypes)->begin
    @testset "$ET" for ET in eltypes
        range = ET <: Real ? (ET(1):ET(10)) : ET
        for (sz,dims) in [(10,)=>[1], (10,10)=>[1,2], (10,10,10)=>[1,2,3], (10,10,10)=>[],
                            (10,)=>:, (10,10)=>:, (10,10,10)=>:,
                            (10,10,10)=>[1], (10,10,10)=>[2], (10,10,10)=>[3],
                            (0,)=>[1]]
            @test compare(A->sum(A), AT, rand(range, sz))
            @test compare(A->sum(A; dims=dims), AT, rand(range, sz))
            @test compare(A->prod(A), AT, rand(range, sz))
            @test compare(A->prod(A; dims=dims), AT, rand(range, sz))
            if typeof(abs(rand(range))) in eltypes
                # abs(::Complex{Int}) promotes to Float64
                @test compare(A->sum(abs, A), AT, rand(range, sz))
                @test compare(A->prod(abs, A), AT, rand(range, sz))
            end
        end

        if ET in (Float32, Float64, Int64, ComplexF32, ComplexF64)
            # smaller-scale test to avoid very large values and roundoff issues
            for (sz,red) in [(2,)=>(1,), (2,2)=>(1,1), (2,2,2)=>(1,1,1), (2,2,2)=>(2,2,2),
                                (2,2,2)=>(1,2,2), (2,2,2)=>(2,1,2), (2,2,2)=>(2,2,1)]
                @test compare((A,R)->sum!(R, A), AT, rand(range, sz), rand(ET, red))
                @test compare((A,R)->prod!(R, A), AT, rand(range, sz), rand(ET, red))
            end
        end
    end
end

@testsuite "reductions/minimum maximum extrema" (AT, eltypes)->begin
    @testset "$ET" for ET in eltypes
        range = ET <: Real ? (ET(1):ET(10)) : ET
        for (sz,dims) in [(10,)=>[1], (10,10)=>[1,2], (10,10,10)=>[1,2,3], (10,10,10)=>[],
                          (10,)=>:, (10,10)=>:, (10,10,10)=>:,
                          (10,10,10)=>[1], (10,10,10)=>[2], (10,10,10)=>[3]]
            if !(ET <: Complex)
                @test compare(A->minimum(A), AT, rand(range, sz))
                @test compare(A->minimum(x->x*x, A), AT, rand(range, sz))
                @test compare(A->minimum(A; dims=dims), AT, rand(range, sz))
                @test compare(A->maximum(A), AT, rand(range, sz))
                @test compare(A->maximum(x->x*x, A), AT, rand(range, sz))
                @test compare(A->maximum(A; dims=dims), AT, rand(range, sz))
                @test compare(A->extrema(A), AT, rand(range, sz))
                @test compare(A->extrema(x->x*x, A), AT, rand(range, sz))
                @test compare(A->extrema(A; dims=dims), AT, rand(range, sz))
            end
        end

        for (sz,red) in [(10,)=>(1,), (10,10)=>(1,1), (10,10,10)=>(1,1,1), (10,10,10)=>(10,10,10),
                         (10,10,10)=>(1,10,10), (10,10,10)=>(10,1,10), (10,10,10)=>(10,10,1)]
            if !(ET <: Complex)
                @test compare((A,R)->minimum!(R, A), AT, rand(range, sz), fill(typemax(ET), red))
                @test compare((A,R)->maximum!(R, A), AT, rand(range, sz), fill(typemin(ET), red))
                @test compare((A,R)->extrema!(R, A), AT, rand(range, sz), fill((typemax(ET),typemin(ET)), red))
            end
        end
    end
end

@testsuite "reductions/any all count" (AT, eltypes)->begin
    for Ac in ([false, false], [false, true], [true, true],
                [false false; false false], [false true; false false],
                [true true; false false], [true true; true true])
        @test compare(A->any(A), AT, Ac)
        @test compare(A->all(A), AT, Ac)
        @test compare(A->count(A), AT, Ac)

        if ndims(Ac) > 1
            @test compare(A->any(A; dims=2), AT, Ac)
            @test compare(A->all(A; dims=2), AT, Ac)
            @test compare(A->count(A; dims=2), AT, Ac)
        end
    end
    for Ac in ([1, 1], [1, 2], [2, 2],
                [1 1; 1 1], [1 2; 1 1],
                [2 2; 1 1], [2 2; 2 2])
        @test compare(A->any(iseven, A), AT, Ac)
        @test compare(A->all(iseven, A), AT, Ac)
        @test compare(A->count(iseven, A), AT, Ac)

        if ndims(Ac) > 1
            @test compare(A->any(iseven, A; dims=2), AT, Ac)
            @test compare(A->all(iseven, A; dims=2), AT, Ac)
            @test compare(A->count(iseven, A; dims=2), AT, Ac)
        end

        A = AT(Ac)
        @test A == copy(A)
        @test A !== copy(A)
        @test A == deepcopy(A)
        @test A !== deepcopy(A)

        B = similar(A)
        @allowscalar B[1] = 3
        @test A != B
    end
end

@testsuite "reductions/== isequal" (AT, eltypes)->begin
    @testset "$ET" for ET in eltypes
        range = ET <: Real ? (ET(1):ET(10)) : ET

        # different sizes should trip up both (CUDA.jl#1524)
        @test compare((A, B) -> A == B, AT, rand(range, (2,3)), rand(range, 6))
        @test compare((A, B) -> isequal(A, B), AT, rand(range, (2,3)), rand(range, 6))

        # equal sizes depend on values
        for sz in [(10,), (10,10), (10,10,10), (0,)]
            @test compare((A, B) -> A == B, AT, rand(range, sz), rand(range, sz))
            @test compare((A, B) -> isequal(A, B), AT, rand(range, sz), rand(range, sz))
            Ac = rand(range, sz)
            @test compare((A, B) -> A == B, AT, Ac, Ac)
            @test compare((A, B) -> isequal(A, B), AT, Ac, Ac)
            if isfloattype(ET) && length(Ac) > 0
                # Test cases where == and isequal behave differently
                Bc = copy(Ac)
                # 0.0 == -0.0 but !isequal(0.0, -0.0)
                Ac[1] = zero(ET)
                Bc[1] = -zero(ET)
                @test compare((A, B) -> A == B, AT, Ac, Bc)
                @test compare((A, B) -> isequal(A, B), AT, Ac, Bc)
                # NaN != NaN but isequal(NaN, NaN)
                Ac[1] = Bc[1] = ET(NaN)
                @test compare((A, B) -> A == B, AT, Ac, Bc)
                @test compare((A, B) -> isequal(A, B), AT, Ac, Bc)
            end
        end
    end

    # missing values should only trip up ==
    @test compare((A, B) -> A == B, AT, [missing], [missing])
    @test compare((A, B) -> isequal(A, B), AT, [missing], [missing])
end
