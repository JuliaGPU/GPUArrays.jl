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
                if VERSION >= v"1.8.0-DEV.1465"
                    @test compare(A->extrema(A), AT, rand(range, sz))
                    @test compare(A->extrema(x->x*x, A), AT, rand(range, sz))
                    @test compare(A->extrema(A; dims=dims), AT, rand(range, sz))
                end
            end
        end

        for (sz,red) in [(10,)=>(1,), (10,10)=>(1,1), (10,10,10)=>(1,1,1), (10,10,10)=>(10,10,10),
                         (10,10,10)=>(1,10,10), (10,10,10)=>(10,1,10), (10,10,10)=>(10,10,1)]
            if !(ET <: Complex)
                @test compare((A,R)->minimum!(R, A), AT, rand(range, sz), fill(typemax(ET), red))
                @test compare((A,R)->maximum!(R, A), AT, rand(range, sz), fill(typemin(ET), red))
                if VERSION >= v"1.8.0-DEV.1465"
                    @test compare((A,R)->extrema!(R, A), AT, rand(range, sz), fill((typemax(ET),typemin(ET)), red))
                end
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
