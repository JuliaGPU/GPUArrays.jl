@testsuite "mapreduce essentials" AT->begin
    @testset "mapreducedim! $ET" for ET in supported_eltypes()
        T = AT{ET}
        range = ET <: Real ? (ET(1):ET(10)) : ET
        for (sz,red) in [(10,)=>(1,), (10,10)=>(1,1), (10,10,10)=>(1,1,1), (10,10,10)=>(10,10,10),
                         (10,10,10)=>(1,10,10), (10,10,10)=>(10,1,10), (10,10,10)=>(10,10,1)]
            @test compare((A,R)->Base.mapreducedim!(identity, +, R, A), AT, rand(range, sz), zeros(ET, red))
            @test compare((A,R)->Base.mapreducedim!(identity, *, R, A), AT, rand(range, sz), ones(ET, red))
            @test compare((A,R)->Base.mapreducedim!(x->x+x, +, R, A), AT, rand(range, sz), zeros(ET, red))
        end

        # implicit singleton dimensions
        @test compare((A,R)->Base.mapreducedim!(identity, +, R, A), AT, rand(range, (2,2)), zeros(ET, (2,)))
        @test compare((A,R)->Base.mapreducedim!(identity, +, R, A), AT, rand(range, (2,3)), zeros(ET, (2,)))
    end

    @testset "reducedim! $ET" for ET in supported_eltypes()
        T = AT{ET}
        range = ET <: Real ? (ET(1):ET(10)) : ET
        for (sz,red) in [(10,)=>(1,), (10,10)=>(1,1), (10,10,10)=>(1,1,1), (10,10,10)=>(10,10,10),
                         (10,10,10)=>(1,10,10), (10,10,10)=>(10,1,10), (10,10,10)=>(10,10,1)]
            @test compare((A,R)->Base.reducedim!(+, R, A), AT, rand(range, sz), zeros(ET, red))
            @test compare((A,R)->Base.reducedim!(*, R, A), AT, rand(range, sz), ones(ET, red))
        end
    end

    @testset "mapreduce $ET" for ET in supported_eltypes()
        T = AT{ET}
        range = ET <: Real ? (ET(1):ET(10)) : ET
        for (sz,dims) in [(10,)=>[1], (10,10)=>[1,2], (10,10,10)=>[1,2,3], (10,10,10)=>[],
                          (10,)=>:, (10,10)=>:, (10,10,10)=>:,
                          (10,10,10)=>[1], (10,10,10)=>[2], (10,10,10)=>[3]]
            @test compare(A->mapreduce(identity, +, A; dims=dims, init=zero(ET)), AT, rand(range, sz))
            @test compare(A->mapreduce(identity, *, A; dims=dims, init=one(ET)), AT, rand(range, sz))
            @test compare(A->mapreduce(x->x+x, +, A; dims=dims, init=zero(ET)), AT, rand(range, sz))
        end
    end

    @testset "reduce $ET" for ET in supported_eltypes()
        T = AT{ET}
        range = ET <: Real ? (ET(1):ET(10)) : ET
        for (sz,dims) in [(10,)=>[1], (10,10)=>[1,2], (10,10,10)=>[1,2,3], (10,10,10)=>[],
                          (10,)=>:, (10,10)=>:, (10,10,10)=>:,
                          (10,10,10)=>[1], (10,10,10)=>[2], (10,10,10)=>[3]]
            @test compare(A->reduce(+, A; dims=dims, init=zero(ET)), AT, rand(range, sz))
            @test compare(A->reduce(*, A; dims=dims, init=one(ET)), AT, rand(range, sz))
        end
    end
end

@testsuite "mapreduce derivatives" AT->begin
    @testset "sum prod minimum maximum $ET" for ET in supported_eltypes()
        T = AT{ET}
        range = ET <: Real ? (ET(1):ET(10)) : ET
        for (sz,dims) in [(10,)=>[1], (10,10)=>[1,2], (10,10,10)=>[1,2,3], (10,10,10)=>[],
                          (10,)=>:, (10,10)=>:, (10,10,10)=>:,
                          (10,10,10)=>[1], (10,10,10)=>[2], (10,10,10)=>[3]]
            @test compare(A->sum(A), AT, rand(range, sz))
            @test compare(A->sum(abs, A), AT, rand(range, sz))
            @test compare(A->sum(A; dims=dims), AT, rand(range, sz))
            @test compare(A->prod(A), AT, rand(range, sz))
            @test compare(A->prod(abs, A), AT, rand(range, sz))
            @test compare(A->prod(A; dims=dims), AT, rand(range, sz))
            if !(ET <: Complex)
                @test compare(A->minimum(A), AT, rand(range, sz))
                @test compare(A->minimum(x->x*x, A), AT, rand(range, sz))
                @test compare(A->minimum(A; dims=dims), AT, rand(range, sz))
                @test compare(A->maximum(A), AT, rand(range, sz))
                @test compare(A->maximum(x->x*x, A), AT, rand(range, sz))
                @test compare(A->maximum(A; dims=dims), AT, rand(range, sz))
            end
        end
        for (sz,red) in [(10,)=>(1,), (10,10)=>(1,1), (10,10,10)=>(1,1,1), (10,10,10)=>(10,10,10),
                         (10,10,10)=>(1,10,10), (10,10,10)=>(10,1,10), (10,10,10)=>(10,10,1)]
            if !(ET <: Complex)
                @test compare((A,R)->minimum!(R, A), AT, rand(range, sz), fill(typemax(ET), red))
                @test compare((A,R)->maximum!(R, A), AT, rand(range, sz), fill(typemin(ET), red))
            end
        end
        OT = isbitstype(widen(ET)) ? widen(ET) : ET
        if OT in supported_eltypes()
            # smaller-scale test to avoid very large values and roundoff issues
            for (sz,red) in [(2,)=>(1,), (2,2)=>(1,1), (2,2,2)=>(1,1,1), (2,2,2)=>(2,2,2),
                            (2,2,2)=>(1,2,2), (2,2,2)=>(2,1,2), (2,2,2)=>(2,2,1)]
                @test compare((A,R)->sum!(R, A), AT, rand(range, sz), rand(OT, red))
                @test compare((A,R)->prod!(R, A), AT, rand(range, sz), rand(OT, red))
            end
        end
    end

    @testset "any all count ==" begin
        for Ac in ([false, false], [false, true], [true, true],
                   [false false; false false], [false true; false false],
                   [true true; false false], [true true; true true])
            @test compare(A->any(A), AT, Ac)
            @test compare(A->all(A), AT, Ac)
            @test compare(A->count(A), AT, Ac)

            if ndims(Ac) > 1
                @test compare(A->any(A; dims=2), AT, Ac)
                @test compare(A->all(A; dims=2), AT, Ac)
                if VERSION >= v"1.5"
                    @test compare(A->count(A; dims=2), AT, Ac)
                end
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
                if VERSION >= v"1.5"
                    @test compare(A->count(iseven, A; dims=2), AT, Ac)
                end
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
end
