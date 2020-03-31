function test_mapreduce(AT)
    @testset "mapreducedim! $ET" for ET in supported_eltypes() begin
        T = AT{ET}
        range = ET <: Real ? (ET(1):ET(10)) : ET
        for (sz,red) in [(10,)=>(1,), (10,10)=>(1,1), (10,10,10)=>(1,1,1), (10,10,10)=>(10,10,10),
                         (10,10,10)=>(1,10,10), (10,10,10)=>(10,1,10), (10,10,10)=>(10,10,1)]
            @test compare((A,R)->Base.mapreducedim!(identity, +, R, A), AT, rand(range, sz), zeros(ET, red))
            @test compare((A,R)->Base.mapreducedim!(identity, *, R, A), AT, rand(range, sz), ones(ET, red))
            @test compare((A,R)->Base.mapreducedim!(x->x+x, +, R, A), AT, rand(range, sz), zeros(ET, red))
            return
        end

        # implicit singleton dimensions
        @test compare((A,R)->Base.mapreducedim!(identity, +, R, A), AT, rand(range, (2,2)), zeros(ET, (2,)))
        @test compare((A,R)->Base.mapreducedim!(identity, +, R, A), AT, rand(range, (2,3)), zeros(ET, (2,)))
    end
    end

    @testset "reducedim! $ET" for ET in supported_eltypes() begin
        T = AT{ET}
        range = ET <: Real ? (ET(1):ET(10)) : ET
        for (sz,red) in [(10,)=>(1,), (10,10)=>(1,1), (10,10,10)=>(1,1,1), (10,10,10)=>(10,10,10),
                         (10,10,10)=>(1,10,10), (10,10,10)=>(10,1,10), (10,10,10)=>(10,10,1)]
            @test compare((A,R)->Base.reducedim!(+, R, A), AT, rand(range, sz), zeros(ET, red))
            @test compare((A,R)->Base.reducedim!(*, R, A), AT, rand(range, sz), ones(ET, red))
        end
    end
    end

    @testset "mapreduce $ET" for ET in supported_eltypes() begin
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
    end

    @testset "reduce $ET" for ET in supported_eltypes() begin
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

    @testset "sum prod minimum maximum $ET" for ET in supported_eltypes() begin
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
        OT = isbitstype(widen(ET)) ? widen(ET) : ET
        for (sz,red) in [(10,)=>(1,), (10,10)=>(1,1), (10,10,10)=>(1,1,1), (10,10,10)=>(10,10,10),
                         (10,10,10)=>(1,10,10), (10,10,10)=>(10,1,10), (10,10,10)=>(10,10,1)]
            if !(ET <: Complex)
                @test compare((A,R)->minimum!(R, A), AT, rand(range, sz), fill(typemax(ET), red))
                @test compare((A,R)->maximum!(R, A), AT, rand(range, sz), fill(typemin(ET), red))
            end
        end
        # smaller-scale test to avoid very large values and roundoff issues
        for (sz,red) in [(2,)=>(1,), (2,2)=>(1,1), (2,2,2)=>(1,1,1), (2,2,2)=>(2,2,2),
                         (2,2,2)=>(1,2,2), (2,2,2)=>(2,1,2), (2,2,2)=>(2,2,1)]
            @test compare((A,R)->sum!(R, A), AT, rand(range, sz), zeros(OT, red))
            @test compare((A,R)->prod!(R, A), AT, rand(range, sz), ones(OT, red))
        end
    end
    end

    @testset "any all count ==" begin
        for Ac in ([false, false], [false, true], [true, true],
                   [false false; false false], [false true; false false],
                   [true true; false false], [true true; true true])
            @test compare(A->any(A), AT, Ac)
            @test compare(A->all(A), AT, Ac)
        end
        for Ac in ([1, 1], [1, 2], [2, 2],
                   [1 1; 1 1], [1 2; 1 1],
                   [2 2; 1 1], [2 2; 2 2])
            @test compare(A->any(iseven, A), AT, Ac)
            @test compare(A->all(iseven, A), AT, Ac)
            @test compare(A->count(iseven, A), AT, Ac)

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

    # old tests: can be removed, but left in here for a while to ensure the new impl works
    @testset "mapreduce" begin
        for ET in supported_eltypes()
            T = AT{ET}
            @testset "$ET" begin
                range = ET <: Integer ? (ET(-2):ET(2)) : ET
                @testset "mapreducedim" begin
                    for N in (2, 10)
                        @test compare(x -> sum(x, dims=2),      AT, rand(range, N, N))
                        @test compare(x -> sum(x, dims=1),      AT, rand(range, N, N))
                        @test compare(x -> sum(x, dims=(1, 2)), AT, rand(range, N, N))

                        @test compare(x -> sum(x, dims=2),      AT, rand(range, N, 10))
                        @test compare(x -> sum(x, dims=1),      AT, rand(range, N, 10))

                        @test compare(x -> sum(x, dims=2),      AT, rand(range, 10, N))
                        @test compare(x -> sum(x, dims=1),      AT, rand(range, 10, N))

                        _zero = zero(ET)
                        _addone(z) = z + one(z)
                        @test compare(x->mapreduce(_addone, +, x; dims = 2),
                                      AT, rand(range, N, N))
                        @test compare(x->mapreduce(_addone, +, x; dims = 2, init = _zero),
                                      AT, rand(range, N, N))

                        @test compare(x->mapreduce(+, +, x; dims = 2),
                                      AT, rand(range, N, N), rand(range, N, N))
                        @test compare(x->mapreduce(+, +, x; dims = 2, init = _zero),
                                      AT, rand(range, N, N). rand(range, N, N))
                    end
                end
                @testset "sum maximum minimum prod" begin
                    for dims in ((4048,), (1024,1024), (77,), (1923,209))
                        @test compare(sum,  AT, rand(range, dims))
                        @test compare(prod, AT, rand(range, dims))
                        @test compare(x -> sum(abs, x),  AT, rand(range, dims))
                        @test compare(x -> prod(abs, x), AT, rand(range, dims))
                        @test compare(x -> sum(abs2, x),  AT, rand(range, dims))
                        @test compare(x -> prod(abs2, x), AT, rand(range, dims))
                        ET <: Complex || @test compare(maximum, AT,rand(range, dims))
                        ET <: Complex || @test compare(minimum, AT,rand(range, dims))
                    end
                end
            end
        end
        @testset "any all ==" begin
            for Ac in ([false, false], [false, true], [true, true])
                A = AT(Ac)
                @test A isa AT{Bool,1}
                @test any(A) == any(Ac)
                @test all(A) == all(Ac)
                @test A == copy(A)
                @test A !== copy(A)
                @test A == deepcopy(A)
                @test A !== deepcopy(A)
            end
        end
    end
end
