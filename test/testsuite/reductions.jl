
@testsuite "reductions/mapreducedim!_large" (AT, eltypes)->begin
    @testset "$ET" for ET in eltypes
        # Skip smaller floating types due to precision issues
        if ET in (Float16, ComplexF16)
            continue
        end

        range = ET <: Real ? (ET(1):ET(10)) : ET
        # Reduce larger array sizes to test multiple-element reading in certain implementations
        for (sz,red) in [(500000,)=>(1,), (1000,500)=>(1,1), (500,1000)=>(1,1),
                         (500,1000)=>(500,1), (1000,500)=>(1,500)]
            @test compare((A,R)->Base.mapreducedim!(identity, +, R, A), AT, rand(range, sz), zeros(ET, red))
        end
    end
end

@testsuite "reductions/mapreducedim!" (AT, eltypes)->begin
    @testset "$ET" for ET in eltypes
        range = ET <: Real ? (ET(1):ET(10)) : ET
        for (sz,red) in [(10,)=>(1,), (10,10)=>(1,1), (10,10,10)=>(1,1,1), (10,10,10)=>(10,10,10),
                         (10,10,10)=>(1,10,10), (10,10,10)=>(10,1,10), (10,10,10)=>(10,10,1),
                         (0,)=>(1,)]
            # mapreducedim!
            @test compare((A,R)->Base.mapreducedim!(identity, +, R, A), AT, rand(range, sz), zeros(ET, red))
            @test compare((A,R)->Base.mapreducedim!(identity, *, R, A), AT, rand(range, sz), ones(ET, red))
            @test compare((A,R)->Base.mapreducedim!(x->x+x, +, R, A), AT, rand(range, sz), zeros(ET, red))

            # reducedim!
            @test compare((A,R)->Base.reducedim!(+, R, A), AT, rand(range, sz), zeros(ET, red))
            @test compare((A,R)->Base.reducedim!(*, R, A), AT, rand(range, sz), ones(ET, red))
        end

        # implicit singleton dimensions
        @test compare((A,R)->Base.mapreducedim!(identity, +, R, A), AT, rand(range, (2,2)), zeros(ET, (2,)))
        @test compare((A,R)->Base.mapreducedim!(identity, +, R, A), AT, rand(range, (2,3)), zeros(ET, (2,)))

        # mapreducedim! into wrapper types
        for t in [transpose, adjoint]
            @test compare((A,R)->Base.mapreducedim!(identity, +, R, A), AT, rand(range, (2,2)), t(zeros(ET, (2,))))
            @test compare((A,R)->Base.mapreducedim!(abs2, +, R, A), AT, rand(range, (3,2,10)), t(zeros(ET, (2,3))))

            # TODO: reenable once https://github.com/JuliaGPU/Metal.jl/issues/907 is fixed
            # @test compare((A,R)->sum!(abs2, R, A), AT, rand(range, (3,2,10)), t(zeros(ET, (2,3))))

            A, R = AT(rand(range, (3,2,10))), t(AT(zeros(ET, (2,3))))
            @test Base.mapreducedim!(identity, *, R, A) === R
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
            # mapreduce
            @test compare(A->mapreduce(identity, +, A; dims=dims, init=zero(ET)), AT, rand(range, sz))
            @test compare(A->mapreduce(identity, *, A; dims=dims, init=one(ET)), AT, rand(range, sz))
            @test compare(A->mapreduce(x->x+x, +, A; dims=dims, init=zero(ET)), AT, rand(range, sz))

            # reduce
            @test compare(A->reduce(+, A; dims=dims, init=zero(ET)), AT, rand(range, sz))
            @test compare(A->reduce(*, A; dims=dims, init=one(ET)), AT, rand(range, sz))
        end
    end
    # Test more corner cases. Tests from AcceleraterKernels.jl
    # Cover empty (size 0) and non-singleton (size 3) axes; the size-10 loop above
    # already covers the common non-edge shape.
    for dims in [1,2,3,4,[1,2],[1,3],[1,4],[2,3],[2,4],[3,4],[1,2,3],[1,2,4],[1,3,4],[2,3,4],[1,2,3,4]],
        isize in (0, 3), jsize in (0, 3), ksize in (0, 3)
        @test compare(A->mapreduce(x->x+x, +, A; init=zero(Int32), dims), AT, rand(Int32(1):Int32(10), isize, jsize, ksize))
        @test compare(A->reduce(+, A; init=zero(Int32), dims), AT, rand(Int32(1):Int32(10), isize, jsize, ksize))
    end
end

@testsuite "reductions/sum prod" (AT, eltypes)->begin
    @testset "$ET" for ET in eltypes
        range = ET <: Real ? (ET(1):ET(10)) : ET

        # whole-array reductions: exercise each unique shape only once
        for sz in ((10,), (10,10), (10,10,10), (0,))
            @test compare(A->sum(A), AT, rand(range, sz))
            @test compare(A->prod(A), AT, rand(range, sz))
            if typeof(abs(rand(range))) in eltypes
                # abs(::Complex{Int}) promotes to Float64
                @test compare(A->sum(abs, A), AT, rand(range, sz))
                @test compare(A->prod(abs, A), AT, rand(range, sz))
            end
        end

        # reductions along specific dims
        for (sz,dims) in [(10,)=>[1], (10,10)=>[1,2], (10,10,10)=>[1,2,3], (10,10,10)=>[],
                            (10,)=>:, (10,10)=>:, (10,10,10)=>:,
                            (10,10,10)=>[1], (10,10,10)=>[2], (10,10,10)=>[3],
                            (0,)=>[1]]
            @test compare(A->sum(A; dims=dims), AT, rand(range, sz))
            @test compare(A->prod(A; dims=dims), AT, rand(range, sz))
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

@testsuite "reductions/and or xor" (AT, eltypes)->begin
    @testset "$ET" for ET in filter(x -> x <: Integer, eltypes)
        range = ET <: Real ? (ET(1):ET(10)) : ET
        for (sz,dims) in [(10,)=>[1], (10,10)=>[1,2], (10,10,10)=>[1,2,3], (10,10,10)=>[],
                          (10,)=>:, (10,10)=>:, (10,10,10)=>:,
                          (10,10,10)=>[1], (10,10,10)=>[2], (10,10,10)=>[3],
                          (0,)=>[1]]
            @test compare(A->reduce(&, A; dims=dims, init=~zero(ET)), AT, rand(range, sz))
            @test compare(A->reduce(|, A; dims=dims, init=zero(ET)), AT, rand(range, sz))
            @test compare(A->reduce(⊻, A; dims=dims, init=zero(ET)), AT, rand(range, sz))
        end
    end
end

@testsuite "reductions/minimum maximum extrema" (AT, eltypes)->begin
    @testset "$ET" for ET in eltypes
        ET <: Complex && continue
        range = ET <: Real ? (ET(1):ET(10)) : ET

        # whole-array reductions: exercise each unique shape only once
        for sz in ((10,), (10,10), (10,10,10))
            @test compare(A->minimum(A), AT, rand(range, sz))
            @test compare(A->minimum(x->x*x, A), AT, rand(range, sz))
            @test compare(A->maximum(A), AT, rand(range, sz))
            @test compare(A->maximum(x->x*x, A), AT, rand(range, sz))
            @test compare(A->extrema(A), AT, rand(range, sz))
            @test compare(A->extrema(x->x*x, A), AT, rand(range, sz))
        end

        # reductions along specific dims
        for (sz,dims) in [(10,)=>[1], (10,10)=>[1,2], (10,10,10)=>[1,2,3], (10,10,10)=>[],
                          (10,)=>:, (10,10)=>:, (10,10,10)=>:,
                          (10,10,10)=>[1], (10,10,10)=>[2], (10,10,10)=>[3]]
            @test compare(A->minimum(A; dims=dims), AT, rand(range, sz))
            @test compare(A->maximum(A; dims=dims), AT, rand(range, sz))
            @test compare(A->extrema(A; dims=dims), AT, rand(range, sz))
        end

        for (sz,red) in [(10,)=>(1,), (10,10)=>(1,1), (10,10,10)=>(1,1,1), (10,10,10)=>(10,10,10),
                         (10,10,10)=>(1,10,10), (10,10,10)=>(10,1,10), (10,10,10)=>(10,10,1)]
            @test compare((A,R)->minimum!(R, A), AT, rand(range, sz), fill(typemax(ET), red))
            @test compare((A,R)->maximum!(R, A), AT, rand(range, sz), fill(typemin(ET), red))
            @test compare((A,R)->extrema!(R, A), AT, rand(range, sz), fill((typemax(ET),typemin(ET)), red))
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

@testsuite "reductions/contract" (AT, eltypes)->begin
    M = rand(1:10, 10, 20)

    # Without `init`, `mapreducedim!` folds into the destination's values
    @test compare((R, A) -> Base.mapreducedim!(identity, +, R, A), AT, rand(1:3, 1, 20), M)
    @test compare((R, A) -> Base.mapreducedim!(abs2, +, R, A), AT, rand(1:3, 10), M)

    # A user `init` that is not neutral is applied once
    @test compare(A -> sum(A; init=10), AT, rand(1:10, 100_000))
    @test compare(A -> sum(A; dims=1, init=10), AT, rand(1:10, 1000, 3))

    # Operators without a known neutral element
    @test compare(A -> reduce((a, b) -> a + b, A), AT, rand(1:10, 100_000))
    @test compare(A -> mapreduce(abs2, (a, b) -> max(a, b), A), AT, rand(-10:10, 1000))
    if AT <: AbstractGPUArray   # (Base cannot, along `dims`)
        @test Array(mapreduce(abs2, (a, b) -> a + b, AT(M); dims=2)) == sum(abs2, M; dims=2)
    end

    # Tuple and named-tuple accumulators
    @test compare(A -> findmin(A), AT, rand(Float32, 1000))
    @test compare(A -> findmax(A), AT, rand(Float32, 100, 10))
    @test compare((A, B) -> A == B, AT, [1, 2, 3], [1, 2, 3])
    @test compare((A, B) -> A == B, AT, [1, 2, 3], [1, 5, 3])

    # Result types follow Base (compared with Base itself, whose rules differ between Julia
    # versions): small integers widen, a scalar result has the type the fold settles on, and a
    # reduction along `dims` with `init` has `init`'s type
    I = Int32[1 2; 3 5]
    for (red, args) in ((A -> sum(A), (Int8[100, 100],)), (A -> sum(A; dims=2), (Int8[100 100],)),
                        (A -> sum(A; init=Int8(0)), (Int32[1, 2],)),
                        (A -> sum(A; init=1.5f0), (I,)),
                        (A -> sum(A; dims=1, init=Int8(0)), (I,)),
                        (A -> count(isodd, A; init=Int8(0)), (I,)),
                        (A -> reduce((a, b) -> floor(Int32, a) + floor(Int32, b), A; init=0.5f0),
                         (Int16[1, 2],)))
        cpu, gpu = red(args...), red(AT(args...))
        @test gpu isa AbstractArray ? eltype(gpu) === eltype(cpu) && Array(gpu) == cpu :
                                      gpu === cpu
    end
    # An explicit `init=nothing` is an initial value
    @test_throws Exception sum(AT(Int32[1, 2]); init=nothing)
    something_add(a, b) = something(a, Int32(0)) + something(b, Int32(0))
    @test reduce(something_add, AT(Int32[1, 2]); init=nothing) === Int32(3)
    # Base's checks of `dims`
    @test_throws ArgumentError sum(AT(Int32[1, 2]); dims=0)
    @test_throws ArgumentError sum(AT(Int32[1 2; 3 4]); dims=1.5)

    # Empty reductions follow Base
    @test sum(AT(Int[])) === 0
    @test prod(AT(Float32[])) === 1f0
    @test sum(AT(Int[]); init=Int8(0)) === Int8(0)
    @test reduce((a, b) -> a + b, AT(Int[]); init=0) === 0
    # (the same error as Base's, whose type differs between Julia versions)
    errtype(f) = try f(); nothing catch err; typeof(err) end
    @test errtype(() -> maximum(AT(Int[]))) === errtype(() -> maximum(Int[])) !== nothing
    @test errtype(() -> reduce((a, b) -> a + b, AT(Int[]))) ===
          errtype(() -> reduce((a, b) -> a + b, Int[])) !== nothing
    @test compare(A -> sum(A; dims=1), AT, zeros(Int, 0, 3))
    @test compare(A -> sum(A; dims=2), AT, zeros(Int, 0, 3))
    @test_throws ArgumentError maximum(AT(zeros(Int, 0, 3)); dims=1)

    # `Broadcasted` sources, and several arrays
    @test compare((A, B) -> sum(Broadcast.instantiate(Broadcast.broadcasted(*, A, B))), AT,
                  rand(Float32, 100), rand(Float32, 100))
    @test compare((A, B) -> mapreduce(*, +, A, B), AT, rand(Float32, 100), rand(Float32, 100))
    @test compare((A, B) -> mapreduce(*, +, A, B), AT, rand(Float32, 100), rand(Float32, 50))
    @test compare((A, B) -> mapreduce(*, +, A, B; dims=1), AT, rand(Float32, 10, 10), rand(Float32, 10, 10))
    # ... including arrays without a backend, such as the indices `findfirst` reduces with
    @test compare(A -> mapreduce(+, (a, b) -> a + b, A, GPUArrays.EachIndex(A)), AT, Int32[1, 2])
    @test compare(A -> mapreduce(+, +, A, GPUArrays.EachIndex(A); init=10), AT, Int32[1, 2])
end

@testsuite "reductions/base" (AT, eltypes)->begin
    # Base's exact results (see `compare_exact`), which AcceleratedKernels leaves to GPUArrays
    same(f, xs...) = compare_exact(f, AT, xs...)

    # empty inputs: Base's value, else its error
    for (f, x) in ((sum, Int32[]), (prod, Int32[]), (count, Bool[]), (minimum, Int32[]),
                   (A -> reduce(max, A), Int32[]), (A -> reduce((a, b) -> a + b, A), Int32[]),
                   (A -> sum(A; init=1), Float32[]), (A -> maximum(A; init=Int32(-1)), Int32[]),
                   (A -> reduce(+, A; init=nothing), Int32[]),
                   (A -> mapreduce(x -> error("never called"), +, A; init=7), Int32[]))
        @test same(f, x)
    end
    # one element: Base's `mapreduce_first`, or `op(init, x)`
    for (f, x) in ((A -> reduce((a, b) -> a + b, A), [true]), (sum, [true]), (sum, Int8[3]),
                   (A -> reduce(+, A; init=Int8(0)), [true]), (A -> mapreduce(x -> x + 1, +, A), Int8[1]),
                   (maximum, Int8[3]), (A -> reduce(*, A; init=0.5f0), Int32[3]))
        @test same(f, x)
    end
    # ... whose map may index device arrays
    w = AT(Int32[11, 22])
    @test mapreduce(x -> w[x], +, AT(Int32[2])) === Int32(22)
    # scalar result types
    h8 = Int8[100, 100, 27]
    for f in (sum, A -> reduce(+, A), A -> reduce(+, A; init=0), A -> sum(A; init=Int16(0)),
              A -> reduce((a, b) -> Base.add_sum(a, b), A), maximum, A -> count(>(50), A))
        @test same(f, h8)
    end
    @test same(A -> sum(A; init=Int8(0)), [1, 2])
    if VERSION >= v"1.13-"   # (Julia 1.10's Base wraps these in the small type)
        @test same(A -> sum(A; init=UInt8(0)), Int8[-1, -2])
        @test same(A -> count(A; init=UInt8(0)), [true, true])
    end

    # along `dims`: `typeof(init)`, else the fold type
    m8 = rand(Int8(-9):Int8(9), 40, 30)
    for dims in (1, 2, (1, 2), 3)
        for f in (A -> sum(A; dims), A -> sum(A; dims, init=Int16(1)), A -> maximum(A; dims),
                  A -> count(x -> x > 0, A; dims))
            @test same(f, m8)
        end
        # (Base's pairwise path reduces `Int8`s in `Int8`, which the values here do not overflow)
        @test same(A -> reduce(+, A; dims, init=0.5), Int16.(m8))
    end
    # ... and empty reduced dimensions: `init`, else Base's initial value or error
    e8 = zeros(Int8, 0, 3)
    for (f, x) in ((A -> sum(A; dims=1), e8), (A -> prod(A; dims=1), e8),
                   (A -> minimum(A; dims=1, init=Int8(7)), e8), (A -> minimum(A; dims=1), e8),
                   (A -> maximum(A; dims=1), zeros(Int8, 3, 0)),
                   (A -> minimum(A; dims=1), zeros(Int32, 0, 0)),
                   (A -> mapreduce(x -> x + 1, +, A; dims=1), zeros(Int32, 0, 2)),
                   (A -> mapreduce(x -> x + 1, *, A; dims=1), zeros(Int32, 0, 2)),
                   (A -> count(A; dims=2), zeros(Bool, 3, 0)))
        @test same(f, x)
    end
    # ... whose map may index device arrays
    wh = Int32[11, 22]
    @test Array(mapreduce(x -> w[x + 1], +, AT(zeros(Int32, 0, 2)); dims=1)) ==
          mapreduce(x -> wh[x + 1], +, zeros(Int32, 0, 2); dims=1)

    # signed zeros: Base's sums along `dims` start from zero, whole-array sums do not
    z = fill(-0.0f0, 40, 3)
    for f in (sum, A -> sum(A; dims=1), A -> sum(A; dims=2), A -> sum(A; init=0.0f0),
              A -> prod(A; dims=1), A -> maximum(A; dims=1), A -> reduce(+, A; dims=(1, 2)))
        @test same(f, z)
    end

    # the in-place reductions, with `init=true` and `false`, into destinations of another type
    A = rand(1:9, 20, 30)
    B = rand(Bool, 20, 30)
    for init in (true, false), dims in (1, 2)
        sz = dims == 1 ? (1, 30) : (20, 1)
        for (f!, r, x) in ((sum!, rand(1:3, sz), A), (sum!, Float32.(rand(1:3, sz)), A),
                           (prod!, rand(1.0f0:2.0f0, sz), A .% 2 .+ 1),
                           (maximum!, rand(1:3, sz), A), (minimum!, rand(Int16(1):Int16(3), sz), A),
                           (any!, rand(Bool, sz), B), (all!, rand(Bool, sz), B),
                           (count!, rand(1:3, sz), B),
                           (extrema!, fill((5, 5), sz), A))
            @test same((r, x) -> f!(r, x; init), r, x)
        end
        @test same((r, x) -> sum!(abs2, r, x; init), rand(1:3, sz), A)
    end
    # ... of empty inputs
    for (f!, r) in ((sum!, ones(Float32, 1, 3)), (prod!, zeros(Float32, 1, 3)),
                    (maximum!, zeros(Float32, 1, 3)), (count!, ones(Int, 1, 3))), init in (true, false)
        @test same((r, x) -> f!(r, x; init), r, zeros(f! === count! ? Bool : Float32, 0, 3))
    end
    # ... and `Base.mapreducedim!`, which folds into the destination
    @test same((r, x) -> Base.mapreducedim!(abs2, +, r, x), rand(1:3, 1, 30), A)

    # findmin and findmax: Base's indices, without an `init`
    for (f, x) in ((findmin, Float32[3, 1, 2]), (findmax, Float32[3, 1, 2]),
                   (A -> findmax(A; dims=1), Float32[3, 1, 2]),
                   (A -> findmin(A; dims=2), rand(Float32, 20, 30)),
                   (A -> findmax(A; dims=(1, 2)), rand(Float32, 20, 30)),
                   (A -> findmax(x -> (x > 0.5f0, -x), A), rand(Float32, 100)),
                   (argmin, rand(Float32, 100)), (A -> argmax(A; dims=1), rand(Float32, 20, 30)),
                   (findmin, Int[]), (A -> findmax(A; dims=1), zeros(Float32, 0, 3)),
                   (A -> findmax(A; dims=1), zeros(Float32, 3, 0)),
                   (A -> findmin(A; dims=1), fill(3.0f0)), (A -> findmax(A; dims=2), fill(3.0f0)),
                   (A -> argmin(abs, A), Int32[-4, 2, 1, -1]), (A -> argmax(abs, A), Int32[-4, 2, 1, -1]),
                   (A -> argmax(abs, A), Int32[]))
        @test same(f, x)
    end

    # 0-dimensional arrays, views and reshapes
    for (f, x) in ((sum, fill(3)), (A -> sum(A; dims=1), fill(3)), (A -> mapreduce(abs2, +, A; init=1), fill(3)),
                   (findmax, fill(3.0f0)),
                   (A -> sum(view(A, 2:9, :); dims=1), rand(1:9, 10, 4)),
                   (A -> maximum(view(A, 1:2:9)), rand(1:9, 10)),
                   (A -> sum(reshape(A, 4, 5); dims=2), rand(1:9, 20)))
        @test same(f, x)
    end
    @test same((r, x) -> sum!(view(r, 1:1, :), x), zeros(Int, 2, 4), rand(1:9, 3, 4))
end

@testsuite "reductions/neutral_element" (AT, eltypes)->begin
    # GPUArrays extends GPUArraysCore's function, so every package shares one set of methods
    @test GPUArrays.neutral_element === GPUArrays.GPUArraysCore.neutral_element
    @test GPUArrays.neutral_element(+, Float32) === 0.0f0
    @test GPUArrays.neutral_element(*, Int32) === Int32(1)
    @test GPUArrays.neutral_element(min, Int16) === typemax(Int16)
    @test GPUArrays.neutral_element(max, Float64) === -Inf
    @test GPUArrays.neutral_element(&, UInt8) === 0xff
    @test GPUArrays.neutral_element(Base._extrema_rf, NTuple{2,Int8}) === (typemax(Int8), typemin(Int8))
    @test_throws ErrorException GPUArrays.neutral_element((x, y) -> x, Int)
    if isdefined(Base, :and_all)    # Julia 1.13
        @test GPUArrays.neutral_element(Base.and_all, Bool) === true
        @test GPUArrays.neutral_element(Base.or_any, Bool) === false
    end
end
