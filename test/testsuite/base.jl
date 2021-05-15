function cartesian_iter(state, res, A, Asize)
    for i in CartesianIndices(Asize)
        res[i] = A[i]
    end
    return
end

function clmap!(ctx, f, out, b)
    i = linear_index(ctx) # get the kernel index it gets scheduled on
    out[i] = f(b[i])
    return
end

function ntuple_test(ctx, result, ::Val{N}) where N
    result[1] = ntuple(Val(N)) do i
        Float32(i) * 77f0
    end
    return
end

function ntuple_closure(ctx, result, ::Val{N}, testval) where N
    result[1] = ntuple(Val(N)) do i
        Float32(i) * testval
    end
    return
end

@testsuite "base" (AT, eltypes)->begin
    @testset "copy!" begin
        for (dst, src,) in (
                            (rand(Float32, (10,)),   rand(Float32, (10,))),   # vectors
                            (rand(Float32, (10,20)), rand(Float32, (10,20))), # multidim
                            )
            dst = AT(dst)
            src = AT(src)
            
            copy!(dst, src)
            @test dst == src
        end
    end

    @testset "copyto!" begin
        x = fill(0f0, (10, 10))
        y = rand(Float32, (20, 10))
        a = AT(x)
        b = AT(y)
        r1 = CartesianIndices((1:7, 3:8))
        r2 = CartesianIndices((4:10, 3:8))
        copyto!(x, r1, y, r2)
        copyto!(a, r1, b, r2)
        @test x == Array(a)
        r2 = CartesianIndices((4:11, 3:8))
        @test_throws ArgumentError copyto!(a, r1, b, r2)

        r2 = CartesianIndices((4:10, 3:8))
        x2 = fill(0f0, (10, 10))
        copyto!(x2, r1, b, r2)
        @test x2 == x

        fill!(a, 0f0)
        copyto!(a, r1, y, r2)
        @test Array(a) == x

        x = fill(0f0, (10,))
        y = rand(Float32, (20,))
        a = AT(x)
        b = AT(y)
        r1 = CartesianIndices((1:7,))
        r2 = CartesianIndices((4:10,))
        copyto!(x, r1, y, r2)
        copyto!(a, r1, b, r2)
        @test x == Array(a)
        r2 = CartesianIndices((4:11,))
        @test_throws ArgumentError copyto!(a, r1, b, r2)

        x = fill(0f0, (10,))
        y = rand(Float32, (20,))
        a = AT(x)
        b = AT(y)
        r1 = 1:7
        r2 = 4:10
        copyto!(x, r1, y, r2)
        copyto!(a, r1, b, r2)
        @test x == Array(a)

        x = fill(0f0, (10,))
        y = fill(1f0, (10,))
        a = AT(x)
        b = AT(y)
        copyto!(a, b)
        @test Float32.(y) == Array(a)

        # wrapped gpu array to wrapped gpu array
        x = rand(Float32, 4, 4)
        a = AT(x)
        b = view(a, 2:3, 2:3)
        c = AT{eltype(b)}(undef, size(b))
        copyto!(c, b)
        Array(c) == Array(b)

        # wrapped gpu array to cpu array
        z = Array{eltype(b)}(undef, size(b))
        copyto!(z, b)
        @test z == Array(b)

        # cpu array to wrapped gpu array
        copyto!(b, z)

        # bug in copyto!
        ## needless N type parameter
        @test compare((x,y)->copyto!(y, selectdim(x, 2, 1)), AT, ones(Float32, 2, 2, 2), zeros(Float32, 2, 2))
        ## inability to copyto! smaller destination
        ## (this was broken on Julia <1.5)
        @test compare((x,y)->copyto!(y, selectdim(x, 2, 1)), AT, ones(Float32, 2, 2, 2), zeros(Float32, 3, 3))

        if (Float32 in eltypes && Float64 in eltypes)
            # mismatched types
            let src = rand(Float32, 4)
                dst = AT{Float64}(undef, size(src))
                copyto!(dst, src)
                @test Array(dst) == src
            end
            let dst = Array{Float64}(undef, 4)
                src = AT(rand(Float32, size(dst)))
                copyto!(dst, src)
                @test Array(src) == dst
            end
        end
    end

    @testset "cat" begin
        @test compare(hcat, AT, rand(Float32, 3), rand(Float32, 3))
        @test compare(hcat, AT, rand(Float32, ), rand(Float32, 1, 3))
        @test compare(hcat, AT, rand(Float32, 1, 3), rand(Float32))
        @test compare(hcat, AT, rand(Float32, 3), rand(Float32, 3, 3))
        @test compare(hcat, AT, rand(Float32, 3, 3), rand(Float32, 3))
        @test compare(hcat, AT, rand(Float32, 3, 3), rand(Float32, 3, 3))
        #@test compare(hcat, AT, rand(Float32, ), rand(Float32, 3, 3))
        #@test compare(hcat, AT, rand(Float32, 3, 3), rand(Float32))

        @test compare(vcat, AT, rand(Float32, 3), rand(Float32, 3))
        @test compare(vcat, AT, rand(Float32, 3, 3), rand(Float32, 3, 3))
        @test compare(vcat, AT, rand(Float32, ), rand(Float32, 3))
        @test compare(vcat, AT, rand(Float32, 3), rand(Float32))
        @test compare(vcat, AT, rand(Float32, ), rand(Float32, 3, 3))
        #@test compare(vcat, AT, rand(Float32, 3, 3), rand(Float32))

        @test compare((a,b) -> cat(a, b; dims=4), AT, rand(Float32, 3, 4), rand(Float32, 3, 4))
    end

    @testset "reshape" begin
        @test compare(reshape, AT, rand(Float32, 10), Ref((10,)))
        @test compare(reshape, AT, rand(Float32, 10), Ref((10,1)))
        @test compare(reshape, AT, rand(Float32, 10), Ref((1,10)))

        @test_throws Exception reshape(AT(rand(Float32, 10)), (10,2))
    end

    @testset "reinterpret" begin
        a = rand(ComplexF32, 22)
        A = AT(a)
        af0 = reinterpret(Float32, a)
        Af0 = reinterpret(Float32, A)
        @test Array(Af0) == af0
        a = rand(ComplexF32, 4, 4)
        A = AT(a)

        a = rand(ComplexF32, 10 * 10)
        A = AT(a)
        af0 = reshape(reinterpret(Float32, vec(a)), (20, 10))
        Af0 = reshape(reinterpret(Float32, vec(A)), (20, 10))
        @test Array(Af0) == af0
    end

    AT <: AbstractGPUArray && @testset "ntuple test" begin
        result = AT(Vector{NTuple{3, Float32}}(undef, 1))
        gpu_call(ntuple_test, result, Val(3))
        @test Array(result)[1] == (77, 2*77, 3*77)
        x = 88f0
        gpu_call(ntuple_closure, result, Val(3), x)
        @test Array(result)[1] == (x, 2*x, 3*x)
    end

    AT <: AbstractGPUArray && @testset "cartesian iteration" begin
        Ac = rand(Float32, 32, 32)
        A = AT(Ac)
        result = fill!(copy(A), 0.0f0)
        gpu_call(cartesian_iter, result, A, size(A))
        Array(result) == Ac
    end

    AT <: AbstractGPUArray && @testset "Custom kernel from Julia function" begin
        x = AT(rand(Float32, 100))
        y = AT(rand(Float32, 100))
        gpu_call(clmap!, -, x, y; target=x)
        jy = Array(y)
        @test map!(-, jy, jy) ≈ Array(x)
    end

    @testset "map" begin
        @test compare((a, b)-> map(+, a, b),    AT, rand(Float32, 10), rand(Float32, 10))
        @test compare((a, b)-> map!(-, a, b),   AT, rand(Float32, 10), rand(Float32, 10))
        @test compare((a, b, c, d)-> map!(*, a, b, c, d), AT, rand(Float32, 10), rand(Float32, 10), rand(Float32, 10), rand(Float32, 10))
    end

    @testset "repeat" begin
        @test compare(a-> repeat(a, 5, 6),  AT, rand(Float32, 10))
        @test compare(a-> repeat(a, 5),     AT, rand(Float32, 10))
        @test compare(a-> repeat(a, 5),     AT, rand(Float32, 5, 4))
        @test compare(a-> repeat(a, 4, 3),  AT, rand(Float32, 10, 15))
        @test compare(a-> repeat(a, 0),     AT, rand(Float32, 10))
        @test compare(a-> repeat(a, 0),     AT, rand(Float32, 5, 4))
        @test compare(a-> repeat(a, 4, 0),  AT, rand(Float32, 10, 15))

        if VERSION ≥ v"1.6"
            # Test inputs.
            x = rand(Float32, 10)
            xmat = rand(Float32, 2, 10)
            arr = rand(Float32, 3, 2, 10)

            # Inner.
            @test compare(a -> repeat(a, inner=(2, )), AT, x)
            @test compare(a -> repeat(a, inner=(2, 3)), AT, xmat)
            @test compare(a -> repeat(a, inner=(2, 3, 4)), AT, xarr)
            # Outer.
            @test compare(a -> repeat(a, outer=(2, )), AT, x)
            @test compare(a -> repeat(a, outer=(2, 3)), AT, xmat)
            @test compare(a -> repeat(a, outer=(2, 3, 4)), AT, xarr)
            # Both.
            @test compare(a -> repeat(a, inner=(2, ), outer=(2, )), AT, x)
            @test compare(a -> repeat(a, inner=(2, 3), outer=(2, 3)), AT, xmat)
            @test compare(a -> repeat(a, inner=(2, 3, 4), outer=(2, 1, 4)), AT, xarr)
            # Repeat which expands dimensionality.
            @test compare(a -> repeat(a, inner=(2, 1, 3)), AT, x)
            @test compare(a -> repeat(a, outer=(2, 1, 3)), AT, x)
            @test compare(a -> repeat(a, inner=(2, 1, 3), outer=(2, 2, 3)), AT, x)

            @test compare(a -> repeat(a, inner=(2, 1, 3)), AT, xmat)
            @test compare(a -> repeat(a, outer=(2, 1, 3)), AT, xmat)
            @test compare(a -> repeat(a, inner=(2, 1, 3), outer=(2, 2, 3)), AT, xmat)
        end
    end

    @testset "permutedims" begin
        @test compare(x->permutedims(x, [1, 2]), AT, rand(Float32, 4, 4))

        inds = rand(1:100, 150, 150)
        @test compare(x->permutedims(view(x, inds, :), (3, 2, 1)), AT, rand(Float32, 100, 100))
    end

    @testset "circshift" begin
        @test compare(x->circshift(x, (0,1)), AT, reshape(Vector(1:16), (4,4)))
    end

    @testset "copy" begin
        a = AT([1])
        b = copy(a)
        fill!(b, 0)
        @test Array(b) == [0]
        @test Array(a) == [1]
    end

    @testset "input output" begin
        # compact=false to avoid type aliases
        replstr(x, kv::Pair...) = sprint((io,x) -> show(IOContext(io, :compact => false, :limit => true, :displaysize => (24, 80), kv...), MIME("text/plain"), x), x)
        showstr(x, kv::Pair...) = sprint((io,x) -> show(IOContext(io, :limit => true, :displaysize => (24, 80), kv...), x), x)

        @testset "showing" begin
            # vectors and non-vector arrays showing
            # are handled differently in base/arrayshow.jl
            A = AT(Int64[1])
            B = AT(Int64[1 2;3 4])

            msg = replstr(A)
            @test occursin(Regex("^1-element $AT{Int64,\\s?1.*}:\n 1\$"), msg)

            # # result of e.g. `print` differs on 32bit and 64bit machines
            # due to different definition of `Int` type
            # print([1]) shows as [1] on 64bit but Int64[1] on 32bit
            msg = showstr(A)
            @test msg == "[1]" || msg == "Int64[1]"

            msg = replstr(B)
            @test occursin(Regex("^2×2 $AT{Int64,\\s?2.*}:\n 1  2\n 3  4\$"), msg)

            msg = showstr(B)
            @test msg == "[1 2; 3 4]" || msg == "Int64[1 2; 3 4]"

            # the printing of Adjoint depends on global state
            msg = replstr(A')
            @test occursin(Regex("^1×1 Adjoint{Int64,\\s?$AT{Int64,\\s?1.*}}:\n 1\$"), msg) ||
                occursin(Regex("^1×1 LinearAlgebra.Adjoint{Int64,\\s?$AT{Int64,\\s?1.*}}:\n 1\$"), msg) ||
                occursin(Regex("^1×1 adjoint\\(::$AT{Int64,\\s?1.*}\\) with eltype Int64:\n 1\$"), msg)
        end
    end
end
