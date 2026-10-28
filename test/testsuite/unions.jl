# Arrays whose element type is an isbits union (`Union{Int32, Nothing}`, ...). Like `Array`,
# they store each element inline as one of the union's members, with a selector byte per
# element that says which one.

# a struct with a union field is stored inline too, but with the selector inside the element
struct UnionField
    x::Union{Int32, Float32}
end

# compare elements by identity, as `==` propagates `missing` and `isequal` ignores types
# (`isequal(Int8(1), 1f0)`)
identical(a, b) = size(a) == size(b) && all(map(===, a, b))

@kernel function union_increment!(x)
    i = @index(Global, Linear)
    val = x[i]
    if val isa Integer
        x[i] = val + one(val)
    end
end
increment_integers!(x) = union_increment!(get_backend(x))(x; ndrange=length(x))

@kernel function union_isequal!(ok, x, y)
    i = @index(Global, Linear)
    ok[i] = x[i] === y[i]
end

# swap the members of `Union{Int64, NTuple{3, Int32}}`, whose elements are padded
@kernel function union_swap!(x)
    i = @index(Global, Linear)
    val = x[i]
    x[i] = val isa Int64 ? (Int32(val), Int32(0), Int32(0)) : Int64(sum(val))
end

@kernel function union_field!(x)
    i = @index(Global, Linear)
    x[i] = UnionField(isodd(i) ? Int32(i) : Float32(i))
end

@testsuite "unions" (AT, eltypes)->begin
    @testset "copies" begin
        for a in (Union{Int32, Nothing}[1, nothing, 3, nothing, 5],
                  # members of different sizes and alignments
                  Union{Int8, Float32}[Int8(1), 2f0, Int8(3), 4f0, Int8(5)],
                  Union{NTuple{3, Int8}, Int16}[Int8.((1, 2, 3)), Int16(4), Int8.((5, 6, 7)),
                                                Int16(8), Int8.((9, 10, 11))],
                  # elements of 12 bytes, padded to 16
                  Union{Int64, NTuple{3, Int32}}[1, Int32.((2, 3, 4)), 5, Int32.((6, 7, 8)), 9],
                  # only singletons, so selector bytes but no data
                  Union{Nothing, Missing}[nothing, missing, missing, nothing, missing],
                  # nested unions are flattened
                  Union{Union{Missing, Nothing}, Int32}[0, missing, nothing, 3, missing])
            T = eltype(a)
            b = AT(a)
            @test b isa AT{T, 1}
            @test identical(Array(b), a)
            @test identical(Array(copy(b)), a)
            @test identical(Array(reshape(b, 1, 5)), reshape(a, 1, 5))
            if VERSION >= v"1.11"   # (`sizeof` of such an `Array` used to ignore padding)
                @test sizeof(b) == sizeof(a)
            end

            c = similar(b)
            copyto!(c, b)
            @test identical(Array(c), a)
            copyto!(c, 2, b, 1, 3)
            @test identical(Array(c), a[[1, 1, 2, 3, 5]])

            # partial copies to and from the host
            copyto!(c, 3, a, 4, 2)
            @test identical(Array(c), a[[1, 1, 4, 5, 5]])
            d = similar(a)
            copyto!(d, 2, b, 3, 2)
            @test identical(d[2:3], a[3:4])

            # broadcasting reads and writes the elements too
            @test identical(Array(similar(b) .= b), a)
            @test identical(Array(fill!(similar(b), a[2])), fill(a[2], 5))
        end
    end

    @testset "views" begin
        # the parent's first selector bytes differ from the view's
        a = Union{Int32, Nothing}[nothing, nothing, 1, nothing, 3, 5]
        b = AT(a)
        v = view(b, 3:5)
        @test identical(Array(v), a[3:5])
        @test identical(Array(copy(v)), a[3:5])
        @test identical(Array(something.(v, Int32(0))), Int32[1, 0, 3])
        @test @allowscalar v[1] === Int32(1)

        # writes through a view don't touch the parent outside of it
        increment_integers!(v)
        @test identical(Array(b), Union{Int32, Nothing}[nothing, nothing, 2, nothing, 4, 5])
        v .= something.(v, Int32(0))
        @test identical(Array(b), Union{Int32, Nothing}[nothing, nothing, 2, 0, 4, 5])
        fill!(view(b, 4:5), nothing)
        @test identical(Array(b), Union{Int32, Nothing}[nothing, nothing, 2, nothing, nothing, 5])
        copyto!(view(b, 1:2), Union{Int32, Nothing}[7, nothing])
        @test identical(Array(b), Union{Int32, Nothing}[7, nothing, 2, nothing, nothing, 5])
        copyto!(view(b, 4:5), view(b, 1:2))
        @test identical(Array(b), Union{Int32, Nothing}[7, nothing, 2, 7, nothing, 5])
        view(b, 2:6) .= view(b, 1:5)   # overlapping, so the source is copied first
        @test identical(Array(b), Union{Int32, Nothing}[7, 7, nothing, 2, 7, nothing])
        copyto!(b, Union{Int32, Nothing}[7, nothing, 2, 7, nothing, 5])

        # non-contiguous views, views of views, and reshaped views
        w = view(b, 1:2:5)
        w .= ifelse.(w .=== nothing, Int32(0), nothing)
        @test identical(Array(b), Union{Int32, Nothing}[nothing, nothing, nothing, 7, 0, 5])
        @test identical(Array(view(view(b, 2:6), 3:5)), Union{Int32, Nothing}[7, 0, 5])
        increment_integers!(view(view(b, 2:6), 3:5))
        @test identical(Array(b), Union{Int32, Nothing}[nothing, nothing, nothing, 8, 1, 6])
        @test identical(Array(reshape(view(b, 3:6), 2, 2)),
                   Union{Int32, Nothing}[nothing 1; 8 6])

        # views of unions of singletons have no byte offset, only one in the selector bytes
        c = AT([nothing, nothing, missing, missing])
        @test identical(Array(view(c, 3:4)), [missing, missing])
        @test identical(Array(copy(view(c, 2:3))), [nothing, missing])
        ok = AT(fill(false, 2))
        union_isequal!(get_backend(ok))(ok, view(c, 3:4), AT([missing, missing]); ndrange=2)
        @test all(Array(ok))
        fill!(view(c, 2:3), missing)
        @test identical(Array(c), [nothing, missing, missing, missing])
        view(c, 2:4) .= view(c, 1:3)
        @test identical(Array(c), [nothing, nothing, missing, missing])
    end

    @testset "kernels" begin
        a = AT(Union{Int32, Nothing}[1, nothing, 3])
        increment_integers!(a)
        @test identical(Array(a), Union{Int32, Nothing}[2, nothing, 4])

        b = AT(Union{Union{Missing, Nothing}, Int32}[0, missing, nothing])
        increment_integers!(b)
        @test identical(Array(b), Union{Union{Missing, Nothing}, Int32}[1, missing, nothing])

        c = AT(Union{Int64, NTuple{3, Int32}}[1, Int32.((2, 3, 4)), 5])
        union_swap!(get_backend(c))(c; ndrange=length(c))
        @test identical(Array(c), Union{Int64, NTuple{3, Int32}}[Int32.((1, 0, 0)), 9, Int32.((5, 0, 0))])
        union_swap!(get_backend(c))(view(c, 2:3); ndrange=2)
        @test identical(Array(c), Union{Int64, NTuple{3, Int32}}[Int32.((1, 0, 0)), Int32.((9, 0, 0)), 5])

        d = AT{UnionField}(undef, 4)
        union_field!(get_backend(d))(d; ndrange=length(d))
        @test Array(d) == [UnionField(Int32(1)), UnionField(2f0),
                            UnionField(Int32(3)), UnionField(4f0)]
        @test Array(map(u -> u.x isa Int32, d)) == [true, false, true, false]
    end

    @testset "fill!" begin
        # also for members that could be filled with a memset, which would skip the selectors
        for (T, xs) in ((Union{Int8, UInt8}, (Int8(-1), 0xff)),
                        (Union{Int16, UInt16}, (Int16(-1), 0xffff)),
                        (Union{Int32, Float32}, (Int32(1), 2f0)),
                        (Union{Int8, Int32}, (Int8(1), Int32(2))),
                        (Union{Nothing, Missing}, (nothing, missing)))
            b = AT{T}(undef, 3)
            for x in xs
                fill!(b, x)
                @test identical(Array(b), T[x, x, x])
            end
        end
    end

    @testset "array operations" begin
        x = AT(Union{Missing, Bool}[missing, true, false])
        @test identical(Array(x .| false), Union{Missing, Bool}[missing, true, false])
        @test identical(Array(map(!, x)), Union{Missing, Bool}[missing, false, true])
        @test any(x) === true
        @test all(x) === false
        @test count(ismissing, x) == 1
        @test identical(Array(reverse(x)), Union{Missing, Bool}[false, true, missing])
        @test identical(Array(vcat(x, x)), Union{Missing, Bool}[missing, true, false, missing, true, false])
        @test identical(Array(x[2:3]), Union{Missing, Bool}[true, false])

        y = AT(Union{Int32, Nothing}[1, nothing, 3, nothing])
        @test sum(v -> something(v, Int32(0)), y) == 4
        @test mapreduce(v -> something(v, Int32(10)), max, y) === Int32(10)
        @test Array(sum(something.(reshape(y, 2, 2), Int32(0)); dims=1)) == Int32[1 3]
        @test identical(Array(y[AT([4, 1])]), Union{Int32, Nothing}[nothing, 1])
        @test Array(findall(isnothing, y)) == [2, 4]
    end

    @testset "scalar indexing" begin
        a = AT(Union{Int32, Nothing}[1, nothing, 3])
        @allowscalar begin
            @test a[1] === Int32(1)
            @test a[2] === nothing
            a[1] = nothing
            a[2] = 2
            @test view(a, 2:3)[2] === Int32(3)
        end
        @test identical(Array(a), Union{Int32, Nothing}[nothing, 2, 3])

        b = AT(Union{Nothing, Missing}[nothing, nothing])
        @allowscalar b[2] = missing
        @test identical(Array(b), [nothing, missing])
    end

    @testset "resize!" begin
        a = AT(Union{Int32, Nothing}[1, nothing, 3])
        resize!(a, 5)
        @test length(a) == 5
        @test identical(Array(view(a, 1:3)), Union{Int32, Nothing}[1, nothing, 3])
        resize!(a, 2)
        @test identical(Array(a), Union{Int32, Nothing}[1, nothing])

        c = AT(Union{Nothing, Missing}[nothing, nothing, missing, missing])
        if AT <: AbstractGPUArray   # (contiguous views are arrays of their own)
            # a view gets its own memory, leaving the parent alone
            b = AT(Union{Int32, Nothing}[nothing, nothing, 1, nothing, 3])
            v = view(b, 3:5)
            resize!(v, 4)
            @test identical(Array(view(v, 1:3)), Union{Int32, Nothing}[1, nothing, 3])
            resize!(v, 2)
            @test identical(Array(v), Union{Int32, Nothing}[1, nothing])
            @test identical(Array(b), Union{Int32, Nothing}[nothing, nothing, 1, nothing, 3])

            w = view(c, 3:4)
            resize!(w, 3)
            @test identical(Array(view(w, 1:2)), [missing, missing])
            resize!(w, 1)
            @test identical(Array(w), [missing])
        end
        resize!(c, 6)
        @test identical(Array(view(c, 1:4)), [nothing, nothing, missing, missing])
        resize!(c, 1)
        @test identical(Array(c), [nothing])
    end
end
