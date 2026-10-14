@kernel function staticdims_index_kernel!(dest, src)
    i, j, k = @index(Global, NTuple)
    @inbounds dest[i, j, k] = src[i, j, k] + i + 10j + 100k
end

@testsuite "staticdims" (AT, eltypes)->begin
    # a `StaticDimsArray` wraps GPU memory
    AT <: AbstractGPUArray || return
    static = GPUArrays.Static.static
    StaticInt = GPUArrays.Static.StaticInt

    T = Float32 in eltypes ? Float32 : first(eltypes)
    dims = (static(4), static(3), 5)
    host = rand(T, 4, 3, 5)
    a = StaticDimsArray(AT(vec(host)), dims)

    @testset "size" begin
        @test a isa AbstractGPUArray{T, 3}
        @test size(a) == (4, 3, 5)
        @test GPUArrays.static_size(a) === dims
        @test GPUArrays.known_size(typeof(a)) === (4, 3, nothing)
        @test Array(a) == host
        @test_throws DimensionMismatch StaticDimsArray(AT{T}(undef, 10), dims)
        # a multidimensional parent is viewed as a vector
        @test StaticDimsArray(AT(host), dims) == a
    end

    @testset "kernel arguments" begin
        backend = get_backend(a)
        b = similar(a)
        @test b isa StaticDimsArray{T, 3, typeof(dims)}
        staticdims_index_kernel!(backend)(b, a; ndrange = size(a))
        expected = [host[i, j, k] + i + 10j + 100k for i in 1:4, j in 1:3, k in 1:5]
        @test Array(b) ≈ expected
    end

    @testset "broadcasting" begin
        b = similar(a)
        b .= a .* 2
        @test Array(b) ≈ host .* 2
        @test b isa StaticDimsArray
        # mixed with plain arrays of the back-end
        b .= a .+ AT(host)
        @test Array(b) ≈ host .+ host
        # out of place: an array of the back-end
        c = a .+ one(T)
        @test Array(c) ≈ host .+ one(T)
        fill!(b, zero(T))
        @test all(iszero, Array(b))
    end

    @testset "reductions" begin
        @test sum(a) ≈ sum(host)
        @test Array(sum(a; dims = 3)) ≈ sum(host; dims = 3)
        @test Array(maximum(a; dims = 1)) ≈ maximum(host; dims = 1)
    end

    @testset "indexing" begin
        @test (@allowscalar a[2, 3, 4]) == host[2, 3, 4]
        @test Array(a[:, 2, :]) == host[:, 2, :]
        @test Array(view(a, :, :, 2)) == host[:, :, 2]
        @test Array(reshape(a, 12, 5)) == reshape(host, 12, 5)
    end

    @testset "aliasing" begin
        @test Base.mightalias(a, parent(a))
        @test !Base.mightalias(a, copy(a))
    end
end
