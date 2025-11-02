@kernel function cartesian_iter(res, A)
    i = @index(Global, Cartesian)
    res[i] = A[i]
end

@kernel function clmap!(f, out, b)
    i = @index(Global, Linear) # get the kernel index it gets scheduled on
    out[i] = f(b[i])
end

@kernel function ntuple_test(result, ::Val{N}) where N
    result[1] = ntuple(Val(N)) do i
        Float32(i) * 77f0
    end
end

@kernel function ntuple_closure(result, ::Val{N}, testval) where N
    result[1] = ntuple(Val(N)) do i
        Float32(i) * testval
    end
end

@testsuite "base" (AT, eltypes)->begin
    if AT <: AbstractGPUArray
        @testset "storage" begin
          x = AT(rand(Float32, 10))
          @test GPUArrays.storage(x) isa GPUArrays.DataRef
          GPUArrays.unsafe_free!(x)
        end
    end

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
        x = fill(0f0, 1)
        y = rand(Float32, 1)
        a = AT(x)
        copyto!(x, 1:1, y, 1:1)
        copyto!(a, 1:1, y, 1:1)
        @test x == Array(a)

        x = fill(0f0, 10)
        y = rand(Float32, 20)
        a = AT(x)
        b = AT(y)
        r1 = 1:7
        r2 = 11:17
        copyto!(x, r1, y, r2)
        copyto!(a, r1, b, r2)
        @test x == Array(a)

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
        ntuple_test(get_backend(result))(result, Val(3); ndrange = 1)
        @test Array(result)[1] == (77, 2*77, 3*77)
        x = 88f0
        ntuple_closure(get_backend(result))(result, Val(3), x; ndrange = 1)
        @test Array(result)[1] == (x, 2*x, 3*x)
    end

    AT <: AbstractGPUArray && @testset "cartesian iteration" begin
        Ac = rand(Float32, 32, 32)
        A = AT(Ac)
        result = fill!(copy(A), 0.0f0)
        cartesian_iter(get_backend(A))(result, A; ndrange = size(A))
        Array(result) == Ac
    end

    AT <: AbstractGPUArray && @testset "Custom kernel from Julia function" begin
        x = AT(rand(Float32, 100))
        y = AT(rand(Float32, 100))
        clmap!(get_backend(x))(-, x, y; ndrange = size(x))
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

        # Test inputs.
        x = rand(Float32, 10)
        xmat = rand(Float32, 2, 10)
        xarr = rand(Float32, 3, 2, 10)

        # Note: When testing repeat(x; inner) need to hit both `repeat_inner_src_kernel!`
        # and `repeat_inner_dst_kernel!` to get full coverage.

        # Inner.
        @test compare(a -> repeat(a, inner=(2, )), AT, x)
        @test compare(a -> repeat(a, inner=(2, 3)), AT, xmat)
        @test compare(a -> repeat(a, inner=(3, 2)), AT, xmat)
        @test compare(a -> repeat(a, inner=(2, 3, 4)), AT, xarr)
        @test compare(a -> repeat(a, inner=(4, 3, 2)), AT, xarr)
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
        @test compare(a -> repeat(a, inner=(3, 1, 2)), AT, x)
        @test compare(a -> repeat(a, outer=(2, 1, 3)), AT, x)
        @test compare(a -> repeat(a, inner=(2, 1, 3), outer=(2, 2, 3)), AT, x)

        @test compare(a -> repeat(a, inner=(2, 1, 3)), AT, xmat)
        @test compare(a -> repeat(a, inner=(3, 1, 2)), AT, xmat)
        @test compare(a -> repeat(a, outer=(2, 1, 3)), AT, xmat)
        @test compare(a -> repeat(a, inner=(2, 1, 3), outer=(2, 2, 3)), AT, xmat)
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

    @testset "view" begin
      @test compare(AT, rand(Float32, 5)) do x
        y = x[2:4]
        y .= 1
        x
      end

      @test compare(AT, rand(Float32, 5)) do x
        y = view(x, 2:4)
        y .= 1
        x
      end

      @test compare(x->view(x, :, 1:4, 3), AT, rand(Float32, 5, 4, 3))

      let x = AT(rand(Float32, 5, 4, 3))
        @test_throws BoundsError view(x, :, :, 1:10)
      end

      # bug in parentindices conversion
      let x = AT{Int}(undef, 1, 1)
        x[1,:] .= 42
        @test Array(x)[1,1] == 42
      end

      # bug in conversion of indices (#506)
      show(devnull, AT(view(ones(Float32, 1), [1])))

      # performance loss due to Array indices
      let x = AT{Int}(undef, 1)
        i = [1]
        y = view(x, i)
        @test parent(y) isa AT
        @test parentindices(y) isa Tuple{<:AT}
      end

      @testset "GPU array source" begin
          a = rand(Float32, 3)
          i = rand(1:3, 2)
          @test compare(view, AT, a, i)
          @test compare(view, AT, a, view(i, 2:2))
      end

      @testset "CPU array source" begin
          a = rand(Float32, 3)
          i = rand(1:3, 2)
          @test compare(view, AT, a, i)
          @test compare(view, AT, a, view(i, 2:2))
      end
    end

    @testset "reshape" begin
      A = [1 2 3 4
           5 6 7 8]
      gA = reshape(AT(A),1,8)
      _A = reshape(A,1,8)
      _gA = Array(gA)
      @test all(_A .== _gA)
      A = [1,2,3,4]
      gA = reshape(AT(A),4)
    end

    @testset "reverse" begin
      # 1-d out-of-place
      @test compare(x->reverse(x), AT, rand(Float32, 1000))
      @test compare(x->reverse(x, 10), AT, rand(Float32, 1000))
      @test compare(x->reverse(x, 10, 90), AT, rand(Float32, 1000))

      # 1-d in-place
      @test compare(x->reverse!(x), AT, rand(Float32, 1000))
      @test compare(x->reverse!(x, 10), AT, rand(Float32, 1000))
      @test compare(x->reverse!(x, 10, 90), AT, rand(Float32, 1000))

      # n-d out-of-place
      for shape in ([1, 2, 4, 3], [4, 2], [5], [0], [1], [2^5, 2^5, 2^5]),
          dim in 1:length(shape)
          @testset "Shape: $shape, Dim: $dim"
              @test compare(x->reverse(x; dims=dim), AT, rand(Float32, shape...))

              cpu = rand(Float32, shape...)
              gpu = AT(cpu)
              reverse!(gpu; dims=dim)
              @test Array(gpu) == reverse(cpu; dims=dim)
          end
      end

      # supports multidimensional reverse
      for shape in ([1,1,1,1], [1, 2, 4, 3], [2^5, 2^5, 2^5]),
          dim in ((1,2),(2,3),(1,3),:)
          @testset "Shape: $shape, Dim: $dim"
              @test compare(x->reverse(x; dims=dim), AT, rand(Float32, shape...))

              cpu = rand(Float32, shape...)
              gpu = AT(cpu)
              reverse!(gpu; dims=dim)
              @test Array(gpu) == reverse(cpu; dims=dim)
          end
      end

      # wrapped array
      @test compare(x->reverse(x), AT, reshape(rand(Float32, 2,2), 4))

      # error throwing
      cpu = rand(Float32, 1,2,3,4)
      gpu = AT(cpu)
      @test_throws ArgumentError reverse!(gpu, dims=5)
      @test_throws ArgumentError reverse!(gpu, dims=0)
      @test_throws ArgumentError reverse(gpu, dims=5)
      @test_throws ArgumentError reverse(gpu, dims=0)
  end

    @testset "reinterpret" begin
      A = Int32[-1,-2,-3]
      dA = AT(A)
      dB = reinterpret(UInt32, dA)
      @test reinterpret(UInt32, A) == Array(dB)

      @test collect(reinterpret(Int32, AT(fill(1f0))))[] == reinterpret(Int32, 1f0)

      @testset "reinterpret(reshape)" begin
        a = AT(ComplexF32[1.0f0+2.0f0*im, 2.0f0im, 3.0f0im])
        b = reinterpret(reshape, Float32, a)
        @test a isa AT{ComplexF32, 1}
        if AT <: AbstractGPUArray
          # only GPUArrays materialize the reinterpret(reshape) wrapper
          @test b isa AT{Float32, 2}
        end
        @test(Array(b) == [1.0 0.0 0.0; 2.0 2.0 3.0],
              broken=(AT <: Array &&
                        (v"1.11.0-DEV.727" <= VERSION < v"1.11.0-beta2" || # broken in JuliaLang/julia#51760 & reverted in beta2
                          v"1.12.0-" <= VERSION < v"1.12.0-DEV.528"
                        )
                      )
              )

        a = AT(Float32[1.0 0.0 0.0; 2.0 2.0 3.0])
        b = reinterpret(reshape, ComplexF32, a)
        @test Array(b) == ComplexF32[1.0f0+2.0f0*im, 2.0f0im, 3.0f0im]
      end

      if AT <: AbstractGPUArray
        # XXX: use a predicate function?
        supports_bitsunion = try
          AT([1,nothing])
          true
        catch
          false
        end

        if supports_bitsunion
          @test_throws "cannot reinterpret an `Union{Nothing, Int64}` array to `Float64`, because not all types are bitstypes" reinterpret(Float64, AT([1,nothing]))
        end

        @test_throws "cannot reinterpret a zero-dimensional `Float32` array to `Int128` which is of a different size" reinterpret(Int128, AT(fill(1f0)))

        @test_throws "cannot reinterpret an `Float32` array to `Int128` whose first dimension has size `3`." reinterpret(Int128, AT(ones(Float32, 3)))
      end
    end
end
