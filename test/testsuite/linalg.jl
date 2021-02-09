@testsuite "linear algebra" AT->begin
    @testcase "adjoint and transpose" begin
        @test compare(adjoint, AT, rand(Float32, 32, 32))
        @test compare(adjoint!, AT, rand(Float32, 32, 32), rand(Float32, 32, 32))
        @test compare(transpose, AT, rand(Float32, 32, 32))
        @test compare(transpose!, AT, rand(Float32, 32, 32), rand(Float32, 32, 32))
        @test compare((x,y)->copyto!(x, adjoint(y)), AT, rand(Float32, 32, 32), rand(Float32, 32, 32))
        @test compare((x,y)->copyto!(x, transpose(y)), AT, rand(Float32, 32, 32), rand(Float32, 32, 32))
        @test compare(transpose!, AT, Array{Float32}(undef, 32, 32), rand(Float32, 32, 32))
        @test compare(transpose!, AT, Array{Float32}(undef, 128, 32), rand(Float32, 32, 128))
    end

    @testset "copytri!" begin
        @testcase for uplo in ('U', 'L')
            @test compare(x -> LinearAlgebra.copytri!(x, uplo), AT, rand(Float32, 128, 128))
        end
    end

    @testcase "copyto! for triangular" begin
        for TR in (UpperTriangular, LowerTriangular)
            @test compare(transpose!, AT, Array{Float32}(undef, 128, 32), rand(Float32, 32, 128))

            cpu_a = Array{Float32}(undef, 128, 128)
            gpu_a = AT{Float32}(undef, 128, 128)
            gpu_b = AT{Float32}(undef, 128, 128)

            rand!(gpu_a)
            copyto!(cpu_a, TR(gpu_a))
            @test cpu_a == Array(collect(TR(gpu_a)))

            rand!(gpu_a)
            gpu_c = copyto!(gpu_b, TR(gpu_a))
            @test all(Array(gpu_b) .== TR(Array(gpu_a)))
            @test gpu_c isa AT
        end

        for TR in (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular)
            gpu_a = AT{Float32}(undef, 128, 128) |> rand! |> TR
            gpu_b = AT{Float32}(undef, 128, 128) |> TR

            gpu_c = copyto!(gpu_b, gpu_a)
            @test all(Array(gpu_b) .== Array(gpu_a))
            @test all(Array(gpu_c) .== Array(gpu_a))
            @test gpu_c isa TR
        end
    end

    @testcase "permutedims" begin
        @test compare(x -> permutedims(x, (2, 1)), AT, rand(Float32, 2, 3))
        @test compare(x -> permutedims(x, (2, 1, 3)), AT, rand(Float32, 4, 5, 6))
        @test compare(x -> permutedims(x, (3, 1, 2)), AT, rand(Float32, 4, 5, 6))
        @test compare(x -> permutedims(x, [2,1,4,3]), AT, randn(ComplexF64,3,4,5,1))
    end

    @testset "issymmetric/ishermitian" begin
        n = 128
        areal = randn(n,n)/2
        aimg  = randn(n,n)/2

        @testcase for eltya in (Float32, Float64, ComplexF32, ComplexF64)
            a = convert(Matrix{eltya}, eltya <: Complex ? complex.(areal, aimg) : areal)
            asym = transpose(a) + a        # symmetric indefinite
            aherm = a' + a                 # Hermitian indefinite
            @test issymmetric(asym)
            @test ishermitian(aherm)
        end
    end

    @testcase "Array + Diagonal" begin
        n = 128
        A = AT{Float32}(undef, n, n)
        d = AT{Float32}(undef, n)
        rand!(A)
        rand!(d)
        D = Diagonal(d)
        B = A + D
        @test collect(B) ≈ collect(A) + collect(D)
    end

    @testcase "$f! with diagonal $d" for
            (f, f!) in ((triu, triu!), (tril, tril!)),
            d in -2:2
        A = randn(10, 10)
        @test f(A, d) == Array(f!(AT(A), d))
    end

    @testcase "$T gemv y := $f(A) * x * a + y * b" for
            f in (identity, transpose, adjoint),
            T in supported_eltypes()
        y, A, x = rand(T, 4), rand(T, 4, 4), rand(T, 4)

        # workaround for https://github.com/JuliaLang/julia/issues/35163#issue-584248084
        T <: Integer && (y .%= T(10); A .%= T(10); x .%= T(10))

        @test compare(*, AT, f(A), x)
        @test compare(mul!, AT, y, f(A), x)
        @test compare(mul!, AT, y, f(A), x, Ref(T(4)), Ref(T(5)))
        @test typeof(AT(rand(3, 3)) * AT(rand(3))) <: AbstractVector

        if f !== identity
            @test compare(mul!, AT, rand(T, 2,2), rand(T, 2,1), f(rand(T, 2)))
        end
    end

    @testcase "$T gemm C := $f(A) * $g(B) * a + C * b" for
            f in (identity, transpose, adjoint),
            g in (identity, transpose, adjoint),
            T in supported_eltypes()
        A, B, C = rand(T, 4, 4), rand(T, 4, 4), rand(T, 4, 4)

        # workaround for https://github.com/JuliaLang/julia/issues/35163#issue-584248084
        T <: Integer && (A .%= T(10); B .%= T(10); C .%= T(10))

        @test compare(*, AT, f(A), g(B))
        @test compare(mul!, AT, C, f(A), g(B))
        @test compare(mul!, AT, C, f(A), g(B), Ref(T(4)), Ref(T(5)))
        @test typeof(AT(rand(3, 3)) * AT(rand(3, 3))) <: AbstractMatrix
    end

    @testcase "lmul! and rmul! $a x $b with $T" for
            (a,b) in [((3,4),(4,3)), ((3,), (1,3)), ((1,3), (3))],
            T in supported_eltypes()
        @test compare(rmul!, AT, rand(T, a), Ref(rand(T)))
        @test compare(lmul!, AT, Ref(rand(T)), rand(T, b))
    end
end
