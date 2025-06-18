@testsuite "linalg" (AT, eltypes)->begin
    @testset "adjoint and transpose" begin
        @test compare(adjoint, AT, rand(Float32, 32, 32))
        @test compare(adjoint!, AT, rand(Float32, 32, 32), rand(Float32, 32, 32))
        @test compare(adjoint!, AT, rand(Float32, 1, 32), rand(Float32, 32))
        @test compare(adjoint!, AT, rand(Float32, 32), rand(Float32, 1, 32))
        @test compare(transpose, AT, rand(Float32, 32, 32))
        @test compare(transpose!, AT, rand(Float32, 32, 32), rand(Float32, 32, 32))
        @test compare(transpose!, AT, rand(Float32, 1, 32), rand(Float32, 32))
        @test compare(transpose!, AT, rand(Float32, 32), rand(Float32, 1, 32))
        @test compare((x,y)->copyto!(x, adjoint(y)), AT, rand(Float32, 32, 32), rand(Float32, 32, 32))
        @test compare((x,y)->copyto!(x, transpose(y)), AT, rand(Float32, 32, 32), rand(Float32, 32, 32))
        @test compare(transpose!, AT, Array{Float32}(undef, 32, 32), rand(Float32, 32, 32))
        @test compare(transpose!, AT, Array{Float32}(undef, 128, 32), rand(Float32, 32, 128))
    end

    @testset "permutedims" begin
        @test compare(x -> permutedims(x, (2, 1)), AT, rand(Float32, 2, 3))
        @test compare(x -> permutedims(x, (2, 1, 3)), AT, rand(Float32, 4, 5, 6))
        @test compare(x -> permutedims(x, (3, 1, 2)), AT, rand(Float32, 4, 5, 6))
        @test compare(x -> permutedims(x, [2,1,4,3]), AT, randn(ComplexF32,3,4,5,1))
        # test UInt64 version to make sure it works properly when array length is larger than typemax of UInt32.
        AT <: GPUArrays.AbstractGPUArray && @test let
            x = randn(ComplexF32,3,4,5,1)
            y = permutedims(x, (2,1,4,3))
            Array(GPUArrays._permutedims!(UInt64, AT(zero(y)), AT(x), (2,1,4,3))) ≈ y
        end
        # high dimensional tensor
        @test compare(x -> permutedims(x, 18:-1:1), AT, rand(Float32, 4, [2 for _ = 2:18]...))
        # test the Uint64 type version for large array permutedims
        AT <: GPUArrays.AbstractGPUArray && @test let
            x = rand(Float32, 4, [2 for _ = 2:18]...)
            pm = (18:-1:1...,)
            y = permutedims(x, pm)
            Array(GPUArrays._permutedims!(UInt64, AT(zero(y)), AT(x), pm)) ≈ y
        end
    end


    @testset "symmetric" begin
        @testset "Hermitian" begin
            A    = rand(Float32,2,2)
            A    = A*A'+I #posdef
            d_A  = AT(A)
            similar(Hermitian(d_A, :L), Float32)
        end

        @testset "issymmetric/ishermitian" begin
            n = 128
            areal = randn(n,n)/2
            aimg  = randn(n,n)/2

            @testset for eltya in (Float32, Float64, ComplexF32, ComplexF64)
                if !(eltya in eltypes)
                    continue
                end
                a = convert(Matrix{eltya}, eltya <: Complex ? complex.(areal, aimg) : areal)
                asym = transpose(a) + a        # symmetric indefinite
                aherm = a' + a                 # Hermitian indefinite
                @test issymmetric(asym)
                @test ishermitian(aherm)
            end
        end
    end

    @testset "triangular" begin
        @testset "copytri!" begin
            @testset for eltya in (Float32, Float64, ComplexF32, ComplexF64), uplo in ('U', 'L'), conjugate in (true, false)
                n = 128
                areal = randn(n,n)/2
                aimg  = randn(n,n)/2
                if !(eltya in eltypes)
                    continue
                end
                a = convert(Matrix{eltya}, eltya <: Complex ? complex.(areal, aimg) : areal)
                @test compare(x -> LinearAlgebra.copytri!(x, uplo, conjugate), AT, a)
            end
        end

        if isdefined(LinearAlgebra, :copytrito!)
            @testset "copytrito!" begin
                @testset for T in eltypes, uplo in ('L', 'U')
                    n = 16
                    A = rand(T,n,n)
                    B = zeros(T,n,n)
                    @test compare(copytrito!, AT, B, A, uplo)
                end
                @testset for T in eltypes, uplo in ('L', 'U')
                    n = 16
                    m = 32
                    A = uplo == 'U' ? rand(T,m,n) : rand(T,n,m)
                    B = zeros(T,n,n)
                    @test compare(copytrito!, AT, B, A, uplo)
                end
            end
        end

        @testset "copyto! for triangular" begin
            for TR in (UpperTriangular, LowerTriangular)
                @test compare(transpose!, AT, Array{Float32}(undef, 128, 32), rand(Float32, 32, 128))

                cpu_a = Array{Float32}(undef, 128, 128)
                gpu_a = AT{Float32}(undef, 128, 128)
                gpu_b = AT{Float32}(undef, 128, 128)

                copyto!(gpu_a, rand(Float32, (128,128)))
                copyto!(cpu_a, TR(gpu_a))
                @test cpu_a == Array(collect(TR(gpu_a)))

                copyto!(gpu_a, rand(Float32, (128,128)))
                gpu_c = copyto!(gpu_b, TR(gpu_a))
                @test all(Array(gpu_b) .== TR(Array(gpu_a)))
                @test gpu_c isa AT
            end

            for TR in (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular)
                gpu_a = AT(rand(Float32, (128,128))) |> TR
                gpu_b = AT{Float32}(undef, 128, 128) |> TR

                gpu_c = copyto!(gpu_b, gpu_a)
                @test all(Array(gpu_b) .== Array(gpu_a))
                @test all(Array(gpu_c) .== Array(gpu_a))
                @test gpu_c isa TR
            end
        end

        @testset "istril" begin
            for rows in 3:4, cols in 3:4, diag in -4:4
                A = tril(rand(Float32, rows,cols), diag)
                B = AT(A)
                @test istril(A) == istril(B)
            end
        end

        @testset "istriu" begin
            for rows in 3:4, cols in 3:4, diag in -4:4
                A = triu(rand(Float32, rows,cols), diag)
                B = AT(A)
                @test istriu(A) == istriu(B)
            end
        end

        @testset "mul! + Triangular" begin
            @testset "trimatmul! ($TR x $T, $f)" for T in (Float32, ComplexF32), TR in (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular), f in (identity, transpose, adjoint)
                n = 128
                A = AT(rand(T, n,n))
                b = AT(rand(T, n))
                Ct = AT(zeros(T, n))
                C = zeros(T, n)
                mul!(Ct, f(TR(A)), b)
                mul!(C, f(TR(collect(A))), collect(b))
                @test collect(Ct) ≈ C

                B = AT(rand(T, n, n))
                Ct = AT(zeros(T, n, n))
                C = zeros(T, n, n)
                mul!(Ct, f(TR(A)), B)
                mul!(C, f(TR(collect(A))), collect(B))
                @test collect(Ct) ≈ C
            end

            @testset "mattrimul ($TR x $T, $f)" for T in (Float32, ComplexF32), TR in (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular), f in (identity, transpose, adjoint)
                n = 128
                A = AT(rand(T, n,n))
                B = AT(rand(T, n, n))
                Ct = AT(zeros(T, n, n))
                C = zeros(T, n, n)
                mul!(Ct, A, f(TR(B)))
                mul!(C, collect(A), f(TR(collect(B))))
                @test collect(Ct) ≈ C
            end
        end
    end

    @testset "diagonal" begin
        @testset "Array + Diagonal" begin
            n = 128
            A = AT(rand(Float32, (n,n)))
            d = AT(rand(Float32, n))
            D = Diagonal(d)
            B = A + D
            @test collect(B) ≈ collect(A) + collect(D)
        end

        @testset "copy diagonal" begin
            a = AT(rand(Float32, 10))
            D = Diagonal(a)
            C = copy(D)
            @test C isa Diagonal
            @test C.diag isa AT
            @test collect(D) == collect(C)
        end

        @testset "cholesky + Diagonal" begin
            n = 128
            # Add one in order prevent failures due to random numbers being zero
            d = AT(zeros(Float32, n) .+ one(Float32))
            D = Diagonal(d)
            F = collect(D)
            @test collect(cholesky(D).U) ≈ collect(cholesky(F).U)
            @test collect(cholesky(D).L) ≈ collect(cholesky(F).L)

            d = AT([1f0, 2f0, -1f0, 0f0])
            D = Diagonal(d)
            @test cholesky(D, check = false).info == 3
        end

        @testset "\\ + Diagonal" begin
            n = 128
            d = AT(rand(Float32, n))
            D = Diagonal(d)
            b = AT(rand(Float32, n))
            B = AT(rand(Float32, n, n))
            @test collect(D \ b) ≈ Diagonal(collect(d)) \ collect(b)
            @test collect(D \ B) ≈ Diagonal(collect(d)) \ collect(B)

            d = ones(Float32, n)
            d[rand(1:n)] = 0
            d = AT(d)
            D = Diagonal(d)
            @test_throws SingularException D \ B
        end

        @testset "mul! + Diagonal" begin
            for elty in (Float32, ComplexF32)
                n = 128
                d = AT(rand(elty, n))
                D = Diagonal(d)
                B = AT(rand(elty, n, n))
                X = AT(zeros(elty, n, n))
                Y = zeros(elty, n, n)
                α = rand(elty)
                β = rand(elty)
                mul!(X, D, B)
                mul!(Y, Diagonal(collect(d)), collect(B))
                @test collect(X) ≈ Y
                mul!(X, D, B, α, β)
                mul!(Y, Diagonal(collect(d)), collect(B), α, β)
                @test collect(X) ≈ Y
                mul!(X, B, D)
                mul!(Y, collect(B), Diagonal(collect(d)))
                @test collect(X) ≈ Y
                mul!(X, B, D, α, β)
                mul!(Y, collect(B), Diagonal(collect(d)), α, β)
                @test collect(X) ≈ Y
            end
        end

        @testset "ldiv! + Diagonal" begin
            n = 128
            d = AT(rand(Float32, n))
            D = Diagonal(d)
            b = AT(rand(Float32, n))
            B = AT(rand(Float32, n, n))
            X = AT(zeros(Float32, n, n))
            Y = zeros(Float32, n, n)
            ldiv!(X, D, B)
            ldiv!(Y, Diagonal(collect(d)), collect(B))
            @test collect(X) ≈ Y
            ldiv!(D, B)
            @test collect(B) ≈ collect(X)

            d = ones(Float32, n)
            d[rand(1:n)] = 0
            d = AT(d)
            D = Diagonal(d)
            B = AT(rand(Float32, n, n))

            @test_throws SingularException ldiv!(X, D, B)

            # two-argument version throws SingularException
            @test_throws SingularException ldiv!(D, B)
        end

        @testset "$f! with diagonal $d" for (f, f!) in ((triu, triu!), (tril, tril!)),
                                            d in -2:2
            A = randn(Float32, 10, 10)
            @test f(A, d) == Array(f!(AT(A), d))
        end
    end

    @testset "lmul! and rmul!" for (a,b) in [((3,4),(4,3)), ((3,), (1,3)), ((1,3), (3))], T in eltypes
        @test compare(rmul!, AT, rand(T, a), Ref(rand(T)))
        @test compare(lmul!, AT, Ref(rand(T)), rand(T, b))
    end

    @testset "axp{b}y" for T in eltypes
        @test compare(axpby!, AT, Ref(rand(T)), rand(T,5), Ref(rand(T)), rand(T,5))
        @test compare(axpy!, AT, Ref(rand(T)), rand(T,5), rand(T,5))
    end

    @testset "dot" for T in eltypes
        @test compare(dot, AT, rand(T,5), rand(T, 5))
    end

    @testset "rotate!" for T in eltypes
        @test compare(rotate!, AT, rand(T,5), rand(T,5), Ref(rand(real(T))), Ref(rand(T)))
    end

    @testset "reflect!" for T in eltypes
        @test compare(reflect!, AT, rand(T,5), rand(T,5), Ref(rand(real(T))), Ref(rand(T)))
    end

    @testset "iszero and isone" for T in eltypes
        A = one(AT(rand(T, 2, 2)))
        @test isone(A)
        @test iszero(A) == false

        A = zero(AT(rand(T, 2, 2)))
        @test iszero(A)
        @test isone(A) == false
    end

    @testset "kron" begin
        for T in eltypes
            for opa in (vec, identity, transpose, adjoint), opb in (vec, identity, transpose, adjoint)
                @test compare(kron, AT, opa(rand(T, 32, 64)), opb(rand(T, 128, 16)))
            end
        end
    end
end

@testsuite "linalg/mul!/vector-matrix" (AT, eltypes)->begin
    @testset "$T gemv y := $f(A) * x * a + y * b" for f in (identity, transpose, adjoint), T in eltypes
        y, A, x = rand(T, 4), rand(T, 4, 4), rand(T, 4)

        # workaround for https://github.com/JuliaLang/julia/issues/35163#issue-584248084
        T <: Integer && (y .%= T(10); A .%= T(10); x .%= T(10))

        @test compare(*, AT, f(A), x)
        @test compare(mul!, AT, y, f(A), x)
        @test compare(mul!, AT, y, f(A), x, Ref(T(4)), Ref(T(5)))
        @test typeof(AT(rand(T, 3, 3)) * AT(rand(T, 3))) <: AbstractVector

        if f !== identity
            @test compare(mul!, AT, rand(T, 2,2), rand(T, 2,1), f(rand(T, 2)))
        end
    end
end

@testsuite "linalg/mul!/matrix-matrix" (AT, eltypes)->begin
    @testset "$T gemm C := $f(A) * $g(B) * a + C * b" for f in (identity, transpose, adjoint), g in (identity, transpose, adjoint), T in eltypes
        A, B, C = rand(T, 4, 4), rand(T, 4, 4), rand(T, 4, 4)

        # workaround for https://github.com/JuliaLang/julia/issues/35163#issue-584248084
        T <: Integer && (A .%= T(10); B .%= T(10); C .%= T(10))

        @test compare(*, AT, f(A), g(B))
        @test compare(mul!, AT, C, f(A), g(B))
        @test compare(mul!, AT, C, f(A), g(B), Ref(T(4)), Ref(T(5)))
        @test typeof(AT(rand(T, 3, 3)) * AT(rand(T, 3, 3))) <: AbstractMatrix
    end
end

@testsuite "linalg/norm" (AT, eltypes)->begin
    @testset "$p-norm($sz x $T)" for sz in [(2,), (2,0), (2,2,2)],
                                     p in Any[0, 0.5, 1, 1.5, 2, Inf, -Inf],
                                     T in eltypes
        if T == Int8
            continue
        end
        if !in(float(real(T)), eltypes)
            # norm promotes to float, so make sure that type is supported
            continue
        end
        range = real(T) <: Integer ? (T.(1:10)) : T # prevent integer overflow
        arr = rand(range, sz)
        @test compare(norm, AT, arr, Ref(p))
        @test isrealfloattype(typeof(norm(AT(arr), p)))
        if !isempty(arr) && real(T) <: AbstractFloat && !iszero(p) && !isinf(p)
            # Hit anti-under/overflow rescaling
            @allowscalar arr[1] = floatmax(real(T)) / 2
            @test compare(norm, AT, arr, Ref(p))
            arr .= floatmin(real(T)) * 2
            @test compare(norm, AT, arr, Ref(p))
        end
    end
    @testset "$p-opnorm($sz x $T)" for sz in [(2, 0), (2, 3)],
                                     p in Any[1, Inf],
                                     T in eltypes
        if T == Int8
            continue
        end
        if !in(float(real(T)), eltypes)
            # norm promotes to float, so make sure that type is supported
            continue
        end
        range = real(T) <: Integer ? (T.(1:10)) : T # prevent integer overflow
        mat = rand(range, sz)
        @test compare(opnorm, AT, mat, Ref(p))
        @test isrealfloattype(typeof(opnorm(AT(mat), p)))
    end
end
