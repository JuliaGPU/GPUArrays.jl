function test_linalg(AT)
    @testset "linear algebra" begin
        @testset "adjoint and transpose" begin
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
            @testset for uplo in ('U', 'L')
                @test compare(x -> LinearAlgebra.copytri!(x, uplo), AT, rand(Float32, 128, 128))
            end
        end

        @testset "copyto! for triangular" begin
            ga = Array{Float32}(undef, 128, 128)
            gb = AT{Float32}(undef, 128, 128)
            rand!(gb)
            copyto!(ga, UpperTriangular(gb))
            @test ga == Array(collect(UpperTriangular(gb)))
            ga = Array{Float32}(undef, 128, 128)
            gb = AT{Float32}(undef, 128, 128)
            rand!(gb)
            copyto!(ga, LowerTriangular(gb))
            @test ga == Array(collect(LowerTriangular(gb)))
        end

        @testset "permutedims" begin
            @test compare(x -> permutedims(x, (2, 1)), AT, rand(Float32, 2, 3))
            @test compare(x -> permutedims(x, (2, 1, 3)), AT, rand(Float32, 4, 5, 6))
            @test compare(x -> permutedims(x, (3, 1, 2)), AT, rand(Float32, 4, 5, 6))
            @test compare(x -> permutedims(x, [2,1,4,3]), AT, randn(ComplexF64,3,4,5,1))
        end

        @testset "issymmetric/ishermitian" begin
            n = 128
            areal = randn(n,n)/2
            aimg  = randn(n,n)/2

            @testset for eltya in (Float32, Float64, ComplexF32, ComplexF64)
                a = convert(Matrix{eltya}, eltya <: Complex ? complex.(areal, aimg) : areal)
                asym = transpose(a) + a        # symmetric indefinite
                aherm = a' + a                 # Hermitian indefinite
                @test issymmetric(asym)
                @test ishermitian(aherm)
            end
        end
        @testset "Array + Diagonal" begin
            n = 128
            A = AT{Float32}(undef, n, n)
            d = AT{Float32}(undef, n)
            rand!(A)
            rand!(d)
            D = Diagonal(d)
            B = A + D
            @test collect(B) ≈ collect(A) + collect(D)
        end

        @testset "$f! with diagonal $d" for (f, f!) in ((triu, triu!), (tril, tril!)),
                                            d in -2:2
            A = randn(10, 10)
            @test f(A, d) == Array(f!(AT(A), d))
        end

        @testset "$T gemv y := $f(A) * x * a + y * b" for f in (identity, transpose, adjoint), T in supported_eltypes()
            y, A, x = rand(T, 4), rand(T, 4, 4), rand(T, 4)

            # workaround for https://github.com/JuliaLang/julia/issues/35163#issue-584248084
            T <: Integer && (y .%= T(10); A .%= T(10); x .%= T(10))
            
            @test compare(*, AT, f(A), x)
            @test compare(mul!, AT, y, f(A), x)
            @test compare(mul!, AT, y, f(A), x, Ref(T(4)), Ref(T(5)))
        end

        @testset "$T gemm C := $f(A) * $g(B) * a + C * b" for f in (identity, transpose, adjoint), g in (identity, transpose, adjoint), T in supported_eltypes()
            A, B, C = rand(T, 4, 4), rand(T, 4, 4), rand(T, 4, 4)

            # workaround for https://github.com/JuliaLang/julia/issues/35163#issue-584248084
            T <: Integer && (A .%= T(10); B .%= T(10); C .%= T(10))
            
            @test compare(*, AT, f(A), g(B))
            @test compare(mul!, AT, C, f(A), g(B))
            @test compare(mul!, AT, C, f(A), g(B), Ref(T(4)), Ref(T(5)))
        end

        @testset "lmul! and rmul!" for (a,b) in [((3,4),(4,3)), ((3,), (1,3)), ((1,3), (3))], T in supported_eltypes()
            @test compare(rmul!, AT, rand(T, a), Ref(rand(T)))
            @test compare(lmul!, AT, Ref(rand(T)), rand(T, b))
        end
    end
end
