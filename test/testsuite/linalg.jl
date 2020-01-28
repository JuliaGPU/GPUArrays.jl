function test_linalg(AT)
    @testset "linear algebra" begin
        @testset "transpose" begin
            @test compare(adjoint, AT, rand(Float32, 32, 32))
            @test compare(transpose, AT, rand(Float32, 32, 32))
            @test compare(transpose!, AT, Array{Float32}(undef, 32, 32), rand(Float32, 32, 32))
            @test compare(transpose!, AT, Array{Float32}(undef, 128, 32), rand(Float32, 32, 128))
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

        @testset "matrix multiplication" begin
            a = rand(Int8, 3, 3)
            b = rand(Int8, 3, 3)
            d_a = AT{Int8}(a)
            d_b = AT{Int8}(b)
            d_c = d_a*d_b
            @test collect(d_c) == a*b
            a = rand(Complex{Int8}, 3, 3)
            b = rand(Complex{Int8}, 3, 3)
            d_a = AT{Complex{Int8}}(a)
            d_b = AT{Complex{Int8}}(b)
            d_c = d_a'*d_b
            @test collect(d_c) == a'*b
            d_c = d_a*d_b'
            @test collect(d_c) == a*b'
            d_c = d_a'*d_b'
            @test collect(d_c) == a'*b'
            d_c = transpose(d_a)*d_b'
            @test collect(d_c) == transpose(a)*b'
            d_c = d_a'*transpose(d_b)
            @test collect(d_c) == a'*transpose(b)
            d_c = transpose(d_a)*d_b
            @test collect(d_c) == transpose(a)*b
            d_c = d_a*transpose(d_b)
            @test collect(d_c) == a*transpose(b)
            d_c = transpose(d_a)*transpose(d_b)
            @test collect(d_c) == transpose(a)*transpose(b)
            d_c = rmul!(copy(d_a), Complex{Int8}(2, 2))
            @test collect(d_c) == a*Complex{Int8}(2, 2)
            d_c = lmul!(Complex{Int8}(2, 2), copy(d_a))
            @test collect(d_c) == Complex{Int8}(2, 2)*a
        end
    end
end
