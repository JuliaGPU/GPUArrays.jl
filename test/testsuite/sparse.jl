@testsuite "sparse" (AT, eltypes)->begin
    sparse_ATs = sparse_types(AT)
    for sparse_AT in sparse_ATs
        if sparse_AT <: AbstractSparseVector
            vector(sparse_AT, eltypes)
            vector_construction(sparse_AT, eltypes)
            broadcasting_vector(sparse_AT, eltypes)
        elseif sparse_AT <: AbstractSparseMatrix
            matrix(sparse_AT, eltypes)
            matrix_construction(sparse_AT, eltypes)
            broadcasting_matrix(sparse_AT, eltypes)
            mapreduce_matrix(sparse_AT, eltypes)
            linalg(sparse_AT, eltypes)
       end
    end
end

using SparseArrays
using SparseArrays: nonzeroinds, nonzeros, rowvals

function vector(AT, eltypes)
    dense_AT = GPUArrays.dense_array_type(AT)
    for ET in eltypes
        @testset "Sparse vector properties($ET)" begin
            m = 25
            n = 35
            k = 10
            p = 5
            blockdim = 5
            x   = sprand(ET,m,0.2)
            d_x = AT(x)
            @test length(d_x) == m
            @test size(d_x)   == (m,)
            @test size(d_x,1) == m
            @test size(d_x,2) == 1
            @test ndims(d_x)  == 1
            dense_d_x  = dense_AT(x)
            dense_d_x2 = dense_AT(d_x)
            @allowscalar begin
                @test Array(d_x[:])        == x[:]
                @test d_x[firstindex(d_x)] == x[firstindex(x)]
                @test d_x[div(end, 2)]     == x[div(end, 2)]
                @test d_x[end]             == x[end]
                @test Array(d_x[firstindex(d_x):end]) == x[firstindex(x):end]
                @test Array(dense_d_x[firstindex(d_x):end]) == x[firstindex(x):end]
                @test Array(dense_d_x2[firstindex(d_x):end]) == x[firstindex(x):end]
            end
            @test_throws BoundsError d_x[firstindex(d_x) - 1]
            @test_throws BoundsError d_x[end + 1]
            @test nnz(d_x)    == nnz(x)
            @test Array(nonzeros(d_x)) == nonzeros(x)
            @test Array(nonzeroinds(d_x)) == nonzeroinds(x)
            @test Array(rowvals(d_x)) == nonzeroinds(x)
            @test nnz(d_x)    == length(nonzeros(d_x))
        end
    end
end

function vector_construction(AT, eltypes)
    dense_AT = GPUArrays.dense_array_type(AT)
    for ET in eltypes
        m = 25
        n = 35
        k = 10
        p = 5
        blockdim = 5
        x = sprand(ET, m, 0.2)
        d_x = AT(x)
        @test collect(d_x) == collect(x)
        @test similar(d_x) isa AT{ET}
        @test similar(d_x, Float32) isa AT{Float32}
    end
end

function matrix(AT, eltypes)
    dense_AT = GPUArrays.dense_array_type(AT)
    for ET in eltypes
        @testset "Sparse matrix properties($ET)" begin
            m = 25
            n = 35
            k = 10
            p = 5
            blockdim = 5
            x    = sprand(ET, m, 0.2)
            d_x  = AT(x)
            @test size(d_x) == (m, 1)
            x    = sprand(ET, m, n, 0.2)
            d_x  = AT(x)
            d_tx = AT(transpose(x))
            d_ax = AT(adjoint(x))
            @test size(d_tx)   == (n, m)
            @test size(d_ax)   == (n, m)
            @test length(d_x)  == m*n
            @test size(d_x)    == (m, n)
            @test size(d_x, 1) == m
            @test size(d_x, 2) == n
            @test size(d_x, 3) == 1
            @test ndims(d_x)   == 2
            @allowscalar begin
                @test d_x[:, :]            == x[:, :]
                @test d_tx[:, :]           == transpose(x)[:, :]
                @test d_ax[:, :]           == adjoint(x)[:, :]
                @test d_x[(1, 1)]          == x[1, 1]
                @test d_x[firstindex(d_x)] == x[firstindex(x)]
                @test d_x[div(end, 2)]     == x[div(end, 2)]
                @test d_x[end]             == x[end]
                @test d_x[firstindex(d_x), firstindex(d_x)] == x[firstindex(x), firstindex(x)]
                @test d_x[div(end, 2), div(end, 2)]         == x[div(end, 2), div(end, 2)]
                @test d_x[end, end]        == x[end, end]
                for i in 1:size(x, 2)
                    @test Array(d_x[:, i]) == x[:, i]
                end
            end
            @test_throws BoundsError d_x[firstindex(d_x) - 1]
            @test_throws BoundsError d_x[end + 1]
            @test_throws BoundsError d_x[firstindex(d_x) - 1, firstindex(d_x) - 1]
            @test_throws BoundsError d_x[end + 1, end + 1]
            @test_throws BoundsError d_x[firstindex(d_x) - 1:end + 1, :]
            @test_throws BoundsError d_x[firstindex(d_x) - 1, :]
            @test_throws BoundsError d_x[end + 1, :]
            @test_throws BoundsError d_x[:, firstindex(d_x) - 1:end + 1]
            @test_throws BoundsError d_x[:, firstindex(d_x) - 1]
            @test_throws BoundsError d_x[:, end + 1]
            @test nnz(d_x)    == nnz(x)
            @test nnz(d_x)    == length(nonzeros(d_x))
            @test !issymmetric(d_x)
            @test !ishermitian(d_x)
        end
    end
end

function matrix_construction(AT, eltypes)
    dense_AT = GPUArrays.dense_array_type(AT)
    for ET in eltypes
        m = 25
        n = 35
        k = 10
        p = 5
        blockdim = 5
        x = sprand(ET, m, n, 0.2)
        d_x = AT(x)
        @test collect(d_x) == collect(x)
        @test similar(d_x) isa AT{ET}
        @test similar(d_x, Float32) isa AT{Float32}
        @test similar(d_x, Float32, n, m) isa AT{Float32}
        @test similar(d_x, Float32, (n, m)) isa AT{Float32}
        @test similar(d_x, (3, 4)) isa AT{ET}
        @test size(similar(d_x, (3, 4))) == (3, 4)
    end
end

function broadcasting_vector(AT, eltypes)
    dense_AT = GPUArrays.dense_array_type(AT)
    for ET in eltypes
        @testset "SparseVector($ET)" begin
            m  = 64
            p  = 0.5
            x  = sprand(ET, m, p)
            dx = AT(x)

            # zero-preserving
            y  = x  .* ET(1)
            dy = dx .* ET(1)
            @test dy isa AT{ET}
            @test collect(SparseArrays.nonzeroinds(dy)) == collect(SparseArrays.nonzeroinds(dx))
            @test collect(SparseArrays.nonzeroinds(dy)) == SparseArrays.nonzeroinds(y)
            @test collect(SparseArrays.nonzeros(dy))    == SparseArrays.nonzeros(y)
            @test y == SparseVector(dy)

            # not zero-preserving
            y  = x  .+ ET(1)
            dy = dx .+ ET(1)
            @test dy isa dense_AT{ET}
            hy = Array(dy)
            @test Array(y) == hy

            # involving something dense
            y  = x  .+ ones(ET, m)
            dy = dx .+ dense_AT(ones(ET, m))
            @test dy isa dense_AT{ET}
            @test Array(y) == Array(dy)

            # sparse to sparse
            dx = AT(x)
            y  = sprand(ET, m, p)
            dy = AT(y)
            z  = x  .* y
            dz = dx .* dy
            @test dz isa AT{ET}
            @test z == SparseVector(dz)

            # multiple inputs
            y  = sprand(ET, m, p)
            w  = sprand(ET, m, p)
            dy = AT(y)
            dx = AT(x)
            dw = AT(w)
            z  = @. x  * y  * w
            dz = @. dx * dy * dw
            @test dz isa AT{ET}
            @test z == SparseVector(dz)

            y = sprand(ET, m, p)
            w = sprand(ET, m, p)
            dense_arr   = rand(ET, m)
            d_dense_arr = dense_AT(dense_arr)
            dy = AT(y)
            dw = AT(w)
            z  = @. x  * y  * w  * dense_arr
            dz = @. dx * dy * dw * d_dense_arr
            @test dz isa dense_AT{ET}
            @test Array(z) == Array(dz)
            
            y  = sprand(ET, m, p)
            dy = AT(y)
            dx = AT(x)
            z  = x  .* y  .* ET(2)
            dz = dx .* dy .* ET(2)
            @test dz isa AT{ET}
            @test z == SparseVector(dz)

            # type-mismatching
            ## non-zero-preserving
            dx = AT(x)
            dy = dx .+ 1
            y  = x .+ 1
            @test dy isa dense_AT{promote_type(ET, Int)}
            @test Array(y) == Array(dy)
            ## zero-preserving
            dy = dx .* 1
            y  = x  .* 1
            @test dy isa AT{promote_type(ET, Int)}
            @test collect(SparseArrays.nonzeroinds(dy))  == collect(SparseArrays.nonzeroinds(dx))
            @test collect(SparseArrays.nonzeroinds(dy))  == SparseArrays.nonzeroinds(y)
            @test collect(SparseArrays.nonzeros(dy)) == SparseArrays.nonzeros(y)
            @test y == SparseVector(dy)
        end
    end
end

function broadcasting_matrix(AT, eltypes)
    dense_AT = GPUArrays.dense_array_type(AT)
    for ET in eltypes
       @testset "SparseMatrix($ET)" begin
            m, n = 5, 6
            p   = 0.5
            x   = sprand(ET, m, n, p)
            dx  = AT(x)
            # zero-preserving
            y  = x  .* ET(1)
            dy = dx .* ET(1)
            @test dy isa AT{ET}
            @test y == SparseMatrixCSC(dy)

            # not zero-preserving
            y  = x  .+ ET(1)
            dy = dx .+ ET(1)
            @test dy isa dense_AT{ET}
            hy = Array(dy)
            dense_y = Array(y)
            @test Array(y) == Array(dy)

            # involving something dense
            y  = x  .* ones(ET, m, n)
            dy = dx .* dense_AT(ones(ET, m, n))
            @test dy isa dense_AT{ET}
            @test Array(y) == Array(dy)
            
            # multiple inputs
            y  = sprand(ET, m, n, p)
            dy = AT(y)
            z  = x  .* y  .* ET(2)
            dz = dx .* dy .* ET(2)
            @test dz isa AT{ET}
            @test z == SparseMatrixCSC(dz)

            # multiple inputs
            w  = sprand(ET, m, n, p)
            dw = AT(w)
            z  = x  .* y  .* w
            dz = dx .* dy .* dw
            @test dz isa AT{ET}
            @test z == SparseMatrixCSC(dz)
        end
    end
end

function mapreduce_matrix(AT, eltypes)
    dense_AT = GPUArrays.dense_array_type(AT)
    for ET in eltypes
        @testset "SparseMatrix($ET)" begin
            m,n = 5,6
            p = 0.5
            x = sprand(ET, m, n, p)
            dx = AT(x)

            # dim=:
            y  = sum(x)
            dy = sum(dx)
            @test y ≈ dy

            y  = mapreduce(abs, +, x)
            dy = mapreduce(abs, +, dx)
            @test y ≈ dy

            # dim=1
            y  = sum(x, dims=1)
            dy = sum(dx, dims=1)
            @test y ≈ Array(dy)

            y  = mapreduce(abs, +, x, dims=1)
            dy = mapreduce(abs, +, dx, dims=1)
            @test y ≈ Array(dy)

            # dim=2
            y = sum(x, dims=2)
            dy = sum(dx, dims=2)
            @test y ≈ Array(dy)

            y  = mapreduce(abs, +, x, dims=2)
            dy = mapreduce(abs, +, dx, dims=2)
            @test y ≈ Array(dy)
            if ET in (Float32, Float64)
                dy = mapreduce(abs, +, dx; init=zero(ET))
                y  = mapreduce(abs, +, x; init=zero(ET))
                @test y ≈ dy
            end

            # test with a matrix with fully empty rows
            x = zeros(ET, m, n)
            x[2, :] .= -one(ET)
            x[2, end] = -ET(16)
            dx = AT(sparse(x))
            y  = mapreduce(abs, max, x)
            dy = mapreduce(abs, max, dx)
            @test y ≈ dy
        end
    end
end

function linalg(AT, eltypes)
    dense_AT = GPUArrays.dense_array_type(AT)
    for ET in eltypes
        # sprandn doesn't work nicely with these...
        if !(ET <: Union{Complex{Int16}, Complex{Int32}, Complex{Int64}})
            @testset "Sparse matrix($ET) linear algebra" begin
                m = 10
                A  = sprandn(ET, m, m, 0.2)
                B  = sprandn(ET, m, m, 0.3)
                ZA = spzeros(ET, m, m)
                C  = I(div(m, 2))
                dA = AT(A)
                dB = AT(B)
                dZA = AT(ZA)
                @testset "opnorm and norm" begin
                    @test opnorm(A, Inf) ≈ opnorm(dA, Inf)
                    @test opnorm(A, 1)   ≈ opnorm(dA, 1)
                    @test_throws ArgumentError opnorm(dA, 2)
                end
            end
        end
    end
end
