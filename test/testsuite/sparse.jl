@testsuite "sparse" (AT, eltypes)->begin
    sparse_ATs = sparse_types(AT)
    for sparse_AT in sparse_ATs
        if sparse_AT <: AbstractSparseVector
            vector(sparse_AT, AT, eltypes)
            vector_construction(sparse_AT, AT, eltypes)
            broadcasting_vector(sparse_AT, AT, eltypes)
            iszero_vector(sparse_AT, eltypes)
        elseif sparse_AT <: AbstractSparseMatrix
            matrix(sparse_AT, eltypes)
            matrix_construction(sparse_AT, AT, eltypes)
            broadcasting_matrix(sparse_AT, AT, eltypes)
            mapreduce_matrix(sparse_AT, eltypes)
            linalg(sparse_AT, eltypes)
            iszero_matrix(sparse_AT, eltypes)
       end
    end
    coo_AT = sparse_coo_type(AT)
    coo_AT === nothing || coo_matmul(coo_AT, AT, eltypes)
end

using SparseArrays
using SparseArrays: nonzeroinds, nonzeros, rowvals

function coo_values(::Type{T}) where {T<:Real}
    return T[1, 2, 3, 4]
end

function coo_values(::Type{T}) where {T<:Complex}
    return T[1 + im, 2 - 3im, 3 + 2im, 4 - im]
end

function coo_vector(::Type{T}) where {T<:Real}
    return T[2, 3, 5]
end

function coo_vector(::Type{T}) where {T<:Complex}
    return T[2 - im, 3 + 4im, 5 - 2im]
end

function coo_rhs(::Type{T}) where {T<:Real}
    return T[
        1 2;
        3 4;
        5 6
    ]
end

function coo_rhs(::Type{T}) where {T<:Complex}
    return T[
        1 + im 2 - im;
        3 + 2im 4 - 3im;
        5 - im 6 + im
    ]
end

function coo_lhs(::Type{T}) where {T<:Real}
    return T[
        1 2 3;
        4 5 6
    ]
end

function coo_lhs(::Type{T}) where {T<:Complex}
    return T[
        1 - im 2 + im 3 - 2im;
        4 + 3im 5 - im 6 + im
    ]
end

# A non-zero-preserving broadcast densifies its result: GPU back-ends return the dense
# array type `dense_AT`, whereas CPU `SparseArrays` returns its sparse type with every
# entry explicitly stored. Accept either so the suite is meaningful for both.
is_densified(x, dense_AT, ::Type{ET}) where {ET} =
    dense_AT <: GPUArrays.AnyGPUArray ? x isa dense_AT{ET} : (issparse(x) && eltype(x) === ET)

function coo_matmul(AT, dense_AT, eltypes)
    for ET in (Float16, Float32, ComplexF16, ComplexF32)
        ET in eltypes || continue
        @testset "COO repeated-index matmul($ET)" begin
            I = Int32[1, 1, 2, 3]
            J = Int32[1, 1, 2, 1]
            V = coo_values(ET)
            A = sparse(Int.(I), Int.(J), V, 3, 3)
            dA = AT(dense_AT(I), dense_AT(J), dense_AT(V), (3, 3), length(V))

            x = coo_vector(ET)
            dx = dense_AT(x)
            @test Array(dA * dx) ≈ Array(A * x)
            @test Array(transpose(dA) * dx) ≈ Array(transpose(A) * x)
            @test Array(adjoint(dA) * dx) ≈ Array(adjoint(A) * x)

            old_y = dense_AT(fill(ET(1), 3))
            mul!(old_y, dA, dx, ET(2), ET(3))
            @test Array(old_y) ≈ Array(ET(2) .* (A * x) .+ ET(3))

            B = coo_rhs(ET)
            dB = dense_AT(B)
            @test Array(dA * dB) ≈ Array(A * B)
            @test Array(transpose(dA) * dB) ≈ Array(transpose(A) * B)
            @test Array(adjoint(dA) * dB) ≈ Array(adjoint(A) * B)

            L = coo_lhs(ET)
            dL = dense_AT(L)
            @test Array(dL * dA) ≈ Array(L * A)
            @test Array(dL * transpose(dA)) ≈ Array(L * transpose(A))
            @test Array(dL * adjoint(dA)) ≈ Array(L * adjoint(A))
        end
    end
end

function vector(AT, dense_AT, eltypes)
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

function vector_construction(AT, dense_AT, eltypes)
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

        dense_x = dense_AT(collect(x))
        @test SparseVector(dense_x) == x
        if dense_x isa GPUArrays.AnyGPUArray
            typed_d_x = AT(dense_x)
            @test typed_d_x isa typeof(d_x)
            @test SparseVector(typed_d_x) == x
        end
        @test sparse(dense_x) isa AT{ET}
        @test SparseVector(sparse(dense_x)) == x
    end
end

function matrix(AT, eltypes)
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

function matrix_construction(AT, dense_AT, eltypes)
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

        dense_x = dense_AT(collect(x))
        @test SparseMatrixCSC(dense_x) == x
        if dense_x isa GPUArrays.AnyGPUArray
            typed_d_x = AT(dense_x)
            @test typed_d_x isa typeof(d_x)
            @test SparseMatrixCSC(typed_d_x) == x
        end
        @test issparse(sparse(dense_x))
        @test SparseMatrixCSC(sparse(dense_x)) == x
        if dense_x isa GPUArrays.AnyGPUArray
            @test SparseMatrixCSC(sparse(dense_x; fmt=:csr)) == x
        end
    end
end

function broadcasting_vector(AT, dense_AT, eltypes)
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
            @test is_densified(dy, dense_AT, ET)
            hy = Array(dy)
            @test Array(y) == hy

            # involving something dense
            y  = x  .+ ones(ET, m)
            dy = dx .+ dense_AT(ones(ET, m))
            @test is_densified(dy, dense_AT, ET)
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
            @test is_densified(dz, dense_AT, ET)
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
            @test is_densified(dy, dense_AT, promote_type(ET, Int))
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

function broadcasting_matrix(AT, dense_AT, eltypes)
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
            @test is_densified(dy, dense_AT, ET)
            hy = Array(dy)
            dense_y = Array(y)
            @test Array(y) == Array(dy)

            # involving something dense
            y  = x  .* ones(ET, m, n)
            dy = dx .* dense_AT(ones(ET, m, n))
            @test is_densified(dy, dense_AT, ET)
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

            # create a matrix with nnz < leading_dim
            x = spdiagm(m, m, 2=>rand(ET, m - 2))
            dx = AT(x)
            y = ET(3) * x
            dy = ET(3) * dx
            @test y == SparseMatrixCSC(dy)

            x = spdiagm(m, m, -2=>rand(ET, m - 2))
            dx = AT(x)
            y = ET(3) * x
            dy = ET(3) * dx
            @test y == SparseMatrixCSC(dy)
        end
    end
end

function mapreduce_matrix(AT, eltypes)
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
    for ET in eltypes
        # sprandn doesn't work nicely with these...
        if !(ET <: Union{Int16, Int32, Int64, Complex{Int16}, Complex{Int32}, Complex{Int64}})
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

function iszero_vector(AT, eltypes)
    for ET in eltypes
        @testset "iszero SparseVector($ET)" begin
            m = 25

            # Test non-zero sparse vector
            x = sprand(ET, m, 0.5)
            while iszero(x)
                x = sprand(ET, m, 0.5)
            end
            d_x = AT(x)
            @test iszero(d_x) == iszero(x)
            @test iszero(d_x) == false

            # Test zero sparse vector (no stored elements)
            z = spzeros(ET, m)
            d_z = AT(z)
            @test iszero(d_z) == iszero(z)
            @test iszero(d_z) == true

            # Test sparse vector with stored zeros (e.g., after operations)
            # Create a sparse vector then multiply by zero
            x_zeros = x .* zero(ET)
            d_x_zeros = d_x .* zero(ET)
            @test iszero(d_x_zeros) == iszero(x_zeros)
            @test iszero(d_x_zeros) == true
        end
    end
end

function iszero_matrix(AT, eltypes)
    for ET in eltypes
        @testset "iszero SparseMatrix($ET)" begin
            m, n = 10, 10

            # Test non-zero sparse matrix
            A = sprand(ET, m, n, 0.5)
            while iszero(A)
                A = sprand(ET, m, n, 0.5)
            end
            dA = AT(A)
            @test iszero(dA) == iszero(A)
            @test iszero(dA) == false

            # Test zero sparse matrix (no stored elements)
            ZA = spzeros(ET, m, n)
            dZA = AT(ZA)
            @test iszero(dZA) == iszero(ZA)
            @test iszero(dZA) == true

            # Test sparse matrix with stored zeros
            A_zeros = A .* zero(ET)
            dA_zeros = dA .* zero(ET)
            @test iszero(dA_zeros) == iszero(A_zeros)
            @test iszero(dA_zeros) == true
        end
    end
end
