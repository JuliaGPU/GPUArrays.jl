@testsuite "sparse" (AT, eltypes)->begin
    if AT <: AbstractSparseVector
        broadcasting_vector(AT, eltypes)
    elseif AT <: AbstractSparseMatrix
        broadcasting_matrix(AT, eltypes)
        mapreduce_matrix(AT, eltypes)
    end
end

using SparseArrays

function broadcasting_vector(AT, eltypes)
    dense_AT = GPUArrays._dense_array_type(AT)
    dense_VT = GPUArrays._dense_vector_type(AT)
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
            @test collect(SparseArrays.nonzeroinds(dy))  == collect(SparseArrays.nonzeroinds(dx))
            @test collect(SparseArrays.nonzeroinds(dy))  == SparseArrays.nonzeroinds(y)
            @test collect(SparseArrays.nonzeros(dy)) == SparseArrays.nonzeros(y)
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
    dense_AT = GPUArrays._dense_array_type(AT)
    dense_VT = GPUArrays._dense_vector_type(AT)
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
    dense_AT = GPUArrays._dense_array_type(AT)
    dense_VT = GPUArrays._dense_vector_type(AT)
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

            #y  = mapreduce(abs, +, x, dims=2)
            #dy = mapreduce(abs, +, dx, dims=2)
            #@test y ≈ Array(dy)
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
