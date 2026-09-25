using SparseArrays
using SparseArrays: nonzeroinds, nonzeros, rowvals, getcolptr
using GPUArrays: GPUSparseMatrixCSR, GPUSparseMatrixCSC, GPUSparseMatrixCOO, GPUSparseVector,
                 AbstractGPUSparseArray, check_structure

@testsuite "sparse" (AT, eltypes)->begin
    # the generic sparse formats use `AT`'s vectors as storage; plain arrays have no device
    AT <: AbstractGPUArray || return
    # nothing may fall back to scalar indexing unless a test asks for it
    GPUArrays.allowscalar(false)

    sparse_construction(AT, eltypes)
    sparse_allocation(AT, eltypes)
    sparse_transfer(AT, eltypes)
    sparse_kernels(AT, eltypes)
    sparse_display(AT, eltypes)
    sparse_indexing(AT, eltypes)
    sparse_conversions(AT, eltypes)
    sparse_dense_conversions(AT, eltypes)
    sparse_assembly(AT, eltypes)
    sparse_products(AT, eltypes)
    broadcasting_vector(AT, eltypes)
    broadcasting_matrix(AT, eltypes)
    broadcasting_mixed(AT, eltypes)
    mapreduce_matrix(AT, eltypes)
    sparse_linalg(AT, eltypes)
    iszero_sparse(AT, eltypes)
    sparse_bool(AT)
end

const sparse_matrix_formats = (GPUSparseMatrixCSR, GPUSparseMatrixCSC, GPUSparseMatrixCOO)

# A GPU sparse matrix of format `S` with the entries of `A`, built directly from buffers
# computed on the host, so that the tests do not depend on the conversions they test.
function gpu_sparse(AT, S::Type, A::SparseMatrixCSC)
    if S <: GPUSparseMatrixCSC
        GPUSparseMatrixCSC(AT(getcolptr(A)), AT(rowvals(A)), AT(nonzeros(A)), size(A))
    else
        # the CSC buffers of the transpose are the CSR buffers of `A`
        At = copy(transpose(A))
        if S <: GPUSparseMatrixCSR
            GPUSparseMatrixCSR(AT(getcolptr(At)), AT(rowvals(At)), AT(nonzeros(At)), size(A))
        else
            rows = similar(rowvals(At))
            for i in 1:size(A, 1), k in nzrange(At, i)
                rows[k] = i
            end
            GPUSparseMatrixCOO(AT(rows), AT(rowvals(At)), AT(nonzeros(At)), size(A))
        end
    end
end
gpu_sparse(AT, x::SparseVector) = GPUSparseVector(AT(nonzeroinds(x)), AT(nonzeros(x)), length(x))

# `sprand`/`sprandn` can produce *stored* zeros: with a coarse element type like Float16,
# `rand` returns an exact zero about once every 2048 draws, so roughly one in 60 vectors
# here contains one. That's a valid sparse array, but CPU and GPU implementations
# legitimately disagree on what to do with it -- `x .* 1` drops stored zeros on the CPU
# while our kernels preserve the structure -- so the structural comparisons below fail at
# random. Keep the inputs canonical instead, and test stored zeros explicitly.
sprand_nozeros(args...) = dropzeros!(sprand(args...))
sprandn_nozeros(args...) = dropzeros!(sprandn(args...))

# a random matrix with an empty row and column, a dense row, and a stored zero
function sprand_awkward(::Type{T}, m, n; Ti=Int) where {T}
    A = Matrix(sprand_nozeros(T, m, n, 0.3))
    A[1, :] .= rand(T, n) .+ one(T)
    A[:, 2] .= zero(T)
    A[3, :] .= zero(T)
    S = SparseMatrixCSC{T,Ti}(sparse(A))
    nonzeros(S)[end] = zero(T)
    return S
end

# the same entries, and the same explicitly stored ones
same_sparse(A::SparseMatrixCSC, B::SparseMatrixCSC) =
    size(A) == size(B) && getcolptr(A) == getcolptr(B) && rowvals(A) == rowvals(B) &&
    nonzeros(A) == nonzeros(B)
same_sparse(x::SparseVector, y::SparseVector) =
    length(x) == length(y) && nonzeroinds(x) == nonzeroinds(y) && nonzeros(x) == nonzeros(y)
same_sparse(A::AbstractGPUSparseArray, B) = same_sparse(host(A), B)
host(A::AbstractGPUSparseArray) = ndims(A) == 1 ? SparseVector(A) : SparseMatrixCSC(A)

# the result of a reduction on the host, whether it is a scalar or an array
reduced(x::Number) = x
reduced(x::AbstractArray) = Array(x)

function sparse_construction(AT, eltypes)
    @testset "construction and properties" begin
        @testset "$S{$ET, $Ti}" for S in sparse_matrix_formats, ET in eltypes, Ti in (Int32, Int64)
            A = sprand_awkward(ET, 10, 8; Ti)
            dA = gpu_sparse(AT, S, A)
            @test dA isa S{ET,Ti}
            @test dA isa AbstractGPUSparseArray{ET,Ti,2}
            @test size(dA) == (10, 8)
            @test length(dA) == 80
            @test ndims(dA) == 2
            @test nnz(dA) == nnz(A)
            @test issparse(dA)
            @test SparseArrays.indtype(dA) == Ti
            @test nonzeros(dA) isa AT{ET}
            check_structure(dA)
            @test same_sparse(SparseMatrixCSC(dA), A)
            @test SparseMatrixCSC(dA) isa SparseMatrixCSC{ET,Ti}
            @test Array(dA) == Array(A)
            @test collect(dA) == Array(A)
            @test get_backend(dA) == get_backend(AT(ET[]))
            if S <: GPUSparseMatrixCSC
                @test Array(getcolptr(dA)) == getcolptr(A)
                @test Array(rowvals(dA)) == rowvals(A)
            end
        end

        @testset "vector $ET, $Ti" for ET in eltypes, Ti in (Int32, Int64)
            x = SparseVector{ET,Ti}(sprand_nozeros(ET, 20, 0.3))
            dx = gpu_sparse(AT, x)
            @test dx isa GPUSparseVector{ET,Ti}
            @test size(dx) == (20,)
            @test nnz(dx) == nnz(x)
            @test Array(nonzeroinds(dx)) == nonzeroinds(x)
            @test Array(rowvals(dx)) == nonzeroinds(x)
            @test Array(nonzeros(dx)) == nonzeros(x)
            check_structure(dx)
            @test same_sparse(SparseVector(dx), x)
            @test Array(dx) == Array(x)
        end

        @testset "shapes" begin
            ET = first(eltypes)
            for (m, n) in ((0, 0), (0, 4), (4, 0), (5, 3)), S in sparse_matrix_formats
                A = spzeros(ET, m, n)
                dA = gpu_sparse(AT, S, A)
                check_structure(dA)
                @test same_sparse(SparseMatrixCSC(dA), A)
                @test Array(dA) == zeros(ET, m, n)
            end
            for n in (0, 5)
                x = spzeros(ET, n)
                @test same_sparse(SparseVector(gpu_sparse(AT, x)), x)
            end
        end

        @testset "checks" begin
            ET = first(eltypes)
            ptr = AT(Int32[1, 2, 3])
            ind = AT(Int32[1, 2])
            val = AT(ET[1, 2])
            @test GPUSparseMatrixCSR(ptr, ind, val, (2, 2)) isa GPUSparseMatrixCSR
            # pointer length
            @test_throws ArgumentError GPUSparseMatrixCSR(ptr, ind, val, (3, 2))
            @test_throws ArgumentError GPUSparseMatrixCSC(ptr, ind, val, (2, 3))
            # buffer lengths
            @test_throws ArgumentError GPUSparseMatrixCSR(ptr, ind, AT(ET[1]), (2, 2))
            @test_throws ArgumentError GPUSparseMatrixCOO(ind, AT(Int32[1]), val, (2, 2))
            @test_throws ArgumentError GPUSparseVector(ind, AT(ET[1]), 2)
            # dimensions
            @test_throws ArgumentError GPUSparseMatrixCOO(ind, ind, val, (-1, 2))
            @test_throws ArgumentError GPUSparseVector(AT(Int8[1]), AT(ET[1]), 200)
            # the structure is only checked on request
            @test check_structure(GPUSparseMatrixCSR(AT(Int32[1, 3, 3]), AT(Int32[1, 2]), val, (2, 2))) isa GPUSparseMatrixCSR
            @test_throws ArgumentError check_structure(GPUSparseMatrixCSR(AT(Int32[1, 3, 3]), AT(Int32[2, 1]), val, (2, 2)))
            @test_throws ArgumentError check_structure(GPUSparseMatrixCSC(AT(Int32[1, 2, 3]), AT(Int32[1, 3]), val, (2, 2)))
            @test_throws ArgumentError check_structure(GPUSparseMatrixCOO(AT(Int32[2, 1]), AT(Int32[1, 1]), val, (2, 2)))
            @test_throws ArgumentError check_structure(GPUSparseMatrixCOO(AT(Int32[1, 1]), AT(Int32[1, 1]), val, (2, 2)))
            @test_throws ArgumentError check_structure(GPUSparseVector(AT(Int32[2, 2]), val, 3))
        end
    end
end

function sparse_allocation(AT, eltypes)
    @testset "allocation" begin
        @testset "$S" for S in sparse_matrix_formats
            ET = first(eltypes)
            A = sprand_awkward(ET, 6, 7; Ti=Int32)
            dA = gpu_sparse(AT, S, A)

            # structure-preserving
            B = similar(dA)
            @test B isa S{ET,Int32}
            @test nnz(B) == nnz(dA)
            copyto!(nonzeros(B), nonzeros(dA))
            @test same_sparse(B, A)
            B = similar(dA, Float32)
            @test B isa S{Float32,Int32}
            @test nnz(B) == nnz(dA)
            B = similar(dA, Float32, Int64)
            @test B isa S{Float32,Int64}
            @test nnz(B) == nnz(dA)

            # without stored entries
            for dims in ((6, 7), (3, 4))
                B = similar(dA, Float32, dims)
                @test B isa S{Float32,Int32}
                @test size(B) == dims
                @test nnz(B) == 0
                check_structure(B)
            end
            B = similar(dA, (3, 4))
            @test B isa S{ET,Int32}
            B = similar(dA, Float32, (5,))
            @test B isa GPUSparseVector{Float32,Int32}
            @test length(B) == 5 && nnz(B) == 0
            B = similar(dA, Float32, (2, 3, 4))
            @test B isa AT{Float32,3}
            B = zero(dA)
            @test B isa S{ET,Int32}
            @test size(B) == size(dA) && nnz(B) == 0

            # copies share nothing with the original
            B = copy(dA)
            @test B isa S{ET,Int32}
            @test same_sparse(B, A)
            fill!(nonzeros(B), zero(ET))
            @test same_sparse(dA, A)
            B = S(dA)
            fill!(nonzeros(B), zero(ET))
            @test same_sparse(dA, A)

            # copyto! replaces the structure
            A2 = sprand_awkward(ET, 6, 7; Ti=Int32)
            dA2 = gpu_sparse(AT, S, A2)
            B = copy(dA)
            @test copyto!(B, dA2) === B
            @test same_sparse(B, A2)
            @test_throws DimensionMismatch copyto!(B, gpu_sparse(AT, S, spzeros(ET, 2, 2)))

            # densify
            D = AT(zeros(ET, 6, 7))
            @test copyto!(D, dA) === D
            @test Array(D) == Array(A)
            D = AT(ones(ET, 50))
            copyto!(D, dA)
            @test Array(D) == [vec(Array(A)); ones(ET, 8)]
        end

        @testset "vector" begin
            ET = first(eltypes)
            x = SparseVector{ET,Int32}(sprand_nozeros(ET, 20, 0.3))
            dx = gpu_sparse(AT, x)
            @test similar(dx) isa GPUSparseVector{ET,Int32}
            @test similar(dx, Float32) isa GPUSparseVector{Float32,Int32}
            @test similar(dx, Float32, Int64) isa GPUSparseVector{Float32,Int64}
            @test nnz(similar(dx, Float32, (20,))) == 0
            B = similar(dx, Float32, (4, 5))
            @test B isa GPUSparseMatrixCSC{Float32,Int32}
            @test nnz(B) == 0
            y = copy(dx)
            fill!(nonzeros(y), zero(ET))
            @test same_sparse(dx, x)
            y = zero(dx)
            @test nnz(y) == 0
            y = gpu_sparse(AT, SparseVector{ET,Int32}(sprand_nozeros(ET, 20, 0.5)))
            copyto!(y, dx)
            @test same_sparse(y, x)
            D = AT(zeros(ET, 20))
            copyto!(D, dx)
            @test Array(D) == Array(x)
        end
    end
end

function sparse_transfer(AT, eltypes)
    @testset "adapt" begin
        ET = first(eltypes)
        A = sprand_awkward(ET, 6, 7; Ti=Int32)
        x = SparseVector{ET,Int32}(sprand_nozeros(ET, 20, 0.3))

        # structural: keep format, element and index types
        dA = adapt(AT, A)
        @test dA isa GPUSparseMatrixCSC{ET,Int32}
        @test nonzeros(dA) isa AT
        @test same_sparse(dA, A)
        dx = adapt(AT, x)
        @test dx isa GPUSparseVector{ET,Int32}
        @test same_sparse(dx, x)
        @test adapt(AT, dA) === dA

        # to the host
        B = adapt(Array, dA)
        @test B isa SparseMatrixCSC{ET,Int32}
        @test same_sparse(B, A)
        @test adapt(Array, dx) isa SparseVector{ET,Int32}
        for S in (GPUSparseMatrixCSR, GPUSparseMatrixCOO)
            B = adapt(Array, gpu_sparse(AT, S, A))
            @test B isa S{ET,Int32,Vector{Int32},Vector{ET}}
            @test same_sparse(SparseMatrixCSC(B), A)
        end

        # a target with an element type only converts the values
        T = ET <: Complex ? ComplexF32 : Float32
        if T in eltypes
            dB = adapt(AT{T}, A)
            @test dB isa GPUSparseMatrixCSC{T,Int32}
            @test Array(dB) ≈ Array(A)
            @test adapt(Array{T}, dA) isa SparseMatrixCSC{T,Int32}
        end

        # to an explicit sparse type
        dB = adapt(GPUSparseMatrixCSR, dA)
        @test dB isa GPUSparseMatrixCSR{ET,Int32}
        @test nonzeros(dB) isa AT
        @test same_sparse(dB, A)
        @test_throws ArgumentError adapt(GPUSparseMatrixCSR, A)
    end
end

@kernel function sparse_rowsum_kernel(out, A)
    row = @index(Global, Linear)
    acc = zero(eltype(out))
    for k in A.rowPtr[row]:(A.rowPtr[row+1] - 1)
        acc += A.nzVal[k]
    end
    out[row] = acc
end

@kernel function sparse_colsum_kernel(out, A)
    col = @index(Global, Linear)
    acc = zero(eltype(out))
    for k in nzrange(A, col)
        acc += nonzeros(A)[k]
    end
    out[col] = acc
end

function sparse_kernels(AT, eltypes)
    @testset "user kernels" begin
        ET = first(filter(isrealfloattype, eltypes))
        A = sprand_awkward(ET, 8, 6)
        dA = gpu_sparse(AT, GPUSparseMatrixCSR, A)
        out = AT(zeros(ET, 8))
        sparse_rowsum_kernel(get_backend(dA))(out, dA; ndrange=8)
        @test Array(out) ≈ vec(sum(A; dims=2))
        dA = gpu_sparse(AT, GPUSparseMatrixCSC, A)
        out = AT(zeros(ET, 6))
        sparse_colsum_kernel(get_backend(dA))(out, dA; ndrange=6)
        @test Array(out) ≈ vec(sum(A; dims=1))
    end
end

function sparse_display(AT, eltypes)
    @testset "display" begin
        ET = first(eltypes)
        A = sprand_awkward(ET, 6, 7; Ti=Int32)
        for S in sparse_matrix_formats
            dA = gpu_sparse(AT, S, A)
            str = summary(dA)
            @test str == "6×7 $(nameof(S)){$ET, Int32} with $(nnz(A)) stored entries in $(nameof(AT))"
            # the entries are rendered as for the host matrix
            host_str = sprint(show, MIME"text/plain"(), A)
            dev_str = sprint(show, MIME"text/plain"(), dA)
            @test split(dev_str, '\n')[2:end] == split(host_str, '\n')[2:end]
            @test sprint(show, dA) == sprint(show, A)
        end
        x = SparseVector{ET,Int32}(sprand_nozeros(ET, 20, 0.3))
        dx = gpu_sparse(AT, x)
        @test startswith(summary(dx), "20-element GPUSparseVector{$ET, Int32} with $(nnz(x)) stored")
        host_str = sprint(show, MIME"text/plain"(), x)
        dev_str = sprint(show, MIME"text/plain"(), dx)
        @test split(dev_str, '\n')[2:end] == split(host_str, '\n')[2:end]
        @test sprint(show, dx) == sprint(show, x)
        @test sprint(show, MIME"text/plain"(), gpu_sparse(AT, spzeros(ET, 3))) ==
              summary(gpu_sparse(AT, spzeros(ET, 3)))
    end
end

function sparse_indexing(AT, eltypes)
    @testset "indexing" begin
        ET = first(eltypes)
        A = sprand_awkward(ET, 6, 7)
        @testset "$S" for S in sparse_matrix_formats
            dA = gpu_sparse(AT, S, A)
            @test_throws ErrorException dA[1, 1]
            @allowscalar begin
                @test all(dA[i, j] == A[i, j] for i in 1:6, j in 1:7)
                @test dA[(2, 3)] == A[2, 3]
                @test dA[end] == A[end]
                @test dA[5] == A[5]
            end
            @test_throws BoundsError dA[0, 1]
            @test_throws BoundsError dA[7, 1]
            @test_throws BoundsError dA[1, 8]
            @test same_sparse(dA[:, :], A)
            @test_throws ArgumentError dA[1, 1] = one(ET)
        end
        x = sprand_nozeros(ET, 20, 0.3)
        dx = gpu_sparse(AT, x)
        @allowscalar begin
            @test all(dx[i] == x[i] for i in 1:20)
            @test dx[end] == x[end]
        end
        @test_throws BoundsError dx[21]
        @test same_sparse(dx[:], x)
        @test_throws ArgumentError dx[1] = one(ET)
    end
end

function sparse_conversions(AT, eltypes)
    @testset "conversions" begin
        @testset "$ET" for ET in eltypes
            A = sprand_awkward(ET, 9, 7; Ti=Int32)
            dcsr = gpu_sparse(AT, GPUSparseMatrixCSR, A)
            dcsc = gpu_sparse(AT, GPUSparseMatrixCSC, A)

            # between CSR and CSC
            B = GPUSparseMatrixCSC(dcsr)
            @test B isa GPUSparseMatrixCSC{ET,Int32}
            check_structure(B)
            @test same_sparse(B, A)
            B = GPUSparseMatrixCSR(dcsc)
            @test B isa GPUSparseMatrixCSR{ET,Int32}
            check_structure(B)
            @test same_sparse(B, A)
            @test convert(GPUSparseMatrixCSR, dcsr) === dcsr
            @test same_sparse(convert(GPUSparseMatrixCSR, dcsc), A)

            # element and index types
            B = GPUSparseMatrixCSR{ET,Int64}(dcsc)
            @test B isa GPUSparseMatrixCSR{ET,Int64}
            @test same_sparse(B, SparseMatrixCSC{ET,Int64}(A))
            B = GPUSparseMatrixCSC{ET,Int64}(dcsc)
            @test B isa GPUSparseMatrixCSC{ET,Int64}
            @test same_sparse(B, A)
            if ET <: Real && Float32 in eltypes
                B = GPUSparseMatrixCSC{Float32}(dcsr)
                @test B isa GPUSparseMatrixCSC{Float32,Int32}
                @test Array(B) ≈ Array(A)
            end
            @test_throws ArgumentError GPUSparseMatrixCSR{ET,Int8}(gpu_sparse(AT, GPUSparseMatrixCSR, spzeros(ET, 200, 2)))

            # materializing transposes into the opposite format
            B = GPUSparseMatrixCSC(transpose(dcsr))
            @test B isa GPUSparseMatrixCSC{ET,Int32}
            @test same_sparse(B, copy(transpose(A)))
            B = GPUSparseMatrixCSR(adjoint(dcsc))
            @test B isa GPUSparseMatrixCSR{ET,Int32}
            @test same_sparse(B, copy(adjoint(A)))

            # shapes
            for (m, n) in ((0, 0), (0, 4), (4, 0))
                Z = spzeros(ET, Int32, m, n)
                @test same_sparse(GPUSparseMatrixCSC(gpu_sparse(AT, GPUSparseMatrixCSR, Z)), Z)
                @test same_sparse(GPUSparseMatrixCSR(gpu_sparse(AT, GPUSparseMatrixCSC, Z)), Z)
            end

            # the host arrays have no storage to convert to
            @test_throws ArgumentError GPUSparseMatrixCSR(A)
        end
    end
end

function broadcasting_vector(AT, eltypes)
    @testset "SparseVector broadcasting" begin
        @testset "$ET" for ET in eltypes
            m  = 64
            p  = 0.5
            x  = sprand_nozeros(ET, m, p)
            dx = gpu_sparse(AT, x)

            # zero-preserving
            y  = x  .* ET(1)
            dy = dx .* ET(1)
            @test dy isa GPUSparseVector{ET}
            @test collect(SparseArrays.nonzeroinds(dy)) == collect(SparseArrays.nonzeroinds(dx))
            @test collect(SparseArrays.nonzeroinds(dy)) == SparseArrays.nonzeroinds(y)
            @test collect(SparseArrays.nonzeros(dy))    == SparseArrays.nonzeros(y)
            @test y == SparseVector(dy)

            # not zero-preserving
            y  = x  .+ ET(1)
            dy = dx .+ ET(1)
            @test dy isa AT{ET}
            hy = Array(dy)
            @test Array(y) == hy

            # involving something dense
            y  = x  .+ ones(ET, m)
            dy = dx .+ AT(ones(ET, m))
            @test dy isa AT{ET}
            @test Array(y) == Array(dy)

            # sparse to sparse
            dx = gpu_sparse(AT, x)
            y  = sprand_nozeros(ET, m, p)
            dy = gpu_sparse(AT, y)
            z  = x  .* y
            dz = dx .* dy
            @test dz isa GPUSparseVector{ET}
            @test z ≈ SparseVector(dz)

            # multiple inputs
            y  = sprand_nozeros(ET, m, p)
            w  = sprand_nozeros(ET, m, p)
            dy = gpu_sparse(AT, y)
            dx = gpu_sparse(AT, x)
            dw = gpu_sparse(AT, w)
            z  = @. x  * y  * w
            dz = @. dx * dy * dw
            @test dz isa GPUSparseVector{ET}
            @test z ≈ SparseVector(dz)

            y = sprand_nozeros(ET, m, p)
            w = sprand_nozeros(ET, m, p)
            dense_arr   = rand(ET, m)
            d_dense_arr = AT(dense_arr)
            dy = gpu_sparse(AT, y)
            dw = gpu_sparse(AT, w)
            z  = @. x  * y  * w  * dense_arr
            dz = @. dx * dy * dw * d_dense_arr
            @test dz isa AT{ET}
            @test Array(z) ≈ Array(dz)

            y  = sprand_nozeros(ET, m, p)
            dy = gpu_sparse(AT, y)
            dx = gpu_sparse(AT, x)
            z  = x  .* y  .* ET(2)
            dz = dx .* dy .* ET(2)
            @test dz isa GPUSparseVector{ET}
            @test z ≈ SparseVector(dz)

            # type-mismatching
            ## non-zero-preserving
            dx = gpu_sparse(AT, x)
            dy = dx .+ 1
            y  = x .+ 1
            @test dy isa AT{promote_type(ET, Int)}
            @test Array(y) == Array(dy)
            ## zero-preserving
            dy = dx .* 1
            y  = x  .* 1
            @test dy isa GPUSparseVector{promote_type(ET, Int)}
            @test collect(SparseArrays.nonzeroinds(dy))  == collect(SparseArrays.nonzeroinds(dx))
            @test collect(SparseArrays.nonzeroinds(dy))  == SparseArrays.nonzeroinds(y)
            @test collect(SparseArrays.nonzeros(dy)) == SparseArrays.nonzeros(y)
            @test y == SparseVector(dy)

            # without stored entries
            x  = spzeros(ET, m)
            dx = gpu_sparse(AT, x)
            y  = sprand_nozeros(ET, m, p)
            dy = gpu_sparse(AT, y)
            @test SparseVector(dx .* ET(2)) == x .* ET(2)
            @test SparseVector(dx .* dy) == x .* y
            @test Array(dx .+ dy) == Array(x .+ y)
            @test length(gpu_sparse(AT, spzeros(ET, 0)) .* gpu_sparse(AT, spzeros(ET, 0))) == 0
        end
    end

    # several sparse arguments, long enough that host-side processing of the combined
    # structure (e.g. sorting it through scalar indexing) would be noticeable
    @testset "long vectors" begin
        ET = first(filter(isrealfloattype, eltypes))
        m  = 10^4
        x  = sprand_nozeros(ET, m, 0.1)
        y  = sprand_nozeros(ET, m, 0.1)
        dx = gpu_sparse(AT, x)
        dy = gpu_sparse(AT, y)
        dz = dx .* dy
        @test dz isa GPUSparseVector{ET}
        @test SparseVector(dz) == x .* y
        @test Array(dx .+ dy) == Array(x .+ y)
    end
end

function broadcasting_matrix(AT, eltypes)
    @testset "SparseMatrix broadcasting" begin
        @testset "$S{$ET}" for S in (GPUSparseMatrixCSR, GPUSparseMatrixCSC), ET in eltypes
            m, n = 5, 6
            p   = 0.5
            x   = sprand_nozeros(ET, m, n, p)
            dx  = gpu_sparse(AT, S, x)
            # zero-preserving
            y  = x  .* ET(1)
            dy = dx .* ET(1)
            @test dy isa S{ET}
            @test y == SparseMatrixCSC(dy)
            # the output does not share buffers with the input
            fill!(nonzeros(dy), zero(ET))
            @test same_sparse(dx, x)

            # not zero-preserving
            y  = x  .+ ET(1)
            dy = dx .+ ET(1)
            @test dy isa AT{ET}
            @test Array(y) == Array(dy)

            # involving something dense
            y  = x  .* ones(ET, m, n)
            dy = dx .* AT(ones(ET, m, n))
            @test dy isa AT{ET}
            @test Array(y) == Array(dy)

            # multiple inputs
            y  = sprand_nozeros(ET, m, n, p)
            dy = gpu_sparse(AT, S, y)
            z  = x  .* y  .* ET(2)
            dz = dx .* dy .* ET(2)
            @test dz isa S{ET}
            @test z ≈ SparseMatrixCSC(dz)

            # multiple inputs
            w  = sprand_nozeros(ET, m, n, p)
            dw = gpu_sparse(AT, S, w)
            z  = x  .* y  .* w
            dz = dx .* dy .* dw
            @test dz isa S{ET}
            @test z ≈ SparseMatrixCSC(dz)

            # create a matrix with nnz < leading_dim
            x = spdiagm(m, m, 2=>rand(ET, m - 2))
            dx = gpu_sparse(AT, S, x)
            y = ET(3) * x
            dy = ET(3) * dx
            @test y == SparseMatrixCSC(dy)

            x = spdiagm(m, m, -2=>rand(ET, m - 2))
            dx = gpu_sparse(AT, S, x)
            y = ET(3) * x
            dy = ET(3) * dx
            @test y == SparseMatrixCSC(dy)

            # without rows or columns
            for dims in ((0, 4), (4, 0))
                x = spzeros(ET, dims...)
                dx = gpu_sparse(AT, S, x)
                @test same_sparse(SparseMatrixCSC(dx .* dx), x .* x)
                @test same_sparse(SparseMatrixCSC(dx .* ET(2)), x .* ET(2))
                @test size(dx .+ dx) == dims
            end

            # stored zeros are kept
            x = sprand_awkward(ET, 6, 7)
            dx = gpu_sparse(AT, S, x)
            @test nnz(dx .* ET(2)) == nnz(x)
        end
    end
end

function broadcasting_mixed(AT, eltypes)
    @testset "mixed-format broadcasting" begin
        @testset "$ET" for ET in eltypes
            A = sprand_awkward(ET, 6, 7)
            B = sprand_nozeros(ET, 6, 7, 0.4)
            # the result has the format of the first sparse argument
            for S in sparse_matrix_formats, S′ in sparse_matrix_formats
                dA = gpu_sparse(AT, S, A)
                dB = gpu_sparse(AT, S′, B)
                dC = dA .* dB
                @test dC isa S{ET}
                check_structure(dC)
                @test SparseMatrixCSC(dC) ≈ A .* B
                dC = dA .+ dB
                @test dC isa S{ET}
                @test SparseMatrixCSC(dC) ≈ A .+ B
                @test SparseMatrixCSC(dA + dB) ≈ A + B
                @test SparseMatrixCSC(dA - dB) ≈ A - B
                @test Array(dA .+ dB .+ ET(1)) ≈ Array(A .+ B .+ ET(1))
            end
            for S in sparse_matrix_formats
                dA = gpu_sparse(AT, S, A)
                @test SparseMatrixCSC(ET(2) * dA) == ET(2) * A
                @test SparseMatrixCSC(-dA) == -A
                three = ET(3)
                @test SparseMatrixCSC(map(v -> v * three, dA)) == map(v -> v * three, A)
                @test Array(map(+, AT(Matrix(B)), dA)) ≈ Matrix(B) + A
                @test SparseMatrixCSC(conj(dA)) == conj(A)
                @test Array(dA .* AT(Matrix(B))) ≈ Array(A .* Matrix(B))
            end
            @test_throws ErrorException gpu_sparse(AT, GPUSparseMatrixCSR, A) .* gpu_sparse(AT, sprand_nozeros(ET, 6, 0.5))

            # SparseArrays' own methods for sparse vectors are replaced
            x = sprand_nozeros(ET, 30, 0.3)
            y = sprand_nozeros(ET, 30, 0.3)
            dx, dy = gpu_sparse(AT, x), gpu_sparse(AT, y)
            @test SparseVector(dx + dy) ≈ x + y
            @test SparseVector(dx - dy) ≈ x - y
            for f in (+, -, *)
                @test SparseVector(map(f, dx, dy)) ≈ map(f, x, y)
                @test SparseVector(broadcast(f, dx, dy)) ≈ broadcast(f, x, y)
            end
            if ET <: Real
                @test SparseVector(map(max, dx, dy)) == map(max, x, y)
            end
            @test_throws DimensionMismatch dx + gpu_sparse(AT, sprand_nozeros(ET, 31, 0.3))
        end
    end
end

function mapreduce_matrix(AT, eltypes)
    @testset "SparseMatrix mapreduce" begin
        @testset "$S{$ET}" for S in sparse_matrix_formats, ET in eltypes
            m,n = 5,6
            p = 0.5
            x = sprand_nozeros(ET, m, n, p)
            dx = gpu_sparse(AT, S, x)

            # dim=:
            y  = sum(x)
            dy = sum(dx)
            @test y ≈ dy

            # `abs` may give an element type the back-end does not support (Complex{Int16})
            if Base.promote_op(abs, ET) in eltypes
                y  = mapreduce(abs, +, x)
                dy = mapreduce(abs, +, dx)
                @test y ≈ dy
                y  = mapreduce(abs, +, x, dims=1)
                dy = mapreduce(abs, +, dx, dims=1)
                @test y ≈ Array(dy)
                y  = mapreduce(abs, +, x, dims=2)
                dy = mapreduce(abs, +, dx, dims=2)
                @test y ≈ Array(dy)
                dy = mapreduce(abs, +, dx; init=zero(Base.promote_op(abs, ET)))
                y  = mapreduce(abs, +, x; init=zero(Base.promote_op(abs, ET)))
                @test y ≈ dy

                # test with a matrix with fully empty rows
                x = zeros(ET, m, n)
                x[2, :] .= -one(ET)
                x[2, end] = -ET(16)
                dx = gpu_sparse(AT, S, sparse(x))
                y  = mapreduce(abs, max, x)
                dy = mapreduce(abs, max, dx)
                @test y ≈ dy
            end

            # dim=1
            y  = sum(x, dims=1)
            dy = sum(dx, dims=1)
            @test y ≈ Array(dy)

            # dim=2
            y = sum(x, dims=2)
            dy = sum(dx, dims=2)
            @test y ≈ Array(dy)

            # result shapes follow Base
            @test size(sum(dx; dims=1)) == (1, n)
            @test size(sum(dx; dims=2)) == (m, 1)

            # functions that don't preserve zeros, with operators other than `+`
            o = one(ET)
            x  = sprand_nozeros(ET, m, n, p)
            dx = gpu_sparse(AT, S, x)
            for dims in (:, 1, 2)
                @test mapreduce(v -> v + o, *, x; dims) ≈ reduced(mapreduce(v -> v + o, *, dx; dims))
                if ET <: Real
                    @test mapreduce(v -> v + o, max, x; dims) ≈ reduced(mapreduce(v -> v + o, max, dx; dims))
                    @test mapreduce(v -> v - o, min, x; dims) ≈ reduced(mapreduce(v -> v - o, min, dx; dims))
                end
                @test mapreduce(v -> v + o, +, x; dims, init=zero(ET)) ≈ reduced(mapreduce(v -> v + o, +, dx; dims, init=zero(ET)))
                @test mapreduce(v -> v + o, *, x; dims, init=o) ≈ reduced(mapreduce(v -> v + o, *, dx; dims, init=o))
            end
            # an operator without a known neutral element, relying on `init`
            @test mapreduce(identity, (a, b) -> a + b, dx; init=zero(ET)) ≈ mapreduce(identity, (a, b) -> a + b, x; init=zero(ET))
            y = sparse([1, 2], [1, 3], ET[1, 2], 3, 3)
            dy = gpu_sparse(AT, S, y)
            @test mapreduce(v -> v + o, +, dy) == mapreduce(v -> v + o, +, y)
            @test mapreduce(v -> v + o, *, dy) == mapreduce(v -> v + o, *, y)
            if ET <: Real
                @test mapreduce(v -> v + o, max, dy) == mapreduce(v -> v + o, max, y)
            end

            # the stored values of a row or column don't start from zero
            if ET <: Real && !(ET <: Unsigned)
                x  = sparse(-rand(ET(1):ET(10), m, n))
                dx = gpu_sparse(AT, S, x)
                @test nnz(dx) == m * n
                for dims in (:, 1, 2)
                    @test maximum(x; dims) == reduced(maximum(dx; dims))
                    @test minimum(-x; dims) == reduced(minimum(gpu_sparse(AT, S, -x); dims))
                end
                # the same with some implicit zeros
                x[1, :] .= 0
                dropzeros!(x)
                dx = gpu_sparse(AT, S, x)
                for dims in (:, 1, 2)
                    @test maximum(x; dims) == reduced(maximum(dx; dims))
                end
            end

            # empty dimensions
            for (k, l) in ((0, n), (m, 0), (0, 0))
                x  = spzeros(ET, k, l)
                dx = gpu_sparse(AT, S, x)
                @test sum(dx) == sum(x)
                @test reduced(sum(dx; dims=1)) == sum(x; dims=1)
                @test reduced(sum(dx; dims=2)) == sum(x; dims=2)
                if Base.promote_op(abs, ET) in eltypes
                    T = Base.promote_op(abs, ET)
                    @test reduced(mapreduce(abs, +, dx; dims=1, init=zero(T))) == mapreduce(abs, +, x; dims=1, init=zero(T))
                end
            end
        end

        @testset "vector $ET" for ET in eltypes
            o = one(ET)
            x = sprand_nozeros(ET, 20, 0.3)
            dx = gpu_sparse(AT, x)
            @test sum(dx) ≈ sum(x)
            @test mapreduce(v -> v + o, *, dx) ≈ mapreduce(v -> v + o, *, x)
            @test sum(gpu_sparse(AT, spzeros(ET, 0))) == zero(ET)
        end
    end
end

function sparse_linalg(AT, eltypes)
    @testset "linear algebra" begin
        # sprandn only works on real or complex float types
        @testset "$S{$ET}" for S in sparse_matrix_formats, ET in filter(isfloattype, eltypes)
            m = 10
            A  = sprandn_nozeros(ET, m, m, 0.2)
            dA = gpu_sparse(AT, S, A)
            @test opnorm(A, Inf) ≈ opnorm(dA, Inf)
            @test opnorm(A, 1)   ≈ opnorm(dA, 1)
            @test_throws ArgumentError opnorm(dA, 2)
            for p in (1, 2, 3, Inf, -Inf, 0)
                @test norm(A, p) ≈ norm(dA, p)
            end
            @test norm(gpu_sparse(AT, S, spzeros(ET, 3, 3)), Inf) == 0
        end
    end
end

function iszero_sparse(AT, eltypes)
    @testset "iszero" begin
        @testset "$ET" for ET in eltypes
            m = 10
            x = sprand_nozeros(ET, m, 0.5)
            while iszero(x)
                x = sprand_nozeros(ET, m, 0.5)
            end
            @test !iszero(gpu_sparse(AT, x))
            @test iszero(gpu_sparse(AT, spzeros(ET, m)))
            @test iszero(gpu_sparse(AT, x) .* zero(ET))

            A = sparse(Matrix(x * transpose(x)))
            for S in sparse_matrix_formats
                @test !iszero(gpu_sparse(AT, S, A))
                @test iszero(gpu_sparse(AT, S, spzeros(ET, m, m)))
                # stored zeros
                @test iszero(gpu_sparse(AT, S, A .* zero(ET)))
            end
        end
    end
end

function sparse_bool(AT)
    @testset "Bool" begin
        A = sprand(Bool, 8, 9, 0.4)
        B = sprand(Bool, 8, 9, 0.4)
        for S in (GPUSparseMatrixCSR, GPUSparseMatrixCSC)
            dA = gpu_sparse(AT, S, A)
            dB = gpu_sparse(AT, S, B)
            @test same_sparse(dA, A)
            @test Array(dA .& dB) == Array(A .& B)
            @test Array(dA .| dB) == Array(A .| B)
            @test sum(dA) == sum(A)
            @test reduced(sum(dA; dims=1)) == sum(A; dims=1)
            @test reduced(mapreduce(identity, |, dA; dims=2)) == mapreduce(identity, |, A; dims=2)
            @test same_sparse(GPUSparseMatrixCSC(gpu_sparse(AT, GPUSparseMatrixCSR, A)), A)
        end
    end
end

function sparse_dense_conversions(AT, eltypes)
    @testset "format and dense conversions" begin
        @testset "$ET" for ET in eltypes
            A = sprand_awkward(ET, 9, 7; Ti=Int32)
            # every pair of formats, exactly, including the stored zero
            for S in sparse_matrix_formats, S′ in sparse_matrix_formats
                dA = gpu_sparse(AT, S, A)
                B = S′(dA)
                @test B isa S′{ET,Int32}
                check_structure(B)
                @test same_sparse(B, A)
                # no hidden sharing with the source
                fill!(nonzeros(B), zero(ET))
                @test same_sparse(dA, A)
            end

            # transposes, into the same and the other formats
            for S in sparse_matrix_formats
                dA = gpu_sparse(AT, S, A)
                B = copy(transpose(dA))
                @test B isa S{ET,Int32}
                check_structure(B)
                @test same_sparse(B, copy(transpose(A)))
                B = copy(adjoint(dA))
                @test same_sparse(B, copy(adjoint(A)))
                @test same_sparse(permutedims(dA), permutedims(A))
                @test same_sparse(permutedims(dA, (1, 2)), A)
                for S′ in sparse_matrix_formats
                    @test same_sparse(S′(transpose(dA)), copy(transpose(A)))
                    @test same_sparse(S′(adjoint(dA)), copy(adjoint(A)))
                end
            end

            # dense to sparse drops the zeros, like `sparse`
            D = Array(A)
            dD = AT(D)
            for S in sparse_matrix_formats
                B = S(dD)
                @test B isa S{ET,Int}
                check_structure(B)
                @test same_sparse(B, sparse(D))
                B = S{ET,Int32}(dD)
                @test B isa S{ET,Int32}
                @test same_sparse(B, SparseMatrixCSC{ET,Int32}(sparse(D)))
            end
            @test sparse(dD) isa GPUSparseMatrixCSC{ET,Int}
            @test same_sparse(sparse(dD), sparse(D))
            @test sparse(dD; fmt=:csr) isa GPUSparseMatrixCSR{ET,Int}
            @test sparse(dD; fmt=:coo) isa GPUSparseMatrixCOO{ET,Int}
            @test_throws ArgumentError sparse(dD; fmt=:bsr)
            for (m, n) in ((0, 0), (0, 3), (3, 0), (3, 4))
                Z = zeros(ET, m, n)
                for S in sparse_matrix_formats
                    @test same_sparse(S(AT(Z)), sparse(Z))
                end
            end

            x = Vector(sprand_nozeros(ET, 30, 0.3))
            dx = sparse(AT(x))
            @test dx isa GPUSparseVector{ET,Int}
            check_structure(dx)
            @test same_sparse(dx, sparse(x))
            @test GPUSparseVector{ET,Int32}(AT(x)) isa GPUSparseVector{ET,Int32}
            @test same_sparse(sparsevec(AT(x)), sparse(x))
            @test nnz(sparse(AT(zeros(ET, 0)))) == 0
        end
    end

    @testset "findnz" begin
        ET = first(eltypes)
        A = sprand_awkward(ET, 9, 7; Ti=Int32)
        I, J, V = findnz(A)
        # in particular for COO, whose conversion once recursed without end (CUDA.jl#3189)
        for S in sparse_matrix_formats
            dI, dJ, dV = findnz(gpu_sparse(AT, S, A))
            @test dI isa AT{Int32} && dJ isa AT{Int32} && dV isa AT{ET}
            @test Array(dI) == I && Array(dJ) == J && Array(dV) == V
        end
        x = SparseVector{ET,Int32}(sprand_nozeros(ET, 20, 0.3))
        dI, dV = findnz(gpu_sparse(AT, x))
        @test (Array(dI), Array(dV)) == findnz(x)
        @test all(isempty, findnz(gpu_sparse(AT, GPUSparseMatrixCOO, spzeros(ET, 3, 3))))
    end
end

# a callable that is not a `Function`
struct SparseSubtract end
(::SparseSubtract)(a, b) = a - b

function sparse_assembly(AT, eltypes)
    @testset "assembly" begin
        @testset "$ET" for ET in eltypes
            m, n = 7, 9
            N = 60
            # unordered, with many repeated coordinates
            I = rand(1:m, N)
            J = rand(1:n, N)
            V = rand(ET, N)
            dI, dJ, dV = AT(I), AT(J), AT(V)

            A = sparse(I, J, V, m, n)
            for fmt in (:csc, :csr, :coo)
                B = sparse(dI, dJ, dV, m, n; fmt)
                @test B isa GPUArrays.sparse_format(fmt){ET,Int}
                check_structure(B)
                @test SparseMatrixCSC(B) ≈ A
            end
            if ET <: Real
                @test same_sparse(sparse(dI, dJ, dV, m, n, max), sparse(I, J, V, m, n, max))
            end
            # a combine function that depends on the order of the repeated entries
            noncommutative(a, b) = a + a - b
            B = sparse(dI, dJ, dV, m, n, noncommutative)
            @test same_sparse(B, sparse(I, J, V, m, n, noncommutative))
            B = sparse(dI, dJ, dV, m, n, noncommutative; fmt=:csr)
            @test same_sparse(B, sparse(I, J, V, m, n, noncommutative))

            # repeated entries are folded in input order, also for `+`, which is not
            # associative in floating point
            if ET <: AbstractFloat
                big = ET(2)^(precision(ET) + 1)
                Is, Js, Vs = [2, 2, 2, 1], [3, 3, 3, 1], [big, -big, one(ET), one(ET)]
                @test same_sparse(sparse(AT(Is), AT(Js), AT(Vs), 2, 3), sparse(Is, Js, Vs, 2, 3))
                @test same_sparse(sparsevec(AT(Is), AT(Vs), 2), sparsevec(Is, Vs, 2))
            end
            # any callable
            @test same_sparse(sparse(dI, dJ, dV, m, n, SparseSubtract()), sparse(I, J, V, m, n, -))
            @test same_sparse(sparsevec(dI, dV, m, SparseSubtract()), sparsevec(I, V, m, -))

            # the dimensions follow from the coordinates
            @test size(sparse(dI, dJ, dV)) == size(sparse(I, J, V))
            # a single value, and index types
            @test same_sparse(sparse(dI, dJ, one(ET), m, n), sparse(I, J, one(ET), m, n))
            B = sparse(AT(Int32.(I)), AT(Int32.(J)), dV, m, n)
            @test B isa GPUSparseMatrixCSC{ET,Int32}
            @test SparseMatrixCSC(B) ≈ A
            # explicit zeros are kept, as in SparseArrays
            @test nnz(sparse(AT([1, 2]), AT([1, 2]), AT(zeros(ET, 2)), 2, 2)) == 2
            # nothing to assemble
            @test nnz(sparse(AT(Int[]), AT(Int[]), AT(ET[]), 3, 4)) == 0
            @test_throws ArgumentError sparse(dI, dJ, dV, m - 1, n)
            @test_throws ArgumentError sparse(dI, AT(J[1:end-1]), dV, m, n)

            # vectors
            x = sparsevec(I, V, m)
            dx = sparsevec(dI, dV, m)
            @test dx isa GPUSparseVector{ET,Int}
            check_structure(dx)
            @test SparseVector(dx) ≈ x
            @test same_sparse(sparsevec(dI, dV, m, noncommutative), sparsevec(I, V, m, noncommutative))
            @test length(sparsevec(dI, dV)) == length(sparsevec(I, V))
            @test same_sparse(sparsevec(dI, dV, noncommutative), sparsevec(I, V, noncommutative))

            # diagonals
            v = rand(ET, 5)
            w = rand(ET, 4)
            @test same_sparse(spdiagm(0 => AT(v), -1 => AT(w)), spdiagm(0 => v, -1 => w))
            @test same_sparse(spdiagm(7, 6, 1 => AT(w)), spdiagm(7, 6, 1 => w))
            @test same_sparse(spdiagm(AT(v)), spdiagm(v))
            @test_throws DimensionMismatch spdiagm(3, 3, 1 => AT(v))
            # without diagonals, SparseArrays applies
            @test which(spdiagm, Tuple{Int,Int}).module === SparseArrays
        end

        @testset "Bool" begin
            I = rand(1:5, 40)
            J = rand(1:6, 40)
            V = rand(Bool, 40)
            # repeated entries are combined with `|` by default
            @test same_sparse(sparse(AT(I), AT(J), AT(V), 5, 6), sparse(I, J, V, 5, 6))
            @test same_sparse(sparse(AT(I), AT(J), true, 5, 6), sparse(I, J, true, 5, 6))
            @test same_sparse(sparsevec(AT(I), AT(V), 5), sparsevec(I, V, 5))
        end
    end
end

# the operand wrappers LinearAlgebra encodes as characters in the storage-level `mul!`
function sparse_operands(A::AbstractMatrix)
    ops = Any[identity, transpose, adjoint]
    if size(A, 1) == size(A, 2)
        append!(ops, [x -> Symmetric(x, :U), x -> Symmetric(x, :L)])
        eltype(A) <: Complex && append!(ops, [x -> Hermitian(x, :U), x -> Hermitian(x, :L)])
    end
    return ops
end

function sparse_products(AT, eltypes)
    @testset "products" begin
        @testset "$S{$ET}" for S in sparse_matrix_formats, ET in eltypes
            # every operand character for one real and one complex element type, a few for
            # the others
            all_ops = ET == first(eltypes) || ET == first(filter(iscomplextype, eltypes))
            A = sprand_awkward(ET, 7, 7; Ti=Int32)
            R = sprand_awkward(ET, 7, 5; Ti=Int32)    # rectangular
            dA = gpu_sparse(AT, S, A)
            dR = gpu_sparse(AT, S, R)
            α, β = ET(2), ET(3)

            # sparse × dense vector (with dense references: SparseArrays lacks some of
            # these products for some element types)
            for (M, dM) in ((A, dA), (R, dR)), op in (all_ops ? sparse_operands(M) : [identity, adjoint])
                x = rand(ET, size(op(M), 2))
                y = rand(ET, size(op(M), 1))
                @test Array(op(dM) * AT(x)) ≈ op(Matrix(M)) * x
                dy = AT(copy(y))
                @test mul!(dy, op(dM), AT(x), α, β) === dy
                @test Array(dy) ≈ α * (op(Matrix(M)) * x) + β * y
            end

            # sparse × dense matrix, and dense × sparse
            dense_ops = all_ops ? [identity, transpose, adjoint] : [identity]
            for (M, dM) in ((A, dA), (R, dR)), op in (all_ops ? sparse_operands(M) : [identity, transpose]), dop in dense_ops
                B = rand(ET, 4, size(op(M), 2))
                B = dop === identity ? permutedims(B) : B   # op(M) * dop(B)
                ref = op(Matrix(M)) * dop(B)
                @test Array(op(dM) * dop(AT(B))) ≈ ref
                C = rand(ET, size(ref))
                dC = AT(copy(C))
                mul!(dC, op(dM), dop(AT(B)), α, β)
                @test Array(dC) ≈ α * ref + β * C

                D = rand(ET, size(op(M), 1), 4)
                D = dop === identity ? permutedims(D) : D   # dop(D) * op(M)
                ref = dop(D) * op(Matrix(M))
                @test Array(dop(AT(D)) * op(dM)) ≈ ref
                @test dop(AT(D)) * op(dM) isa AT
                C = rand(ET, size(ref))
                dC = AT(copy(C))
                mul!(dC, dop(AT(D)), op(dM), α, β)
                @test Array(dC) ≈ α * ref + β * C
            end
            if all_ops
                # dense Symmetric and Hermitian operands
                B = rand(ET, 7, 7)
                for wrap in (Symmetric, Hermitian), uplo in (:U, :L)
                    @test Array(dA * wrap(AT(B), uplo)) ≈ A * wrap(B, uplo)
                    @test Array(wrap(AT(B), uplo) * dA) ≈ wrap(B, uplo) * A
                end
            end

            # with β = 0 the destination is write-only
            if ET <: AbstractFloat
                dy = AT(fill(ET(NaN), 7))
                mul!(dy, dA, AT(ones(ET, 7)), true, false)
                @test Array(dy) ≈ A * ones(ET, 7)
                dC = AT(fill(ET(NaN), 7, 3))
                mul!(dC, dA, AT(ones(ET, 7, 3)), true, false)
                @test Array(dC) ≈ A * ones(ET, 7, 3)
                dC = AT(fill(ET(NaN), 3, 7))
                mul!(dC, AT(ones(ET, 3, 7)), dA, true, false)
                @test Array(dC) ≈ ones(ET, 3, 7) * A
            end

            # sparse vector operands are densified
            x = sprand_nozeros(ET, 7, 0.5)
            dx = gpu_sparse(AT, x)
            @test dA * dx isa AT{ET,1}
            @test Array(dA * dx) ≈ A * Vector(x)
            @test Array(AT(Matrix(A)) * dx) ≈ Matrix(A) * Vector(x)
            dy = AT(ones(ET, 7))
            mul!(dy, dA, dx, α, β)
            @test Array(dy) ≈ α * (A * Vector(x)) + β * ones(ET, 7)

            # empty operands
            Z = spzeros(ET, 3, 0)
            @test Array(gpu_sparse(AT, S, Z) * AT(zeros(ET, 0))) == zeros(ET, 3)
            @test Array(gpu_sparse(AT, S, spzeros(ET, 0, 3)) * AT(ones(ET, 3))) == zeros(ET, 0)
            @test Array(gpu_sparse(AT, S, spzeros(ET, 3, 4)) * AT(ones(ET, 4, 2))) == zeros(ET, 3, 2)
            @test Array(AT(ones(ET, 2, 3)) * gpu_sparse(AT, S, spzeros(ET, 3, 4))) == zeros(ET, 2, 4)
            @test_throws DimensionMismatch dR * AT(ones(ET, 7))
        end

        @testset "$T" for T in (Int, Bool)
            A = sprand(T, 6, 5, 0.5)
            x = rand(T, 5)
            B = rand(T, 5, 3)
            D = rand(T, 3, 6)
            for S in sparse_matrix_formats
                dA = gpu_sparse(AT, S, A)
                @test Array(dA * AT(x)) == A * x
                @test Array(transpose(dA) * AT(rand(T, 6))) isa Vector
                @test Array(dA * AT(B)) == A * B
                @test Array(AT(D) * dA) == D * A
            end
        end

        # Float16 products accumulate in Float32
        if Float16 in eltypes
            A = sprand(Float16, 50, 200, 0.5)
            x = rand(Float16, 200)
            ref = Float16.(Float32.(A) * Float32.(x))
            for S in sparse_matrix_formats
                @test Array(gpu_sparse(AT, S, A) * AT(x)) == ref
            end
        end
    end
end
