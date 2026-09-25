# products of sparse and dense arrays
#
# LinearAlgebra reduces products with transposed, adjoint, Symmetric and Hermitian operands
# to its storage-level `mul!(C, tA, [tB,] A, B, α, β)`, where a character describes each
# operand: 'N', 'T', 'C', and 'S'/'s' or 'H'/'h' for the upper or lower triangle of a
# Symmetric or Hermitian matrix. GPUArrays implements those methods for sparse operands
# (below), and back-ends override them for their vendor libraries, falling back to the
# generic implementations (`generic_spmv!`, `generic_spmm!`) where those do not apply.
#
# Every product is a gather over a compressed layout of the sparse operand, deterministic
# and valid for every element type. Transposed operands (Aᵀx for CSR, Ax for CSC) regroup
# the matrix first; an atomic scatter driven directly by the stored layout would avoid that.

# the element type to accumulate products in: Float16 loses too much precision
product_accumulator(::Type{T}) where {T} = T
product_accumulator(::Type{Float16}) = Float32
product_accumulator(::Type{ComplexF16}) = ComplexF32
function product_accumulator(Ta::Type, Tb::Type)
    T = typeof(zero(Ta) * zero(Tb) + zero(Ta) * zero(Tb))
    return product_accumulator(T)
end

## the operands

# The CSR buffers (pointers, indices, values) of `op(A)` for `t` in 'N', 'T' and 'C', with
# the conjugation of 'C' left to the kernel. Buffers of `A` itself are borrowed, others are
# temporary.
function csr_operand(A::GPUSparseMatrixCSR, t::AbstractChar)
    B = t == 'N' ? A : transpose_view(GPUSparseMatrixCSC(A))
    return B.rowPtr, B.colVal, B.nzVal
end
function csr_operand(A::GPUSparseMatrixCSC, t::AbstractChar)
    B = t == 'N' ? GPUSparseMatrixCSR(A) : transpose_view(A)
    return B.rowPtr, B.colVal, B.nzVal
end
function csr_operand(A::GPUSparseMatrixCOO, t::AbstractChar)
    t == 'N' || return csr_operand(GPUSparseMatrixCSC(A), t)
    # a COO matrix is a CSR matrix with explicit row indices
    return compress(A.rowInd, size(A, 1)), A.colInd, A.nzVal
end

# The CSC buffers of `op(A)`, which are the CSR buffers of its transpose. The conjugation
# of 'C' is again left to the kernel.
csc_operand(A::GPUSparseMatrix, t::AbstractChar) = csr_operand(A, t == 'N' ? 'T' : 'N')

# LinearAlgebra's characters for Symmetric and Hermitian operands combine two terms: the
# triangle of `A` that the wrapper uses, and the transpose (or adjoint) of that triangle
# without the diagonal. Each term is an operand character for the kernels, and a selection
# of the entries of `op(A)` by their position.
function operand_terms(t::AbstractChar)
    t in ('N', 'T', 'C') && return ((t, all_entries),)
    t == 'S' && return (('N', upper_entries), ('T', strict_lower_entries))
    t == 's' && return (('N', lower_entries), ('T', strict_upper_entries))
    t == 'H' && return (('N', upper_hermitian_entries), ('C', strict_lower_entries))
    t == 'h' && return (('N', lower_hermitian_entries), ('C', strict_upper_entries))
    throw(ArgumentError("unsupported operand character '$t'"))
end

# the contribution of a stored entry `v` at `(i, j)`, as selected by a term
all_entries(i, j, v) = v
upper_entries(i, j, v) = j >= i ? v : zero(v)
lower_entries(i, j, v) = j <= i ? v : zero(v)
strict_upper_entries(i, j, v) = j > i ? v : zero(v)
strict_lower_entries(i, j, v) = j < i ? v : zero(v)
# the diagonal of a Hermitian matrix is real
upper_hermitian_entries(i, j, v) = j > i ? v : j == i ? oftype(v, real(v)) : zero(v)
lower_hermitian_entries(i, j, v) = j < i ? v : j == i ? oftype(v, real(v)) : zero(v)

value_op(t::AbstractChar) = t == 'C' ? conj : identity


## kernels

## COV_EXCL_START
# y = α op(A) x + β y, one thread per row of op(A), in CSR layout
@kernel function csr_matvec_kernel(y, ptr, ind, val, x, α, β, entry, valop, ::Type{T}) where {T}
    i = @index(Global, Linear)
    if i <= length(y)
        acc = zero(T)
        for k in @inbounds(ptr[i]):@inbounds(ptr[i+1] - one(eltype(ptr)))
            j = @inbounds ind[k]
            v = entry(i, j, valop(@inbounds val[k]))
            acc += T(v) * T(@inbounds x[j])
        end
        # `y` is write-only when β is zero, so that it may hold NaN
        @inbounds y[i] = iszero(β) ? α * acc : α * acc + β * y[i]
    end
end

# C = α op(A) op(B) + β C with a sparse op(A) in CSR layout, one thread per element of C
@kernel function csr_matmat_kernel(C, ptr, ind, val, B, α, β, entry, valop, tB::Char,
                                   ::Type{T}) where {T}
    i, j = @index(Global, NTuple)
    if i <= size(C, 1) && j <= size(C, 2)
        acc = zero(T)
        for k in @inbounds(ptr[i]):@inbounds(ptr[i+1] - one(eltype(ptr)))
            l = @inbounds ind[k]
            v = entry(i, l, valop(@inbounds val[k]))
            b = tB == 'N' ? @inbounds(B[l, j]) :
                tB == 'T' ? @inbounds(B[j, l]) : conj(@inbounds(B[j, l]))
            acc += T(v) * T(b)
        end
        @inbounds C[i, j] = iszero(β) ? α * acc : α * acc + β * C[i, j]
    end
end

# C = α op(A) op(B) + β C with a sparse op(B) in CSC layout, one thread per element of C.
# Consecutive threads work down a column of C, reading a column of op(A) if it is not
# transposed, so that the dense accesses coalesce.
@kernel function csc_matmat_kernel(C, A, ptr, ind, val, α, β, entry, valop, tA::Char,
                                   ::Type{T}) where {T}
    i, j = @index(Global, NTuple)
    if i <= size(C, 1) && j <= size(C, 2)
        acc = zero(T)
        for k in @inbounds(ptr[j]):@inbounds(ptr[j+1] - one(eltype(ptr)))
            l = @inbounds ind[k]
            v = entry(l, j, valop(@inbounds val[k]))
            a = tA == 'N' ? @inbounds(A[i, l]) :
                tA == 'T' ? @inbounds(A[l, i]) : conj(@inbounds(A[l, i]))
            acc += T(a) * T(v)
        end
        @inbounds C[i, j] = iszero(β) ? α * acc : α * acc + β * C[i, j]
    end
end
## COV_EXCL_STOP

## generic implementations

# the dimensions of the result and of the operands of a product, as matrices
function check_product_dims(C::Dims, A::Dims, B::Dims)
    mat(sz) = (sz[1], get(sz, 2, 1))
    C, A, B = mat(C), mat(A), mat(B)
    A[2] == B[1] ||
        throw(DimensionMismatch("A has dimensions $A, B has dimensions $B"))
    C == (A[1], B[2]) ||
        throw(DimensionMismatch("C has dimensions $C, needs $((A[1], B[2]))"))
    return
end
op_size(A, t::AbstractChar) = t in ('T', 'C') ? reverse(size(A)) : size(A)

# C = β C, treating C as write-only when β is zero
scale!(C, β) = iszero(β) ? fill!(C, zero(eltype(C))) : (C .*= β; C)

## COV_EXCL_START
@kernel function materialize_symmetric_kernel(out, B, upper::Bool, hermitian::Bool)
    i, j = @index(Global, NTuple)
    if i <= size(out, 1) && j <= size(out, 2)
        # the triangle of `B` that holds the values
        stored = upper ? i <= j : i >= j
        v = stored ? @inbounds(B[i, j]) : @inbounds(B[j, i])
        if hermitian
            v = i == j ? oftype(v, real(v)) : stored ? v : conj(v)
        end
        @inbounds out[i, j] = v
    end
end
## COV_EXCL_STOP

# A dense operand wrapped in Symmetric or Hermitian is materialized; the kernels only index
# dense operands as they are, transposed, or adjoint.
function dense_operand(B, t::AbstractChar)
    t in ('N', 'T', 'C') && return B, t
    LinearAlgebra.checksquare(B)
    out = similar(B, eltype(B), size(B))
    isempty(out) ||
        materialize_symmetric_kernel(get_backend(out))(out, B, t in ('S', 'H'), t in ('H', 'h');
                                                       ndrange=size(out))
    return out, 'N'
end

"""
    GPUArrays.generic_spmv!(y, tA, A, x, α, β)

Compute `y = α op(A) x + β y` for a sparse matrix `A` and dense vectors `x` and `y`, where
`op` is given by LinearAlgebra's character `tA` ('N', 'T', 'C', or 'S'/'s'/'H'/'h' for the
upper or lower triangle of a Symmetric or Hermitian `A`). This is the generic
implementation behind `mul!(y, tA, A, x, α, β)`, which back-ends can call when a vendor
routine does not apply.
"""
function generic_spmv!(y::AnyGPUVector, tA::AbstractChar, A::GPUSparseMatrix,
                       x::AnyGPUVector, α::Number, β::Number)
    tA in ('S', 's', 'H', 'h') && LinearAlgebra.checksquare(A)
    check_product_dims(size(y), op_size(A, tA), size(x))
    isempty(y) && return y
    (isempty(x) || nnz(A) == 0) && return scale!(y, β)
    T = product_accumulator(eltype(A), eltype(x))
    for (n, (t, entry)) in enumerate(operand_terms(tA))
        ptr, ind, val = csr_operand(A, t)
        csr_matvec_kernel(get_backend(y))(y, ptr, ind, val, x, α, n == 1 ? β : true,
                                          entry, value_op(t), T; ndrange=length(y))
    end
    return y
end

"""
    GPUArrays.generic_spmm!(C, tA, tB, A, B, α, β)

Compute `C = α op(A) op(B) + β C` where one of `A` and `B` is a sparse matrix and the other
one, like `C`, is dense; `tA` and `tB` are LinearAlgebra's characters as for
[`generic_spmv!`](@ref). This is the generic implementation behind
`mul!(C, tA, tB, A, B, α, β)`.
"""
function generic_spmm!(C::AnyGPUMatrix, tA::AbstractChar, tB::AbstractChar,
                       A::GPUSparseMatrix, B::AnyGPUMatrix, α::Number, β::Number)
    tA in ('S', 's', 'H', 'h') && LinearAlgebra.checksquare(A)
    B, tB = dense_operand(B, tB)
    check_product_dims(size(C), op_size(A, tA), op_size(B, tB))
    isempty(C) && return C
    (size(B, tB == 'N' ? 1 : 2) == 0 || nnz(A) == 0) && return scale!(C, β)
    T = product_accumulator(eltype(A), eltype(B))
    for (n, (t, entry)) in enumerate(operand_terms(tA))
        ptr, ind, val = csr_operand(A, t)
        csr_matmat_kernel(get_backend(C))(C, ptr, ind, val, B, α, n == 1 ? β : true,
                                          entry, value_op(t), Char(tB), T; ndrange=size(C))
    end
    return C
end
function generic_spmm!(C::AnyGPUMatrix, tA::AbstractChar, tB::AbstractChar,
                       A::AnyGPUMatrix, B::GPUSparseMatrix, α::Number, β::Number)
    tB in ('S', 's', 'H', 'h') && LinearAlgebra.checksquare(B)
    A, tA = dense_operand(A, tA)
    check_product_dims(size(C), op_size(A, tA), op_size(B, tB))
    isempty(C) && return C
    (size(A, tA == 'N' ? 2 : 1) == 0 || nnz(B) == 0) && return scale!(C, β)
    T = product_accumulator(eltype(A), eltype(B))
    for (n, (t, entry)) in enumerate(operand_terms(tB))
        ptr, ind, val = csc_operand(B, t)
        csc_matmat_kernel(get_backend(C))(C, A, ptr, ind, val, α, n == 1 ? β : true,
                                          entry, value_op(t), Char(tA), T; ndrange=size(C))
    end
    return C
end


## LinearAlgebra integration

LinearAlgebra.mul!(C::AnyGPUVector, tA::AbstractChar, A::GPUSparseMatrix, B::AnyGPUVector,
                   α::Number, β::Number) =
    generic_spmv!(C, tA, A, B, α, β)
LinearAlgebra.mul!(C::AnyGPUMatrix, tA::AbstractChar, tB::AbstractChar, A::GPUSparseMatrix,
                   B::AnyGPUMatrix, α::Number, β::Number) =
    generic_spmm!(C, tA, tB, A, B, α, β)
LinearAlgebra.mul!(C::AnyGPUMatrix, tA::AbstractChar, tB::AbstractChar, A::AnyGPUMatrix,
                   B::GPUSparseMatrix, α::Number, β::Number) =
    generic_spmm!(C, tA, tB, A, B, α, β)

# Julia < 1.13 dispatches on the non-public `generic_matvecmul!` and `generic_matmatmul!`
# instead, as for dense arrays (see `linalg.jl`)
@static if VERSION < v"1.13.0-rc4"
    for (TA, TB) in ((:GPUSparseMatrix, :AnyGPUMatrix), (:AnyGPUMatrix, :GPUSparseMatrix))
        @eval begin
            LinearAlgebra.generic_matmatmul!(C::AnyGPUMatrix, tA::AbstractChar, tB::AbstractChar,
                                             A::$TA, B::$TB, a::Number, b::Number) =
                LinearAlgebra.mul!(C, tA, tB, A, B, a, b)
            LinearAlgebra.generic_matmatmul!(C::AnyGPUMatrix, tA::AbstractChar, tB::AbstractChar,
                                             A::$TA, B::$TB, _add::MulAddMul=MulAddMul()) =
                LinearAlgebra.mul!(C, tA, tB, A, B, _add.alpha, _add.beta)
        end
    end
    LinearAlgebra.generic_matvecmul!(C::AnyGPUVector, tA::AbstractChar, A::GPUSparseMatrix,
                                     B::AnyGPUVector, a::Number, b::Number) =
        LinearAlgebra.mul!(C, tA, A, B, a, b)
    LinearAlgebra.generic_matvecmul!(C::AnyGPUVector, tA::AbstractChar, A::GPUSparseMatrix,
                                     B::AnyGPUVector, _add::MulAddMul=MulAddMul()) =
        LinearAlgebra.mul!(C, tA, A, B, _add.alpha, _add.beta)
end

# LinearAlgebra allocates the result of a product like its second operand, which gives a
# sparse destination for dense × sparse
const GPUSparseMatrixOperand = Union{GPUSparseMatrix, Transpose{<:Any,<:GPUSparseMatrix},
                                     Adjoint{<:Any,<:GPUSparseMatrix},
                                     Symmetric{<:Any,<:GPUSparseMatrix},
                                     Hermitian{<:Any,<:GPUSparseMatrix}}
function Base.:(*)(A::AnyGPUMatrix, B::GPUSparseMatrixOperand)
    T = Base.promote_op(LinearAlgebra.matprod, eltype(A), eltype(B))
    return mul!(similar(A, T, (size(A, 1), size(B, 2))), A, B)
end

# A sparse vector operand is densified, as CUDA.jl does. (A product with a sparse vector
# could skip its implicit zeros, but that only pays for very sparse vectors.)
densify(x::GPUSparseVector) = copyto!(similar(x.nzVal, eltype(x), length(x)), x)

const GPUMatrixOperand = Union{GPUSparseMatrix, AnyGPUMatrix}
LinearAlgebra.mul!(C::AnyGPUVector, tA::AbstractChar, A::GPUMatrixOperand, B::GPUSparseVector,
                   α::Number, β::Number) =
    LinearAlgebra.mul!(C, tA, A, densify(B), α, β)
@static if VERSION < v"1.13.0-rc4"
    LinearAlgebra.generic_matvecmul!(C::AnyGPUVector, tA::AbstractChar, A::GPUMatrixOperand,
                                     B::GPUSparseVector, a::Number, b::Number) =
        LinearAlgebra.mul!(C, tA, A, B, a, b)
    LinearAlgebra.generic_matvecmul!(C::AnyGPUVector, tA::AbstractChar, A::GPUMatrixOperand,
                                     B::GPUSparseVector, _add::MulAddMul=MulAddMul()) =
        LinearAlgebra.mul!(C, tA, A, B, _add.alpha, _add.beta)
end

# like dense × sparse, LinearAlgebra would allocate a sparse destination
function Base.:(*)(A::Union{AnyGPUMatrix,GPUSparseMatrixOperand}, x::GPUSparseVector)
    T = Base.promote_op(LinearAlgebra.matprod, eltype(A), eltype(x))
    return mul!(similar(x.nzVal, T, size(A, 1)), A, x)
end


## sparse × sparse

## COV_EXCL_START
# the number of products every stored entry `A[i, l]` contributes: the length of row `l`
# of `B`
@kernel function spgemm_counts_kernel(counts, Aind, Bptr)
    e = @index(Global, Linear)
    if e <= length(counts)
        l = @inbounds Aind[e]
        @inbounds counts[e] = Int64(Bptr[l+1]) - Int64(Bptr[l])
    end
end

# every thread computes one product `A[i, l] * B[l, j]`, finding its entry of `A` by binary
# search over the running total of the counts
@kernel function spgemm_expand_kernel(I, J, V, offsets, Arow, Aind, Aval, Bptr, Bind, Bval)
    p = @index(Global, Linear)
    if p <= length(I)
        e = searchsortedfirst(offsets, Int64(p))
        q = p - (e == 1 ? 0 : @inbounds(offsets[e-1]))
        l = @inbounds Aind[e]
        k = @inbounds(Bptr[l]) + q - 1
        @inbounds I[p] = Arow[e]
        @inbounds J[p] = Bind[k]
        @inbounds V[p] = convert(eltype(V), Aval[e]) * convert(eltype(V), Bval[k])
    end
end
## COV_EXCL_STOP

"""
    GPUArrays.generic_spgemm(A, B)

The product of two sparse matrices, in the format of `A`, computed on the device by
expansion, sorting and compression (ESC): every product of a stored entry of `A` with one
of `B` is materialized, then the products are summed per coordinate as in
`sparse(I, J, V, m, n, +)`. The memory needed grows with the number of products. The
generic implementation behind `A * B`; hash-based accumulation with atomic insertion would
avoid materializing all products.
"""
function generic_spgemm(A::GPUSparseMatrix, B::GPUSparseMatrix)
    size(A, 2) == size(B, 1) ||
        throw(DimensionMismatch("A has dimensions $(size(A)), B has dimensions $(size(B))"))
    m, n = size(A, 1), size(B, 2)
    T = Base.promote_op(LinearAlgebra.matprod, eltype(A), eltype(B))
    Ti = promote_type(indtype(A), indtype(B))
    Aptr, Aind, Aval = csr_operand(A, 'N')
    Bptr, Bind, Bval = csr_operand(B, 'N')
    Arow = expand_ptr(Aptr, length(Aval))

    counts = similar(Aval, Int64, length(Aval))
    if !isempty(counts)
        spgemm_counts_kernel(get_backend(counts))(counts, Aind, Bptr; ndrange=length(counts))
    end
    offsets = accumulate(+, counts)
    free_buffer!(counts)
    total = isempty(offsets) ? 0 : Int(@allowscalar offsets[end])

    I = similar(Aind, Ti, total)
    J = similar(Aind, Ti, total)
    V = similar(Aval, product_accumulator(eltype(A), eltype(B)), total)
    if total > 0
        spgemm_expand_kernel(get_backend(V))(I, J, V, offsets, Arow, Aind, Aval, Bptr, Bind,
                                             Bval; ndrange=total)
    end
    free_buffer!(offsets)
    free_buffer!(Arow)
    fmt = A isa GPUSparseMatrixCSC ? :csc : A isa GPUSparseMatrixCSR ? :csr : :coo
    C = assemble_matrix(I, J, V, m, n, +, fmt)
    return eltype(C) == T ? C : with_eltypes(C, T, Ti)
end

# a sparse operand without a lazy wrapper
function materialize(S::Symmetric{<:Any,<:GPUSparseMatrix})
    A = parent(S)
    T = S.uplo == 'U' ? triu(A) : tril(A)
    strict = S.uplo == 'U' ? triu(A, 1) : tril(A, -1)
    return T .+ copy(transpose(strict))
end
function materialize(H::Hermitian{<:Any,<:GPUSparseMatrix})
    A = parent(H)
    strict = H.uplo == 'U' ? triu(A, 1) : tril(A, -1)
    return strict .+ copy(adjoint(strict)) .+ real.(band(A, 0, 0))
end

Base.:(*)(A::GPUSparseMatrixOperand, B::GPUSparseMatrixOperand) =
    generic_spgemm(materialize(A), materialize(B))

"""
    GPUArrays.generic_spgemm!(C, tA, tB, A, B, α, β)

Compute `C = α op(A) op(B) + β C` for sparse matrices, where `C` gets the union of the
structures of both terms (that of the product if `β` is zero). The buffers of `C` are
resized, so `C` must not share them with another array. The generic implementation behind
the sparse-output `mul!`.
"""
function generic_spgemm!(C::GPUSparseMatrix, tA::AbstractChar, tB::AbstractChar,
                         A::GPUSparseMatrix, B::GPUSparseMatrix, α::Number, β::Number)
    P = generic_spgemm(materialize(LinearAlgebra.wrap(A, tA)),
                       materialize(LinearAlgebra.wrap(B, tB)))
    size(P) == size(C) ||
        throw(DimensionMismatch("C has dimensions $(size(C)), needs $(size(P))"))
    R = iszero(β) ? α .* P : α .* P .+ β .* C
    return copyto!(C, convert(matrix_format(C), R))
end

LinearAlgebra.mul!(C::GPUSparseMatrix, tA::AbstractChar, tB::AbstractChar,
                   A::GPUSparseMatrix, B::GPUSparseMatrix, α::Number, β::Number) =
    generic_spgemm!(C, tA, tB, A, B, α, β)
@static if VERSION < v"1.13.0-rc4"
    LinearAlgebra.generic_matmatmul!(C::GPUSparseMatrix, tA::AbstractChar, tB::AbstractChar,
                                     A::GPUSparseMatrix, B::GPUSparseMatrix, a::Number, b::Number) =
        LinearAlgebra.mul!(C, tA, tB, A, B, a, b)
    LinearAlgebra.generic_matmatmul!(C::GPUSparseMatrix, tA::AbstractChar, tB::AbstractChar,
                                     A::GPUSparseMatrix, B::GPUSparseMatrix,
                                     _add::MulAddMul=MulAddMul()) =
        LinearAlgebra.mul!(C, tA, tB, A, B, _add.alpha, _add.beta)
end


## matrix exponential

"""
    exp(A::GPUSparseMatrix; threshold=1e-7, nonzero_tol=1e-14)

The matrix exponential of a square sparse matrix, by scaling and squaring a truncated
Taylor series, dropping entries below `nonzero_tol` as it goes. The series is truncated
when a term's largest magnitude falls below `threshold`.
"""
function LinearAlgebra.exp(A::GPUSparseMatrix; threshold=1e-7, nonzero_tol=1e-14)
    n = LinearAlgebra.checksquare(A)
    A = float(A)
    T = eltype(A)
    # scale A to norm at most 1 by a power of two
    scaling = max(one(real(T)), nextpow(2, max(norm(A, Inf), floatmin(real(T)))))
    A = A ./ T(scaling)
    Id = spdiagm(0 => fill!(similar(nonzeros(A), n), one(T)))
    P = convert(matrix_format(A), Id)
    term = P
    k = 1
    while true
        term = (T(1 / k) * A) * term
        droptol!(term, nonzero_tol)
        P = P + term
        norm(term, Inf) > threshold || break
        k += 1
    end
    for _ in 1:Int(log2(scaling))
        P = P * P
        if nnz(P) / length(P) < 0.25
            droptol!(P, nonzero_tol)
        end
    end
    return P
end
