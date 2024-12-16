## Sparse Vector

Base.length(V::AbstractGPUSparseVector) = V.n
Base.size(V::AbstractGPUSparseVector) = (V.n,)

SparseArrays.nonzeros(V::AbstractGPUSparseVector) = V.nzval
SparseArrays.getnzval(V::AbstractGPUSparseVector) = nonzeros(V)
SparseArrays.nnz(V::AbstractGPUSparseVector) = length(nzval(V))
SparseArrays.nonzeroinds(V::AbstractGPUSparseVector) = V.nzind

function Base.sizehint!(V::AbstractGPUSparseVector, newlen::Integer)
    sizehint!(nonzeroinds(V), newlen)
    sizehint!(nonzeros(V), newlen)
    return V
end

function LinearAlgebra.dot(x::AbstractGPUSparseVector, y::AbstractGPUVector)
    n = length(y)
    length(x) == n || throw(DimensionMismatch(
        "Vector x has a length $(length(x)) but y has a length $n"))
    nzind = nonzeroinds(x)
    nzval = nonzeros(x)
    y_view = y[nzind] # TODO: by using the view it throws scalar indexing
    return dot(nzval, y_view)
end
LinearAlgebra.dot(x::AbstractGPUVector{T}, y::AbstractGPUSparseVector{T}) where T<:Real = dot(y, x)
LinearAlgebra.dot(x::AbstractGPUVector{T}, y::AbstractGPUSparseVector{T}) where T<:Complex = conj(dot(y, x))


## General Sparse Matrix

Base.size(A::AbstractGPUSparseMatrix) = (A.m, A.n)

SparseArrays.nonzeros(A::AbstractGPUSparseMatrix) = A.nzval
SparseArrays.getnzval(A::AbstractGPUSparseMatrix) = nonzeros(A)
SparseArrays.nnz(A::AbstractGPUSparseMatrix) = length(nzval(A))

function LinearAlgebra.rmul!(A::AbstractGPUSparseMatrix, x::Number)
    rmul!(SparseArrays.getnzval(A), x)
    return A
end

function LinearAlgebra.lmul!(x::Number, A::AbstractGPUSparseMatrix)
    lmul!(x, SparseArrays.getnzval(A))
    return A
end

## CSC Matrix

SparseArrays.getcolptr(A::AbstractGPUSparseMatrixCSC) = A.colptr
SparseArrays.rowvals(A::AbstractGPUSparseMatrixCSC) = A.rowval
SparseArrays.getrowval(A::AbstractGPUSparseMatrixCSC) = rowvals(A)
# SparseArrays.nzrange(A::AbstractGPUSparseMatrixCSC, col::Integer) = getcolptr(A)[col]:(getcolptr(A)[col+1]-1) # TODO: this uses scalar indexing

function _goodbuffers_csc(m, n, colptr, rowval, nzval)
    return (length(colptr) == n + 1 && length(rowval) == length(nzval))
    # TODO: also add the condition that colptr[end] - 1 == length(nzval) (allowscalar?)
end

@inline function LinearAlgebra.mul!(C::AbstractGPUVector, A::AbstractGPUSparseMatrixCSC, B::AbstractGPUVector, α::Number, β::Number)
    return LinearAlgebra.generic_matvecmul!(C, LinearAlgebra.wrapper_char(A), LinearAlgebra._unwrap(A), B, LinearAlgebra.MulAddMul(α, β))
end

@inline function LinearAlgebra.generic_matvecmul!(C::AbstractGPUVector, tA, A::AbstractGPUSparseMatrixCSC, B::AbstractGPUVector, _add::LinearAlgebra.MulAddMul)
    return SparseArrays.spdensemul!(C, tA, 'N', A, B, _add)
end

Base.@constprop :aggressive function SparseArrays.spdensemul!(C::AbstractGPUVecOrMat, tA, tB, A::AbstractGPUSparseMatrixCSC, B::AbstractGPUVecOrMat, _add::LinearAlgebra.MulAddMul)
    if tA == 'N'
        return _spmatmul!(C, A, wrap(B, tB), _add.alpha, _add.beta)
    else
        throw(ArgumentError("tA different from 'N' not yet supported"))
    end
end

function _spmatmul!(C::AbstractGPUVecOrMat, A::AbstractGPUSparseMatrixCSC, B::AbstractGPUVecOrMat, α::Number, β::Number)
    size(A, 2) == size(B, 1) ||
        throw(DimensionMismatch("second dimension of A, $(size(A,2)), does not match the first dimension of B, $(size(B,1))"))
    size(A, 1) == size(C, 1) ||
        throw(DimensionMismatch("first dimension of A, $(size(A,1)), does not match the first dimension of C, $(size(C,1))"))
    size(B, 2) == size(C, 2) ||
        throw(DimensionMismatch("second dimension of B, $(size(B,2)), does not match the second dimension of C, $(size(C,2))"))
    
    A_colptr = getcolptr(A)
    A_rowval = rowvals(A)
    A_nzval = getnzval(A)
    β != one(β) && LinearAlgebra._rmul_or_fill!(C, β)

    @kernel function kernel_spmatmul!(C, @Const(A_colptr), @Const(A_rowval), @Const(A_nzval), @Const(B))
        k, col = @index(Global, NTuple)

        @inbounds axj = B[col, k] * α
        @inbounds for j in A_colptr[col]:(A_colptr[col+1]-1) # nzrange(A, col)
            KernelAbstractions.@atomic C[A_rowval[j], k] += A_nzval[j] * axj
        end
    end

    backend_C = KernelAbstractions.get_backend(C)
    backend_A = KernelAbstractions.get_backend(A_nzval)
    backend_B = KernelAbstractions.get_backend(B)

    backend_A == backend_B == backend_C || throw(ArgumentError("All arrays must be on the same backend"))

    kernel! = kernel_spmatmul!(backend_A)
    kernel!(C, A_colptr, A_rowval, A_nzval, B; ndrange=(size(C, 2), size(A, 2)))

    return C
end
