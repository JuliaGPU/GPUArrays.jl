## Wrappers

trans_adj_wrappers_dense_vecormat = ((T -> :(AbstractGPUVecOrMat{$T}), false, identity, identity),
    (T -> :(Transpose{$T,<:AbstractGPUMatrix{$T}}), true, identity, A -> :(parent($A))),
    (T -> :(Adjoint{$T,<:AbstractGPUMatrix{$T}}), true, x -> :(conj($x)), A -> :(parent($A))))

trans_adj_wrappers_csc = ((T -> :(AbstractGPUSparseMatrixCSC{$T}), false, identity, identity),
    (T -> :(Transpose{$T,<:AbstractGPUSparseMatrixCSC{$T}}), true, identity, A -> :(parent($A))),
    (T -> :(Adjoint{$T,<:AbstractGPUSparseMatrixCSC{$T}}), true, x -> :(conj($x)), A -> :(parent($A))))

## Sparse Vector

SparseArrays.getnzval(V::AbstractGPUSparseVector) = nonzeros(V)
SparseArrays.nnz(V::AbstractGPUSparseVector) = length(nzval(V))

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
LinearAlgebra.dot(x::AbstractGPUVector{T}, y::AbstractGPUSparseVector{T}) where {T<:Real} = dot(y, x)
LinearAlgebra.dot(x::AbstractGPUVector{T}, y::AbstractGPUSparseVector{T}) where {T<:Complex} = conj(dot(y, x))


## General Sparse Matrix

KernelAbstractions.get_backend(A::AbstractGPUSparseMatrix) = KernelAbstractions.get_backend(getnzval(A))

SparseArrays.getnzval(A::AbstractGPUSparseMatrix) = nonzeros(A)
SparseArrays.nnz(A::AbstractGPUSparseMatrix) = length(getnzval(A))

function LinearAlgebra.rmul!(A::AbstractGPUSparseMatrix, x::Number)
    rmul!(getnzval(A), x)
    return A
end

function LinearAlgebra.lmul!(x::Number, A::AbstractGPUSparseMatrix)
    lmul!(x, getnzval(A))
    return A
end

## CSC Matrix

SparseArrays.getrowval(A::AbstractGPUSparseMatrixCSC) = rowvals(A)
# SparseArrays.nzrange(A::AbstractGPUSparseMatrixCSC, col::Integer) = getcolptr(A)[col]:(getcolptr(A)[col+1]-1) # TODO: this uses scalar indexing

function _goodbuffers_csc(m, n, colptr, rowval, nzval)
    return (length(colptr) == n + 1 && length(rowval) == length(nzval))
    # TODO: also add the condition that colptr[end] - 1 == length(nzval) (allowscalar?)
end

# @inline function LinearAlgebra.mul!(C::AbstractGPUVector, A::AnyGPUSparseMatrixCSC, B::AbstractGPUVector, α::Number, β::Number)
#     return LinearAlgebra.generic_matvecmul!(C, LinearAlgebra.wrapper_char(A), LinearAlgebra._unwrap(A), B, LinearAlgebra.MulAddMul(α, β))
# end

@inline function LinearAlgebra.generic_matvecmul!(C::AbstractGPUVector, tA, A::AbstractGPUSparseMatrixCSC, B::AbstractGPUVector, _add::LinearAlgebra.MulAddMul)
    return _spmatmul!(C, wrap(A, tA), B, _add.alpha, _add.beta)
end

@inline function LinearAlgebra.generic_matmatmul!(C::AbstractGPUMatrix, tA, tb, A::AbstractGPUSparseMatrixCSC, B::AbstractGPUMatrix, _add::LinearAlgebra.MulAddMul)
    return _spmatmul!(C, wrap(A, tA), wrap(B, tb), _add.alpha, _add.beta)
end

for (wrapa, transa, opa, unwrapa) in trans_adj_wrappers_csc
    for (wrapb, transb, opb, unwrapb) in trans_adj_wrappers_dense_vecormat
        TypeA = wrapa(:(T1))
        TypeB = wrapb(:(T2))
        TypeC = :(AbstractGPUVecOrMat{T3})

        kernel_spmatmul! = transa ? :kernel_spmatmul_T! : :kernel_spmatmul_N!

        indB = transb ? (i, j) -> :(($j, $i)) : (i, j) -> :(($i, $j)) # transpose indices

        @eval function _spmatmul!(C::$TypeC, A::$TypeA, B::$TypeB, α::Number, β::Number) where {T1,T2,T3}
            size(A, 2) == size(B, 1) ||
                throw(DimensionMismatch("second dimension of A, $(size(A,2)), does not match the first dimension of B, $(size(B,1))"))
            size(A, 1) == size(C, 1) ||
                throw(DimensionMismatch("first dimension of A, $(size(A,1)), does not match the first dimension of C, $(size(C,1))"))
            size(B, 2) == size(C, 2) ||
                throw(DimensionMismatch("second dimension of B, $(size(B,2)), does not match the second dimension of C, $(size(C,2))"))

            _A = $(unwrapa(:A))
            _B = $(unwrapb(:B))

            backend_C = KernelAbstractions.get_backend(C)
            backend_A = KernelAbstractions.get_backend(_A)
            backend_B = KernelAbstractions.get_backend(_B)

            backend_A == backend_B == backend_C || throw(ArgumentError("All arrays must be on the same backend"))

            @kernel function kernel_spmatmul_N!(C, @Const(A), @Const(B))
                k, col = @index(Global, NTuple)

                Bi, Bj = $(indB(:col, :k))

                @inbounds axj = $(opb(:(B[Bi, Bj]))) * α
                @inbounds for j in getcolptr(A)[col]:(getcolptr(A)[col+1]-1) # nzrange(A, col)
                    KernelAbstractions.@atomic C[getrowval(A)[j], k] += $(opa(:(getnzval(A)[j]))) * axj
                end
            end

            @kernel function kernel_spmatmul_T!(C, @Const(A), @Const(B))
                k, col = @index(Global, NTuple)

                tmp = zero(eltype(C))
                @inbounds for j in getcolptr(A)[col]:(getcolptr(A)[col+1]-1) # nzrange(A, col)
                    Bi, Bj = $(indB(:(getrowval(A)[j]), :k))
                    tmp += $(opa(:(getnzval(A)[j]))) * $(opb(:(B[Bi, Bj])))
                end
                @inbounds C[col, k] += tmp * α
            end

            β != one(β) && LinearAlgebra._rmul_or_fill!(C, β)

            kernel! = $kernel_spmatmul!(backend_A)
            kernel!(C, _A, _B; ndrange=(size(C, 2), size(_A, 2)))

            return C
        end
    end
end
