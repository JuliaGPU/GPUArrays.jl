# Vendor triangular solves remain accelerated after replacing the ordinary containers.
using LinearAlgebra: BlasFloat, AdjOrTrans
using cuSPARSE: CuSparseMatrixCSR,CuSparseMatrixCSC,CuSparseMatrixCOO,CuSparseMatrixBSR,
                sv!,sm!,sv2!,sm2!

# LinearAlgebra inspects triangular wrappers before reaching the solve executor.
# Vendor-only storage must not fall through its host scalar traversal.
LinearAlgebra.istriu(A::CuSparseMatrixBSR,k::Integer=0) = istriu(CuSparseMatrixCSR(A),k)
LinearAlgebra.istril(A::CuSparseMatrixBSR,k::Integer=0) = istril(CuSparseMatrixCSR(A),k)

# triangular
for SparseMatrixType in (:CuSparseMatrixBSR,)
    @eval begin
        LinearAlgebra.generic_trimatdiv!(C::DenseCuVector{T}, uploc, isunitc, tfun::Function, A::$SparseMatrixType{T}, B::DenseCuVector{T}) where {T<:BlasFloat} =
            sv2!(tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', uploc, isunitc, one(T), A, C === B ? C : copyto!(C, B), 'O')
        LinearAlgebra.generic_trimatdiv!(C::DenseCuMatrix{T}, uploc, isunitc, tfun::Function, A::$SparseMatrixType{T}, B::DenseCuMatrix{T}) where {T<:BlasFloat} =
            sm2!(tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', 'N', uploc, isunitc, one(T), A, C === B ? C : copyto!(C, B), 'O')
        function LinearAlgebra.generic_trimatdiv!(C::DenseCuMatrix{T}, uploc, isunitc, tfun::Function, A::$SparseMatrixType{T}, B::AdjOrTrans{T,<:DenseCuMatrix{T}}) where {T<:BlasFloat}
            transb = LinearAlgebra.wrapper_char(B)
            (transb == 'C') && (T <: Complex) && throw(ErrorException("This operation is not supported by the current CUDA version."))
            C !== parent(B) && copyto!(C, B)
            sm2!(tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', 'N', uploc, isunitc, one(T), A, C, 'O')
        end
        function LinearAlgebra.generic_trimatdiv!(C::Transpose{T,<:DenseCuMatrix{T}}, uploc, isunitc, tfun::Function, A::$SparseMatrixType{T}, B::Transpose{T,<:DenseCuMatrix{T}}) where {T<:BlasFloat}
            B === C || throw(ErrorException("This operation is only supported if B and C are identical."))
            sm2!(tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', 'T', uploc, isunitc, one(T), A, parent(B), 'O')
        end
        function LinearAlgebra.generic_trimatdiv!(C::Adjoint{T,<:DenseCuMatrix{T}}, uploc, isunitc, tfun::Function, A::$SparseMatrixType{T}, B::Adjoint{T,<:DenseCuMatrix{T}}) where {T<:BlasFloat}
            B === C || throw(ErrorException("This operation is only supported if B and C are identical."))
            sm2!(tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', 'C', uploc, isunitc, one(T), A, parent(B), 'O')
        end
    end
end # SparseMatrixType loop

for SparseMatrixType in (:CuSparseMatrixCOO, :CuSparseMatrixCSR, :CuSparseMatrixCSC)
    @eval begin
        function LinearAlgebra.generic_trimatdiv!(C::DenseCuVector{T}, uploc, isunitc, tfun::Function, A::$SparseMatrixType{T}, B::DenseCuVector{T}) where {T<:BlasFloat}
            sv!(tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', uploc, isunitc, one(T), A, B, C, 'O')
        end
        function LinearAlgebra.generic_trimatdiv!(C::DenseCuMatrix{T}, uploc, isunitc, tfun::Function, A::$SparseMatrixType{T}, B::DenseCuMatrix{T}) where {T<:BlasFloat}
            sm!(tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', 'N', uploc, isunitc, one(T), A, B, C, 'O')
        end
        function LinearAlgebra.generic_trimatdiv!(C::DenseCuMatrix{T}, uploc, isunitc, tfun::Function, A::$SparseMatrixType{T}, B::AdjOrTrans{T,<:DenseCuMatrix{T}}) where {T<:BlasFloat}
            transb = LinearAlgebra.wrapper_char(B)
            (transb == 'C') && (T <: Complex) && throw(ErrorException("This operation is not supported by the current CUDA version."))
            sm!(tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', transb, uploc, isunitc, one(T), A, parent(B), C, 'O')
        end
        function LinearAlgebra.generic_trimatdiv!(C::Transpose{T,<:DenseCuMatrix{T}}, uploc, isunitc, tfun::Function, A::$SparseMatrixType{T}, B::Transpose{T,<:DenseCuMatrix{T}}) where {T<:BlasFloat}
            (B !== C) && throw(ErrorException("This operation is only supported if B and C are identical."))
            sm!(tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', 'T', uploc, isunitc, one(T), A, parent(B), parent(C), 'O')
        end
        function LinearAlgebra.generic_trimatdiv!(C::Adjoint{T,<:DenseCuMatrix{T}}, uploc, isunitc, tfun::Function, A::$SparseMatrixType{T}, B::Adjoint{T,<:DenseCuMatrix{T}}) where {T<:BlasFloat}
            (B !== C) && throw(ErrorException("This operation is only supported if B and C are identical."))
            sm!(tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', 'C', uploc, isunitc, one(T), A, parent(B), parent(C), 'O')
        end
    end
end # SparseMatrixType loop
