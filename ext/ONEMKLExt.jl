module ONEMKLExt
using GPUArrays,oneAPI,LinearAlgebra,SparseArrays
using GPUArrays: UnsupportedSparseVendor

@static if Sys.islinux()
using oneAPI.oneMKL: oneAbstractSparseMatrix,oneSparseMatrixCSR
const MKL=oneAPI.oneMKL
struct OneSparseSpMV end
struct OneSparseSpMM end

function GPUArrays.check_storage_compatibility(arrays::oneArray...)
    first_array=first(arrays)
    all(A -> oneAPI.context(A)==oneAPI.context(first_array) &&
             oneAPI.device(A)==oneAPI.device(first_array),arrays) ||
        throw(ArgumentError("oneAPI sparse buffers must share a context and device"))
    nothing
end

function capability(C,A,B)
    eltype(A)===eltype(B)===eltype(C) && eltype(A)<:MKL.onemklFloat ||
        return UnsupportedSparseVendor("oneMKL sparse requires matching BLAS floating values")
    Ti=SparseArrays.indtype(A)
    Ti in (Int32,Int64) && eltype(first(GPUArrays.sparse_buffers(A)))===Ti ||
        return UnsupportedSparseVendor("oneMKL sparse requires uniform Int32 or Int64 indices")
    oneAPI.functional() && oneAPI.Support._sparse_abi_ok[] ||
        return UnsupportedSparseVendor("oneMKL sparse support ABI is unavailable")
    oneAPI.context()==oneAPI.context(A.nzVal) && oneAPI.device()==oneAPI.device(A.nzVal) ||
        return UnsupportedSparseVendor("oneMKL sparse execution requires the matrix context and device to be current")
    for X in (B,C)
        stride(X,1)==1 && (ndims(X)==1 || stride(X,2)>=max(1,size(X,1))) ||
            return UnsupportedSparseVendor("oneMKL sparse requires contiguous vectors and column-major matrices")
    end
    nothing
end
function GPUArrays.vendor_spmv(y::oneVector,trans,A::oneAbstractSparseMatrix,x::oneVector,alpha,beta)
    reason=capability(y,A,x)
    reason===nothing || return reason
    trans in ('N','T','C') || return UnsupportedSparseVendor("oneMKL executor requires ordinary operand characters")
    OneSparseSpMV()
end
GPUArrays.execute_spmv!(::OneSparseSpMV,y,trans,A,x,alpha,beta) =
    MKL.sparse_gemv!(Char(trans),alpha,A,x,beta,y)
function GPUArrays.vendor_spmm(C::oneMatrix,trans,transB,A::oneAbstractSparseMatrix,B::oneMatrix,alpha,beta)
    reason=capability(C,A,B)
    reason===nothing || return reason
    A isa MKL.oneSparseMatrixCOO && return UnsupportedSparseVendor("oneMKL SpMM requires CSR or CSC")
    trans in ('N','T','C') && transB in ('N','T','C') ||
        return UnsupportedSparseVendor("oneMKL executor requires ordinary operand characters")
    transB=='N' || return UnsupportedSparseVendor("oneMKL SpMM requires a nontransposed dense operand")
    OneSparseSpMM()
end
GPUArrays.execute_spmm!(::OneSparseSpMM,C,trans,transB,A,B,alpha,beta) =
    MKL.sparse_gemm!(Char(trans),Char(transB),alpha,A,B,beta,C)

using LinearAlgebra: BlasFloat
LinearAlgebra.generic_trimatdiv!(C::oneVector{T},uplo,diag,f::Function,A::oneSparseMatrixCSR{T},B::oneVector{T}) where {T<:BlasFloat} =
    MKL.sparse_trsv!(uplo,f===identity ? 'N' : f===transpose ? 'T' : 'C',diag,one(T),A,B,C)
LinearAlgebra.generic_trimatdiv!(C::oneMatrix{T},uplo,diag,f::Function,A::oneSparseMatrixCSR{T},B::oneMatrix{T}) where {T<:BlasFloat} =
    MKL.sparse_trsm!(uplo,f===identity ? 'N' : f===transpose ? 'T' : 'C','N',diag,one(T),A,B,C)
end
end
