module CUSPARSEExt

using GPUArrays, CUDACore, cuSPARSE, LinearAlgebra, SparseArrays
using GPUArrays: UnsupportedSparseVendor
using cuSPARSE: CuCanonicalMatrix, CuCanonicalSparse
GPUArrays.product_buffers(A::cuSPARSE.CuSparseMatrixBSR) = (A.rowPtr,A.colVal,A.nzVal)

struct CuSparseSpMV end
struct CuSparseSpMM end
struct CuSparseSpGEMM end

function GPUArrays.normalize_sparse_product_operand(A::cuSPARSE.CuSparseMatrixBSR,t,algorithm)
    t in ('S','s','H','h') || return A,t
    algorithm isa GPUArrays.Portable &&
        throw(ArgumentError("portable conversion of BSR storage is not implemented"))
    # Vendor-only BSR conversion establishes canonical storage before GPUArrays
    # materializes the wrapper and selects the numerical product algorithm.
    canonical=cuSPARSE.CuSparseMatrixCSR(A)
    return GPUArrays.materialize(LinearAlgebra.wrap(canonical,t)),'N'
end

function GPUArrays.check_storage_compatibility(arrays::CuArray...)
    ctx = CUDACore.context(first(arrays))
    all(A -> CUDACore.context(A) == ctx, arrays) ||
        throw(ArgumentError("CUDA sparse operands must belong to the same context"))
    return nothing
end

const VendorFloat = Union{Float16,ComplexF16,Float32,Float64,ComplexF32,ComplexF64}
const VendorNumeric = Union{Integer,AbstractFloat,Complex{<:AbstractFloat}}

function uniform_vendor_indices(A)
    Ti = SparseArrays.indtype(A)
    Ti in (Int32,Int64) || return false
    A isa Union{cuSPARSE.CuSparseMatrixCSR,cuSPARSE.CuSparseMatrixCSC} || return true
    return eltype(GPUArrays.compressed_ptr(A)) === Ti
end

function product_capability(C,A,B)
    T = eltype(C)
    T <: VendorFloat || return UnsupportedSparseVendor("cuSPARSE products require a supported floating output type")
    eltype(A) <: VendorNumeric && eltype(B) <: VendorNumeric ||
        return UnsupportedSparseVendor("cuSPARSE cannot normalize these operand value types")
    T <: Real && (eltype(A) <: Complex || eltype(B) <: Complex) &&
        return UnsupportedSparseVendor("cuSPARSE does not convert complex operands to real output")
    GPUArrays.product_accumulator(eltype(A),eltype(B)) === GPUArrays.product_accumulator(T,T) ||
        return UnsupportedSparseVendor("normalizing to this output type would change sparse accumulation precision")
    if T in (Float16,ComplexF16)
        all(S -> S===T || S===Bool || (T===ComplexF16 && S===Float16),(eltype(A),eltype(B))) ||
            return UnsupportedSparseVendor("normalizing operands to half precision would lose input values")
    end
    cuSPARSE.functional() || return UnsupportedSparseVendor("cuSPARSE is unavailable")
    T === Float16 && cuSPARSE.version() < v"11.4" &&
        return UnsupportedSparseVendor("Float16 products require cuSPARSE 11.4")
    T === ComplexF16 && cuSPARSE.version() < v"11.7.2" &&
        return UnsupportedSparseVendor("ComplexF16 products require cuSPARSE 11.7.2")
    return nothing
end

function GPUArrays.vendor_spmv(y::CuVector,tA,A::CuCanonicalMatrix,x::CuVector,α,β)
    reason = product_capability(y,A,x)
    reason === nothing || return reason
    uniform_vendor_indices(A) || return UnsupportedSparseVendor("this cuSPARSE executor requires uniform Int32 or Int64 indices")
    tA in ('N','T','C') || return UnsupportedSparseVendor("this cuSPARSE executor does not materialize symmetric wrappers")
    stride(x,1) == stride(y,1) == 1 || return UnsupportedSparseVendor("cuSPARSE vector descriptors require unit strides")
    A isa cuSPARSE.CuSparseMatrixCSC && tA == 'C' && eltype(A) <: Complex && cuSPARSE.version() < v"12" &&
        return UnsupportedSparseVendor("complex CSC adjoint requires cuSPARSE 12")
    # Floating mixed precision is not universally supported by cuSPARSE SpMV. Integer
    # and Bool operands retain upstream's explicit normalization to the output type.
    return CuSparseSpMV()
end

function GPUArrays.execute_spmv!(::CuSparseSpMV,y,tA,A,x,α,β)
    T = eltype(y)
    A = vendor_values(A,T)
    x = eltype(x) === T ? x : convert(CuVector{T},x)
    cuSPARSE.mv!(Char(tA),α,A,x,β,y,'O',cuSPARSE.CUSPARSE_SPMV_ALG_DEFAULT)
end

function spmm_capability(C,tA,tB,A,B)
    reason = product_capability(C,A,B)
    reason === nothing || return reason
    S = A isa CuCanonicalMatrix ? A : B
    uniform_vendor_indices(S) || return UnsupportedSparseVendor("this cuSPARSE executor requires uniform Int32 or Int64 indices")
    tA in ('N','T','C') && tB in ('N','T','C') ||
        return UnsupportedSparseVendor("this cuSPARSE executor does not materialize symmetric wrappers")
    D = A isa CuCanonicalMatrix ? B : A
    sparse_trans = A isa CuCanonicalMatrix ? tA : tB
    dense_trans = A isa CuCanonicalMatrix ? tB : tA
    if eltype(C)<:Union{Float16,ComplexF16}
        # Dense-times-sparse is executed as a transposed sparse-times-dense product.
        # Half storage uses a wider accumulator with stricter vendor layout support.
        supported = A isa CuCanonicalMatrix ?
            !(S isa cuSPARSE.CuSparseMatrixCSC) && sparse_trans=='N' :
            S isa cuSPARSE.CuSparseMatrixCSC ? sparse_trans=='N' :
            S isa cuSPARSE.CuSparseMatrixCOO && sparse_trans!='N'
        supported || return UnsupportedSparseVendor("cuSPARSE half-precision SpMM does not support this sparse layout or operation")
        eltype(C)<:Complex && dense_trans=='C' &&
            return UnsupportedSparseVendor("cuSPARSE half-precision SpMM does not support dense conjugate transpose")
    end
    S isa cuSPARSE.CuSparseMatrixCSC && sparse_trans=='C' && eltype(S)<:Complex && cuSPARSE.version()<v"12" &&
        return UnsupportedSparseVendor("complex CSC adjoint requires cuSPARSE 12")
    B isa cuSPARSE.CuSparseMatrixCSR && tB=='C' && eltype(B)<:Complex && cuSPARSE.version()<v"12" &&
        return UnsupportedSparseVendor("transformed dense-times-CSR complex adjoint requires cuSPARSE 12")
    cuSPARSE.version()<v"11" && (B isa CuCanonicalMatrix || (dense_trans!='N' && !(S isa cuSPARSE.CuSparseMatrixCOO))) &&
        return UnsupportedSparseVendor("these dense SpMM layouts require cuSPARSE 11")
    stride(D,1) == stride(C,1) == 1 && stride(D,2) >= max(1,size(D,1)) &&
        stride(C,2) >= max(1,size(C,1)) ||
        return UnsupportedSparseVendor("cuSPARSE matrix descriptors require column-major dense storage")
    return CuSparseSpMM()
end
GPUArrays.vendor_spmm(C::CuMatrix,tA,tB,A::CuCanonicalMatrix,B::CuMatrix,α,β) =
    spmm_capability(C,tA,tB,A,B)
GPUArrays.vendor_spmm(C::CuMatrix,tA,tB,A::CuMatrix,B::CuCanonicalMatrix,α,β) =
    spmm_capability(C,tA,tB,A,B)

function GPUArrays.execute_spmm!(::CuSparseSpMM,C,tA,tB,A,B,α,β)
    T = eltype(C)
    normalize(X::Union{CuCanonicalMatrix,cuSPARSE.CuSparseMatrixBSR}) = vendor_values(X,T)
    normalize(X::CuMatrix) = eltype(X) === T ? X : convert(CuMatrix{T},X)
    cuSPARSE.mm!(Char(tA),Char(tB),α,normalize(A),normalize(B),β,C,'O',cuSPARSE.CUSPARSE_SPMM_ALG_DEFAULT)
end

function GPUArrays.vendor_spgemm(A::CuCanonicalMatrix,B::CuCanonicalMatrix)
    eltype(A) === eltype(B) && eltype(A) <: LinearAlgebra.BlasFloat ||
        return UnsupportedSparseVendor("cuSPARSE SpGEMM requires matching BLAS floating value types")
    uniform_vendor_indices(A) && uniform_vendor_indices(B) &&
        SparseArrays.indtype(A) === Int32 && SparseArrays.indtype(B) === Int32 ||
        return UnsupportedSparseVendor("this cuSPARSE SpGEMM executor requires uniform Int32 storage")
    max(nnz(A),nnz(B)) < typemax(Int32) ||
        return UnsupportedSparseVendor("this cuSPARSE SpGEMM executor requires Int32-representable stored counts")
    cuSPARSE.functional() && cuSPARSE.version() >= v"11" ||
        return UnsupportedSparseVendor("cuSPARSE SpGEMM requires cuSPARSE 11")
    return CuSparseSpGEMM()
end
function GPUArrays.execute_spgemm(::CuSparseSpGEMM,A,B;max_temp_bytes)
    # Regroup explicitly here; the selected execution never calls public multiplication.
    Acsr = A isa cuSPARSE.CuSparseMatrixCSR ? A :
           A isa cuSPARSE.CuSparseMatrixCSC ? GPUArrays.generic_regroup(A) : GPUArrays.generic_compress(A)
    Bcsr = B isa cuSPARSE.CuSparseMatrixCSR ? B :
           B isa cuSPARSE.CuSparseMatrixCSC ? GPUArrays.generic_regroup(B) : GPUArrays.generic_compress(B)
    # COO has no pointer prototype: portable compression chooses Int pointers. The
    # selected vendor ABI requires Int32, already proved representable above.
    A isa cuSPARSE.CuSparseMatrixCOO && (Acsr = GPUArrays.with_eltypes(Acsr,eltype(A),Int32))
    B isa cuSPARSE.CuSparseMatrixCOO && (Bcsr = GPUArrays.with_eltypes(Bcsr,eltype(B),Int32))
    C = cuSPARSE.gemm('N','N',one(eltype(A)),Acsr,Bcsr,'O',cuSPARSE.CUSPARSE_SPGEMM_DEFAULT)
    convert(GPUArrays.matrix_format(A),C)
end

function GPUArrays.vendor_spmv(y::CuVector,tA,A::cuSPARSE.CuSparseMatrixBSR,x::CuVector,α,β)
    reason = product_capability(y,A,x)
    reason === nothing || return reason
    tA in ('N','T','C') && SparseArrays.indtype(A) === Int32 &&
        eltype(A.rowPtr) === Int32 && cuSPARSE.version() >= v"12.6.3" ||
        return UnsupportedSparseVendor("BSR products require Int32 storage, ordinary operands and cuSPARSE 12.6.3")
    return CuSparseSpMV()
end
function GPUArrays.vendor_spmm(C::CuMatrix,tA,tB,A::cuSPARSE.CuSparseMatrixBSR,B::CuMatrix,α,β)
    reason = product_capability(C,A,B)
    reason === nothing || return reason
    tA in ('N','T','C') && tB in ('N','T','C') && SparseArrays.indtype(A) === Int32 &&
        eltype(A.rowPtr) === Int32 && cuSPARSE.version() >= v"12.6.3" ||
        return UnsupportedSparseVendor("BSR products require Int32 storage, ordinary operands and cuSPARSE 12.6.3")
    return CuSparseSpMM()
end

vendor_values(A::CuCanonicalMatrix,T) = eltype(A) === T ? A : GPUArrays.with_values(A,T)
vendor_values(A::cuSPARSE.CuSparseMatrixBSR,T) = eltype(A) === T ? A : cuSPARSE.CuSparseMatrixBSR{T}(A)

# Backend-only formats normalize once, then use centrally owned arithmetic.
for op in (:+,:-)
    @eval begin
        Base.$op(A::cuSPARSE.CuSparseMatrixBSR,B::GPUArrays.GPUSparseMatrix)=Base.$op(cuSPARSE.CuSparseMatrixCSR(A),B)
        Base.$op(A::GPUArrays.GPUSparseMatrix,B::cuSPARSE.CuSparseMatrixBSR)=Base.$op(A,cuSPARSE.CuSparseMatrixCSR(B))
        Base.$op(A::cuSPARSE.CuSparseMatrixBSR,B::cuSPARSE.CuSparseMatrixBSR)=Base.$op(cuSPARSE.CuSparseMatrixCSR(A),cuSPARSE.CuSparseMatrixCSR(B))
    end
end

include("CUSPARSEExt/solves.jl")

end
