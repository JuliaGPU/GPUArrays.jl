module ROCSPARSEExt
using GPUArrays,AMDGPU,SparseArrays,LinearAlgebra
using GPUArrays: UnsupportedSparseVendor
using AMDGPU.rocSPARSE: ROCCanonicalMatrix,ROCSparseMatrix,ROCSparseMatrixCSR,
    ROCSparseMatrixCSC,ROCSparseMatrixCOO,ROCSparseMatrixBSR
const SP=AMDGPU.rocSPARSE
struct ROCSpMV end
struct ROCSpMM end
struct ROCSpGEMM end

function GPUArrays.check_storage_compatibility(arrays::ROCArray...)
    all(A -> AMDGPU.device(A)==AMDGPU.device(first(arrays)),arrays) ||
        throw(ArgumentError("ROCm sparse operands must share a device"))
    nothing
end
GPUArrays.product_buffers(A::ROCSparseMatrixBSR)=(A.rowPtr,A.colVal,A.nzVal)
function capability(C,A,B)
    eltype(A)===eltype(B)===eltype(C) && eltype(C)<:LinearAlgebra.BlasFloat ||
        return UnsupportedSparseVendor("rocSPARSE requires matching BLAS floating types")
    AMDGPU.functional(:rocsparse) || return UnsupportedSparseVendor("rocSPARSE is unavailable")
    sparse=A isa ROCSparseMatrix ? A : B
    AMDGPU.device()==AMDGPU.device(nonzeros(sparse)) ||
        return UnsupportedSparseVendor("rocSPARSE requires the operands' device to be current")
    Ti=SparseArrays.indtype(sparse)
    Ti in (Int32,Int64) || return UnsupportedSparseVendor("rocSPARSE requires Int32 or Int64 indices")
    p=sparse isa ROCSparseMatrixBSR ? sparse.rowPtr : first(GPUArrays.sparse_buffers(sparse))
    eltype(p)===Ti || return UnsupportedSparseVendor("this rocSPARSE executor requires uniform index widths")
    if sparse isa ROCSparseMatrixBSR
        Ti===Int32 && all(iszero(d%sparse.blockDim) for d in size(sparse)) ||
            return UnsupportedSparseVendor("BSR execution requires Int32 indices and block-aligned dimensions")
    end
    for X in (A,B,C)
        X isa ROCSparseMatrix && continue
        stride(X,1)==1 && (ndims(X)==1 || stride(X,2)>=max(1,size(X,1))) ||
            return UnsupportedSparseVendor("rocSPARSE requires contiguous vectors and column-major matrices")
    end
    nothing
end
function GPUArrays.normalize_sparse_product_operand(A::ROCSparseMatrixBSR,t,algorithm)
    t in ('S','s','H','h') || return (A,t)
    algorithm isa GPUArrays.Portable && throw(ArgumentError("portable BSR products are not implemented"))
    (GPUArrays.materialize(LinearAlgebra.wrap(ROCSparseMatrixCSR(A),t)),'N')
end
function GPUArrays.vendor_spmv(y::ROCVector,t,A::ROCSparseMatrix,x::ROCVector,alpha,beta)
    reason=capability(y,A,x)
    reason===nothing || return reason
    t in ('N','T','C') || return UnsupportedSparseVendor("rocSPARSE requires an ordinary operand")
    A isa ROCSparseMatrixBSR && t!='N' &&
        return UnsupportedSparseVendor("rocSPARSE BSR SpMV requires a nontransposed matrix")
    ROCSpMV()
end
GPUArrays.execute_spmv!(::ROCSpMV,y,t,A,x,alpha,beta)=SP.mv!(Char(t),alpha,A,x,beta,y,'O')
function spmm_capability(C,ta,tb,A,B)
    reason=capability(C,A,B)
    reason===nothing || return reason
    ta in ('N','T','C') && tb in ('N','T','C') ||
        return UnsupportedSparseVendor("rocSPARSE requires ordinary operands")
    eltype(C)<:Complex && ((A isa ROCSparseMatrixCSC && ta=='C') ||
                          (B isa ROCSparseMatrixCSR && tb=='C')) &&
        return UnsupportedSparseVendor("rocSPARSE does not support this complex compressed adjoint SpMM")
    (A isa ROCSparseMatrixBSR && ta!='N') &&
        return UnsupportedSparseVendor("rocSPARSE BSR SpMM requires a nontransposed sparse matrix")
    ROCSpMM()
end
GPUArrays.vendor_spmm(C::ROCMatrix,ta,tb,A::ROCSparseMatrix,B::ROCMatrix,alpha,beta)=
    spmm_capability(C,ta,tb,A,B)
GPUArrays.vendor_spmm(C::ROCMatrix,ta,tb,A::ROCMatrix,B::ROCCanonicalMatrix,alpha,beta)=
    spmm_capability(C,ta,tb,A,B)
GPUArrays.execute_spmm!(::ROCSpMM,C,ta,tb,A,B,alpha,beta)=
    SP.mm!(Char(ta),Char(tb),alpha,A,B,beta,C,'O')

function GPUArrays.vendor_spgemm(A::ROCCanonicalMatrix,B::ROCCanonicalMatrix)
    eltype(A)===eltype(B) && eltype(A)<:LinearAlgebra.BlasFloat &&
        SparseArrays.indtype(A)===SparseArrays.indtype(B)===Int32 ||
        return UnsupportedSparseVendor("this rocSPARSE SpGEMM executor requires matching BLAS values and Int32 indices")
    for X in (A,B)
        eltype(first(GPUArrays.sparse_buffers(X)))===Int32 ||
            return UnsupportedSparseVendor("rocSPARSE SpGEMM requires uniform indices")
    end
    AMDGPU.functional(:rocsparse) || return UnsupportedSparseVendor("rocSPARSE is unavailable")
    AMDGPU.device()==AMDGPU.device(nonzeros(A)) ||
        return UnsupportedSparseVendor("rocSPARSE requires the operands' device to be current")
    ROCSpGEMM()
end
function GPUArrays.execute_spgemm(::ROCSpGEMM,A,B;max_temp_bytes)
    Acsr=ROCSparseMatrixCSR{eltype(A),Int32}(A)
    Bcsr=ROCSparseMatrixCSR{eltype(B),Int32}(B)
    C=SP.gemm('N','N',one(eltype(A)),Acsr,Bcsr,'O')
    convert(GPUArrays.matrix_format(A),C)
end

# Backend-only formats normalize once, then use centrally owned arithmetic.
for op in (:+,:-)
    @eval begin
        Base.$op(A::ROCSparseMatrixBSR,B::GPUArrays.GPUSparseMatrix)=Base.$op(ROCSparseMatrixCSR(A),B)
        Base.$op(A::GPUArrays.GPUSparseMatrix,B::ROCSparseMatrixBSR)=Base.$op(A,ROCSparseMatrixCSR(B))
        Base.$op(A::ROCSparseMatrixBSR,B::ROCSparseMatrixBSR)=Base.$op(ROCSparseMatrixCSR(A),ROCSparseMatrixCSR(B))
    end
end

include("ROCSPARSEExt/solves.jl")
end
