abstract BLASImplementation

immutable CUBLASImpl <: BLASImplementation end
immutable CLBLASImpl <: BLASImplementation end
immutable JLBLASImpl <: BLASImplementation end

blas_module(A::AccVecOrMat) = blas_type(context(A))
blas_module(::CUContext) = CUBLAS
blas_module(::CLContext) = CLBLAS
blas_module(::JLContext) = BLAS
function blas_module(A::GLContext)
    if blas_supported(CLContext)
        CUBLAS
    elseif blas_supported(CUContext)
        CUBLAS
    else
        BLAS # performance error ?!
    end
end

typealias AccVecOrMat{T} Union{AbstractAccArray{T, 1}, AbstractAccArray{T, 2}}

# Interface that needs to be overwritten by backend
# Slightly difference behavior from buffer, since not all blas backends work directly with
# the gpu array buffer
function blasbuffer(ctx, A)
    error("$ctx doesn't support BLAS operations with $A")
end

function Base.BLAS.gemm!{T <: Number}(
        transA::Char, transB::Char, alpha::T,
        A::AccVecOrMat{T}, B::AccVecOrMat{T},
        beta::T, C::AccVecOrMat{T}
    )
    ctx = context(A)
    blasmod = blas_module(ctx)
    result = blasmod.gemm!(
        transA, transB, alpha,
        blasbuffer(ctx, A), blasbuffer(ctx, B), beta, blasbuffer(ctx, C)
    )
    convert(typeof(A), result)
end
