

# all backends need to define a blas_module function to map to the correct library
blas_module(A::AccVecOrMat) = blas_module(context(A))

# Interface that needs to be overwritten by backend
# Slightly difference behavior from buffer, since not all blas backends work directly with
# the gpu array buffer
function blasbuffer(ctx, A)
    error("$ctx doesn't support BLAS operations with $(typeof(A))")
end
for T in (Float32, Float64, Complex64, Complex128)
    @eval begin
        function Base.BLAS.gemm!(
                transA::Char, transB::Char, alpha::$T,
                A::AccVecOrMat{$T}, B::AccVecOrMat{$T},
                beta::$T, C::AccVecOrMat{$T}
            )
            ctx = context(A)
            blasmod = blas_module(ctx)
            if transA == 'T' && is_opencl(ctx)
                transA = 'N'
                A = A'
            end
            if transB == 'T' && is_opencl(ctx)
                transB = 'N'
                B = B'
            end
            result = blasmod.gemm!(
                transA, transB, alpha,
                blasbuffer(ctx, A), blasbuffer(ctx, B), beta, blasbuffer(ctx, C)
            )
            C
        end
    end
end

for elty in (Float64, Float32)
    @eval begin
        function Base.BLAS.scal!{N}(
                n::Integer, DA::$elty,
                DX::GPUArray{$elty, N}, incx::Integer
            )
            ctx = context(DX)
            blasmod = blas_module(ctx)
            blasmod.scal!(n, DA, blasbuffer(ctx, DX), incx)
            DX
        end
    end
end

function Base.scale!(X::GPUArray{T}, s::Real) where T <: BLAS.BlasComplex
    R = typeof(real(zero(T)))
    buff = reinterpret(R, vec(X))
    BLAS.scal!(2*length(X), R(s), buff, 1)
    X
end
function Base.scale!(X::GPUArray{T}, s::Real) where T <: Union{Float32, Float64}
    BLAS.scal!(length(X), T(s), X, 1)
    X
end


for elty in (Float32, Float64, Complex64, Complex128)
    @eval begin
        function Base.BLAS.gemv!(trans::Char, alpha::($elty), A::AccVecOrMat{$elty}, X::GPUVector{$elty}, beta::($elty), Y::GPUVector{$elty})
            m, n = size(A, 1), size(A, 2)
            if trans == 'N' && (length(X) != n || length(Y) != m)
                throw(DimensionMismatch("A has dimensions $(size(A)), X has length $(length(X)) and Y has length $(length(Y))"))
            elseif trans == 'C' && (length(X) != m || length(Y) != n)
                throw(DimensionMismatch("A' has dimensions $n, $m, X has length $(length(X)) and Y has length $(length(Y))"))
            elseif trans == 'T' && (length(X) != m || length(Y) != n)
                throw(DimensionMismatch("A.' has dimensions $n, $m, X has length $(length(X)) and Y has length $(length(Y))"))
            end
            ctx = context(A)
            blasmod = blas_module(ctx)
            blasmod.gemv!(
                trans, alpha,
                blasbuffer(ctx, A), blasbuffer(ctx, X), beta, blasbuffer(ctx, Y)
            )
            Y
        end
    end
end
