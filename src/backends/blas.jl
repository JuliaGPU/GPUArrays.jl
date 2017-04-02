@compat const AccVector{T} = AbstractAccArray{T, 1}
@compat const AccVecOrMat{T} = Union{AbstractAccArray{T, 1}, AbstractAccArray{T, 2}}

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
                DX::AbstractAccArray{$elty, N}, incx::Integer
            )
            ctx = context(A)
            blasmod = blas_module(ctx)
            blasmod.scal!(n, DA, blasbuffer(ctx, DX), incx)
            DX
        end
    end
end

for elty in T in (Float32, Float64, Complex64, Complex128)
    @eval begin
        function Base.BLAS.gemv!(trans::Char, alpha::($elty), A::AccVecOrMat{$elty}, X::AccVector{$elty}, beta::($elty), Y::AccVector{$elty})
            m,n = size(A,1),size(A,2)
            if trans == 'N' && (length(X) != n || length(Y) != m)
                throw(DimensionMismatch("A has dimensions $(size(A)), X has length $(length(X)) and Y has length $(length(Y))"))
            elseif trans == 'C' && (length(X) != m || length(Y) != n)
                throw(DimensionMismatch("A' has dimensions $n, $m, X has length $(length(X)) and Y has length $(length(Y))"))
            elseif trans == 'T' && (length(X) != m || length(Y) != n)
                throw(DimensionMismatch("A.' has dimensions $n, $m, X has length $(length(X)) and Y has length $(length(Y))"))
            end
            ctx = context(A)
            blasmod = blas_module(ctx)
            blasmod.gemv!(trans, alpha, A, X, beta, Y)
            Y
        end
    end
end
