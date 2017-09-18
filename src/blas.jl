
# Interface that needs to be overwritten by backend
# Slightly difference behavior from buffer, since not all blas backends work directly with
# the gpu array buffer
function blas_module(A)
    error("$(typeof(A)) doesn't support BLAS operations")
end
function blasbuffer(A)
    error("$(typeof(A)) doesn't support BLAS operations")
end

for T in (Float32, Float64, Complex64, Complex128)
    @eval begin
        function Base.BLAS.gemm!(
                transA::Char, transB::Char, alpha::$T,
                A::GPUVecOrMat{$T}, B::GPUVecOrMat{$T},
                beta::$T, C::GPUVecOrMat{$T}
            )
            blasmod = blas_module(A)
            result = blasmod.gemm!(
                transA, transB, alpha,
                blasbuffer(A), blasbuffer(B), beta, blasbuffer(C)
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
            blasmod = blas_module(DX)
            blasmod.scal!(n, DA, blasbuffer(DX), incx)
            DX
        end
    end
end

Base.scale!(s::Real, X::GPUArray) = scale!(X, s)
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
        function Base.BLAS.gemv!(trans::Char, alpha::($elty), A::GPUVecOrMat{$elty}, X::GPUVector{$elty}, beta::($elty), Y::GPUVector{$elty})
            m, n = size(A, 1), size(A, 2)
            if trans == 'N' && (length(X) != n || length(Y) != m)
                throw(DimensionMismatch("A has dimensions $(size(A)), X has length $(length(X)) and Y has length $(length(Y))"))
            elseif trans == 'C' && (length(X) != m || length(Y) != n)
                throw(DimensionMismatch("A' has dimensions $n, $m, X has length $(length(X)) and Y has length $(length(Y))"))
            elseif trans == 'T' && (length(X) != m || length(Y) != n)
                throw(DimensionMismatch("A.' has dimensions $n, $m, X has length $(length(X)) and Y has length $(length(Y))"))
            end
            blasmod = blas_module(A)
            blasmod.gemv!(
                trans, alpha,
                blasbuffer(A), blasbuffer(X), beta, blasbuffer(Y)
            )
            Y
        end
    end
end


for elty in (Float32, Float64, Complex64, Complex128)
    @eval begin
        function Base.BLAS.axpy!(
                alpha::Number, x::GPUArray{$elty}, y::GPUArray{$elty}
            )
            if length(x) != length(y)
                throw(DimensionMismatch("x has length $(length(x)), but y has length $(length(y))"))
            end
            blasmod = blas_module(x)
            blasmod.axpy!($elty(alpha), blasbuffer(vec(x)), blasbuffer(vec(y)))
            y
        end
    end
end
