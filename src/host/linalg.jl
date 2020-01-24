# integration with LinearAlgebra stdlib

## low-level BLAS calls

function blas_module(A)
    error("$(typeof(A)) doesn't support BLAS operations")
end
function blasbuffer(A)
    error("$(typeof(A)) doesn't support BLAS operations")
end

for elty in (Float32, Float64, ComplexF32, ComplexF64)
    T = VERSION >= v"1.3.0-alpha.115" ? :(Union{($elty), Bool}) : elty
    @eval begin
        function BLAS.gemm!(
                transA::AbstractChar, transB::AbstractChar, alpha::$T,
                A::AbstractGPUVecOrMat{$elty}, B::AbstractGPUVecOrMat{$elty},
                beta::$T, C::AbstractGPUVecOrMat{$elty}
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
        function BLAS.scal!(
                n::Integer, DA::$elty,
                DX::AbstractGPUArray{$elty, N}, incx::Integer
            ) where N
            blasmod = blas_module(DX)
            blasmod.scal!(n, DA, blasbuffer(DX), incx)
            DX
        end
    end
end

LinearAlgebra.rmul!(s::Number, X::AbstractGPUArray) = rmul!(X, s)
function LinearAlgebra.rmul!(X::AbstractGPUArray{T}, s::Number) where T <: BLAS.BlasComplex
    R = typeof(real(zero(T)))
    N = 2*length(X)
    buff = unsafe_reinterpret(R, X, (N,))
    BLAS.scal!(N, R(s), buff, 1)
    X
end
function LinearAlgebra.rmul!(X::AbstractGPUArray{T}, s::Number) where T <: Union{Float32, Float64}
    BLAS.scal!(length(X), T(s), X, 1)
    X
end

for elty in (Float32, Float64, ComplexF32, ComplexF64)
    T = VERSION >= v"1.3.0-alpha.115" ? :(Union{($elty), Bool}) : elty
    @eval begin
        function BLAS.gemv!(trans::AbstractChar, alpha::$T, A::AbstractGPUVecOrMat{$elty}, X::AbstractGPUVector{$elty}, beta::$T, Y::AbstractGPUVector{$elty})
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

for elty in (Float32, Float64, ComplexF32, ComplexF64)
    @eval begin
        function BLAS.axpy!(
                alpha::Number, x::AbstractGPUArray{$elty}, y::AbstractGPUArray{$elty}
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

for elty in (Float32, Float64, ComplexF32, ComplexF64)
    @eval begin
        function BLAS.gbmv!(trans::AbstractChar, m::Integer, kl::Integer, ku::Integer, alpha::($elty), A::AbstractGPUMatrix{$elty}, X::AbstractGPUVector{$elty}, beta::($elty), Y::AbstractGPUVector{$elty})
            n = size(A, 2)
            if trans == 'N' && (length(X) != n || length(Y) != m)
                throw(DimensionMismatch("A has dimensions $n, $m, X has length $(length(X)) and Y has length $(length(Y))"))
            elseif trans == 'C' && (length(X) != m || length(Y) != n)
                throw(DimensionMismatch("A' has dimensions $n, $m, X has length $(length(X)) and Y has length $(length(Y))"))
            elseif trans == 'T' && (length(X) != m || length(Y) != n)
                throw(DimensionMismatch("A.' has dimensions $n, $m, X has length $(length(X)) and Y has length $(length(Y))"))
            end
            blasmod = blas_module(A)
            blasmod.gbmv!(
                trans, m, kl, ku, alpha,
                blasbuffer(A), blasbuffer(X), beta, blasbuffer(Y)
            )
            Y
        end
    end
end


## high-level functionality

function LinearAlgebra.transpose!(At::AbstractGPUArray{T, 2}, A::AbstractGPUArray{T, 2}) where T
    gpu_call(At, (At, A)) do state, At, A
        idx = @cartesianidx A state
        @inbounds At[idx[2], idx[1]] = A[idx[1], idx[2]]
        return
    end
    At
end

function genperm(I::NTuple{N}, perm::NTuple{N}) where N
    ntuple(d-> (@inbounds return I[perm[d]]), Val(N))
end

function LinearAlgebra.permutedims!(dest::AbstractGPUArray, src::AbstractGPUArray, perm) where N
    perm isa Tuple || (perm = Tuple(perm))
    gpu_call(dest, (dest, src, perm)) do state, dest, src, perm
        I = @cartesianidx src state
        @inbounds dest[genperm(I, perm)...] = src[I...]
        return
    end
    return dest
end

function Base.copyto!(A::AbstractArray, B::Adjoint{<:Any, <:AbstractGPUArray})
    copyto!(A, Adjoint(Array(parent(B))))
end
function Base.copyto!(A::AbstractArray, B::Transpose{<:Any, <:AbstractGPUArray})
    copyto!(A, Transpose(Array(parent(B))))
end
function Base.copyto!(A::AbstractArray, B::UpperTriangular{<:Any, <:AbstractGPUArray})
    copyto!(A, UpperTriangular(Array(parent(B))))
end
function Base.copyto!(A::AbstractArray, B::LowerTriangular{<:Any, <:AbstractGPUArray})
    copyto!(A, LowerTriangular(Array(parent(B))))
end

function Base.copyto!(A::AbstractGPUArray, B::Adjoint{T, <: AbstractGPUArray}) where T
    transpose!(A, B.parent)
end
