# integration with LinearAlgebra stdlib

## transpose and adjoint

function transpose_f!(f, At::AbstractGPUArray{T, 2}, A::AbstractGPUArray{T, 2}) where T
    gpu_call(At, A) do ctx, At, A
        idx = @cartesianidx A ctx
        @inbounds At[idx[2], idx[1]] = f(A[idx[1], idx[2]])
        return
    end
    At
end

LinearAlgebra.transpose!(At::AbstractGPUArray, A::AbstractGPUArray) = transpose_f!(transpose, At, A)
LinearAlgebra.adjoint!(At::AbstractGPUArray, A::AbstractGPUArray) = transpose_f!(adjoint, At, A)

function Base.copyto!(A::AbstractGPUArray{T,N}, B::Adjoint{T, <: AbstractGPUArray{T,N}}) where {T,N}
    adjoint!(A, B.parent)
end

function Base.copyto!(A::AbstractGPUArray{T,N}, B::Transpose{T, <: AbstractGPUArray{T,N}}) where {T,N}
    transpose!(A, B.parent)
end

function Base.copyto!(A::AbstractArray{T,N}, B::Adjoint{T, <:AbstractGPUArray{T,N}}) where {T,N}
    copyto!(A, Adjoint(Array(parent(B))))
end
function Base.copyto!(A::AbstractArray{T,N}, B::Transpose{T, <:AbstractGPUArray{T,N}}) where {T,N}
    copyto!(A, Transpose(Array(parent(B))))
end


## copy upper triangle to lower and vice versa

function LinearAlgebra.copytri!(A::AbstractGPUMatrix{T}, uplo::AbstractChar) where T
  n = LinearAlgebra.checksquare(A)
  if uplo == 'U'
      gpu_call(A) do ctx, _A
        I = @cartesianidx _A
        i, j = Tuple(I)
        if j > i
          _A[j,i] = _A[i,j]
        end
        return
      end
  elseif uplo == 'L'
      gpu_call(A) do ctx, _A
        I = @cartesianidx _A
        i, j = Tuple(I)
        if j > i
          _A[i,j] = _A[j,i]
        end
        return
      end
  else
      throw(ArgumentError("uplo argument must be 'U' (upper) or 'L' (lower), got $uplo"))
  end
  A
end


## triangular

# mixed CPU/GPU: B -> A
Base.copyto!(A::AbstractArray{T,N}, B::UpperTriangular{T, <:AbstractGPUArray{T,N}}) where {T,N} = copyto!(A, UpperTriangular(Array(parent(B))))
Base.copyto!(A::AbstractArray{T,N}, B::LowerTriangular{T, <:AbstractGPUArray{T,N}}) where {T,N} = copyto!(A, LowerTriangular(Array(parent(B))))

# GPU/GPU: B -> A
Base.copyto!(A::AbstractGPUArray{T,N}, B::UpperTriangular{T, <:AbstractGPUArray{T,N}}) where {T,N} = LinearAlgebra.triu!(copyto!(A, parent(B)))
Base.copyto!(A::AbstractGPUArray{T,N}, B::LowerTriangular{T, <:AbstractGPUArray{T,N}}) where {T,N} = LinearAlgebra.tril!(copyto!(A, parent(B)))
for T in (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular)
    @eval Base.copyto!(A::$T{T, <:AbstractGPUArray{T,N}}, B::$T{T, <:AbstractGPUArray{T,N}}) where {T,N} = $T(copyto!(parent(A), parent(B)))
end

function LinearAlgebra.tril!(A::AbstractGPUMatrix{T}, d::Integer = 0) where T
  gpu_call(A, d) do ctx, _A, _d
    I = @cartesianidx _A
    i, j = Tuple(I)
    if i < j - _d
      _A[i, j] = 0
    end
    return
  end
  return A
end

function LinearAlgebra.triu!(A::AbstractGPUMatrix{T}, d::Integer = 0) where T
  gpu_call(A, d) do ctx, _A, _d
    I = @cartesianidx _A
    i, j = Tuple(I)
    if j < i + _d
      _A[i, j] = 0
    end
    return
  end
  return A
end


## matrix multiplication

function generic_matmatmul!(C::AbstractGPUVecOrMat{R}, A::AbstractGPUVecOrMat{T}, B::AbstractGPUVecOrMat{S}, a::Number, b::Number) where {T,S,R}
    if size(A,2) != size(B,1)
        throw(DimensionMismatch("matrix A has dimensions $(size(A)), matrix B has dimensions $(size(B))"))
    end
    if size(C,1) != size(A,1) || size(C,2) != size(B,2)
        throw(DimensionMismatch("result C has dimensions $(size(C)), needs $((size(A,1),size(B,2)))"))
    end
    if isempty(A) || isempty(B)
        return fill!(C, zero(R))
    end

    # reshape vectors to matrices
    A′ = reshape(A, (size(A,1), size(A,2)))
    B′ = reshape(B, (size(B,1), size(B,2)))
    C′= reshape(C, (size(C,1), size(C,2)))

    gpu_call(C′, A′, B′) do ctx, C, A, B
        idx = @linearidx C
        i, j = Tuple(CartesianIndices(C)[idx])

        if i <= size(A,1) && j <= size(B,2)
            z2 = zero(A[i, 1]*B[1, j] + A[i, 1]*B[1, j])
            Ctmp = convert(promote_type(R, typeof(z2)), z2)
            for k in 1:size(A,2)
                Ctmp += A[i, k]*B[k, j]
            end
            C[i,j] = Ctmp*a + C[i,j]*b
        end

        return
    end

    C
end

LinearAlgebra.mul!(C::AbstractGPUVecOrMat, A::AbstractGPUVecOrMat, B::AbstractGPUVecOrMat, a::Number, b::Number) = generic_matmatmul!(C, A, B, a, b)
LinearAlgebra.mul!(C::AbstractGPUVecOrMat, A::AbstractGPUVecOrMat, B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}, a::Number, b::Number) = generic_matmatmul!(C, A, B, a, b)
LinearAlgebra.mul!(C::AbstractGPUVecOrMat, A::AbstractGPUVecOrMat, B::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat}, a::Number, b::Number) = generic_matmatmul!(C, A, B, a, b)
LinearAlgebra.mul!(C::AbstractGPUVecOrMat, A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}, B::AbstractGPUVecOrMat, a::Number, b::Number) = generic_matmatmul!(C, A, B, a, b)
LinearAlgebra.mul!(C::AbstractGPUVecOrMat, A::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat}, B::AbstractGPUVecOrMat, a::Number, b::Number) = generic_matmatmul!(C, A, B, a, b)
LinearAlgebra.mul!(C::AbstractGPUVecOrMat, A::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat}, B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}, a::Number, b::Number) = generic_matmatmul!(C, A, B, a, b)
LinearAlgebra.mul!(C::AbstractGPUVecOrMat, A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}, B::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat}, a::Number, b::Number) = generic_matmatmul!(C, A, B, a, b)
LinearAlgebra.mul!(C::AbstractGPUVecOrMat, A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}, B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}, a::Number, b::Number) = generic_matmatmul!(C, A, B, a, b)
LinearAlgebra.mul!(C::AbstractGPUVecOrMat, A::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat}, B::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat}, a::Number, b::Number) = generic_matmatmul!(C, A, B, a, b)

@static if v"1.3.0" <= VERSION <= v"1.3.1"
    LinearAlgebra.mul!(C::AbstractGPUVecOrMat{T}, A::AbstractGPUVecOrMat{T}, B::AbstractGPUVecOrMat{T}, a::Union{Bool,T}, b::Union{Bool,T}) where {T<:LinearAlgebra.BLAS.BlasFloat} = generic_matmatmul!(C, A, B, a, b)
    LinearAlgebra.mul!(C::AbstractGPUVecOrMat{T}, A::AbstractGPUVecOrMat{T}, B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat{T}}, a::Union{Bool,T}, b::Union{Bool,T}) where {T<:LinearAlgebra.BLAS.BlasReal} = generic_matmatmul!(C, A, B, a, b)
    LinearAlgebra.mul!(C::AbstractGPUVecOrMat{T}, A::AbstractGPUVecOrMat{T}, B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat{T}}, a::Union{Bool,T}, b::Union{Bool,T}) where {T<:LinearAlgebra.BLAS.BlasComplex} = generic_matmatmul!(C, A, B, a, b)
    LinearAlgebra.mul!(C::AbstractGPUVecOrMat{T}, A::AbstractGPUVecOrMat{T}, B::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat{T}}, a::Union{Bool,T}, b::Union{Bool,T}) where {T<:LinearAlgebra.BLAS.BlasFloat} = generic_matmatmul!(C, A, B, a, b)
    LinearAlgebra.mul!(C::AbstractGPUVecOrMat{T}, A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat{T}}, B::AbstractGPUVecOrMat{T}, a::Union{Bool,T}, b::Union{Bool,T}) where {T<:LinearAlgebra.BLAS.BlasReal} = generic_matmatmul!(C, A, B, a, b)
    LinearAlgebra.mul!(C::AbstractGPUVecOrMat{T}, A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat{T}}, B::AbstractGPUVecOrMat{T}, a::Union{Bool,T}, b::Union{Bool,T}) where {T<:LinearAlgebra.BLAS.BlasComplex} = generic_matmatmul!(C, A, B, a, b)
    LinearAlgebra.mul!(C::AbstractGPUVecOrMat{T}, A::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat{T}}, B::AbstractGPUVecOrMat{T}, a::Union{Bool,T}, b::Union{Bool,T}) where {T<:LinearAlgebra.BLAS.BlasFloat} = generic_matmatmul!(C, A, B, a, b)
    LinearAlgebra.mul!(C::AbstractGPUVecOrMat{T}, A::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat{T}}, B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat{T}}, a::Union{Bool,T}, b::Union{Bool,T}) where {T<:LinearAlgebra.BLAS.BlasReal} = generic_matmatmul!(C, A, B, a, b)
    LinearAlgebra.mul!(C::AbstractGPUVecOrMat{T}, A::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat{T}}, B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat{T}}, a::Union{Bool,T}, b::Union{Bool,T}) where {T<:LinearAlgebra.BLAS.BlasComplex} = generic_matmatmul!(C, A, B, a, b)
    LinearAlgebra.mul!(C::AbstractGPUVecOrMat{T}, A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat{T}}, B::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat{T}}, a::Union{Bool,T}, b::Union{Bool,T}) where {T<:LinearAlgebra.BLAS.BlasReal} = generic_matmatmul!(C, A, B, a, b)
    LinearAlgebra.mul!(C::AbstractGPUVecOrMat{T}, A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat{T}}, B::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat{T}}, a::Union{Bool,T}, b::Union{Bool,T}) where {T<:LinearAlgebra.BLAS.BlasComplex} = generic_matmatmul!(C, A, B, a, b)
    LinearAlgebra.mul!(C::AbstractGPUVecOrMat{T}, A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat{T}}, B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat{T}}, a::Union{Bool,T}, b::Union{Bool,T}) where {T<:LinearAlgebra.BLAS.BlasReal} = generic_matmatmul!(C, A, B, a, b)
    LinearAlgebra.mul!(C::AbstractGPUVecOrMat{T}, A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat{T}}, B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat{T}}, a::Union{Bool,T}, b::Union{Bool,T}) where {T<:LinearAlgebra.BLAS.BlasComplex} = generic_matmatmul!(C, A, B, a, b)
    LinearAlgebra.mul!(C::AbstractGPUVecOrMat{T}, A::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat{T}}, B::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat{T}}, a::Union{Bool,T}, b::Union{Bool,T}) where {T<:LinearAlgebra.BLAS.BlasFloat} = generic_matmatmul!(C, A, B, a, b)
end


function generic_rmul!(X::AbstractGPUArray, s::Number)
    gpu_call(X, s) do ctx, X, s
        i = @linearidx X
        @inbounds X[i] *= s
        return
    end
    return X
end

LinearAlgebra.rmul!(A::AbstractGPUArray, b::Number) = generic_rmul!(A, b)

function generic_lmul!(s::Number, X::AbstractGPUArray)
    gpu_call(X, s) do ctx, X, s
        i = @linearidx X
        @inbounds X[i] = s*X[i]
        return
    end
    return X
end

LinearAlgebra.lmul!(a::Number, B::AbstractGPUArray) = generic_lmul!(a, B)


## permutedims

function genperm(I::CartesianIndex{N}, perm::NTuple{N}) where N
    CartesianIndex(ntuple(d-> (@inbounds return I[perm[d]]), Val(N)))
end

function LinearAlgebra.permutedims!(dest::AbstractGPUArray, src::AbstractGPUArray, perm) where N
    perm isa Tuple || (perm = Tuple(perm))
    gpu_call(dest, src, perm) do ctx, dest, src, perm
        I = @cartesianidx src ctx
        @inbounds dest[genperm(I, perm)] = src[I]
        return
    end
    return dest
end


## inv for Triangular
for TR in (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular)
    @eval function Base.inv(x::$TR{<:Any,<:AbstractGPUArray})
      out = typeof(parent(x))(I(size(x,1)))
      $TR(LinearAlgebra.ldiv!(x,out))
    end
end
