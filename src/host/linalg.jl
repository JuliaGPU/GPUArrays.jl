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

function Base.copyto!(A::AbstractGPUArray, B::Adjoint{T, <: AbstractGPUArray}) where T
    adjoint!(A, B.parent)
end

function Base.copyto!(A::AbstractGPUArray, B::Transpose{T, <: AbstractGPUArray}) where T
    transpose!(A, B.parent)
end

function Base.copyto!(A::AbstractArray, B::Adjoint{<:Any, <:AbstractGPUArray})
    copyto!(A, Adjoint(Array(parent(B))))
end
function Base.copyto!(A::AbstractArray, B::Transpose{<:Any, <:AbstractGPUArray})
    copyto!(A, Transpose(Array(parent(B))))
end


## triangular

function Base.copyto!(A::AbstractArray, B::UpperTriangular{<:Any, <:AbstractGPUArray})
    copyto!(A, UpperTriangular(Array(parent(B))))
end
function Base.copyto!(A::AbstractArray, B::LowerTriangular{<:Any, <:AbstractGPUArray})
    copyto!(A, LowerTriangular(Array(parent(B))))
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
    A = reshape(A, (size(A,1), size(A,2)))
    B = reshape(B, (size(B,1), size(B,2)))
    C = reshape(C, (size(C,1), size(C,2)))

    gpu_call(C, A, B) do ctx, C, A, B
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
