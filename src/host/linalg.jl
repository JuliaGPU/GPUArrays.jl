# integration with LinearAlgebra stdlib

function LinearAlgebra.transpose!(At::AbstractGPUArray{T, 2}, A::AbstractGPUArray{T, 2}) where T
    gpu_call(At, A) do ctx, At, A
        idx = @cartesianidx A ctx
        @inbounds At[idx[2], idx[1]] = A[idx[1], idx[2]]
        return
    end
    At
end

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

function LinearAlgebra.tril!(A::AbstractGPUMatrix{T}, d::Integer = 0) where T
  function kernel!(ctx, _A, _d)
    I = @cartesianidx _A
    i, j = Tuple(I)
    if i < j - _d
      _A[i, j] = 0
    end
    return nothing
  end

  gpu_call(kernel!, A, d)
  return A
end

function LinearAlgebra.triu!(A::AbstractGPUMatrix{T}, d::Integer = 0) where T
  function kernel!(ctx, _A, _d)
    I = @cartesianidx _A
    i, j = Tuple(I)
    if j < i + _d
      _A[i, j] = 0
    end
    return nothing
  end

  gpu_call(kernel!, A, d)
  return A
end

function LinearAlgebra.copy_transpose!(dst::AbstractGPUArray, src::AbstractGPUArray)
  function kernel(ctx, dst, src)
    I = @cartesianidx dst
    dst[I...] = src[reverse(I)...]
    return
  end

  gpu_call(kernel, dst, src)
  return dst
end


# matrix multiplication

function generic_matmatmul!(C::AbstractVecOrMat{R}, A::AbstractVecOrMat{T}, B::AbstractVecOrMat{S}) where {T,S,R}
    if size(A,2) != size(B,1)
        throw(DimensionMismatch("matrix A has dimensions $(size(A)), matrix B has dimensions $(size(B))"))
    end
    if size(C,1) != size(A,1) || size(C,2) != size(B,2)
        throw(DimensionMismatch("result C has dimensions $(size(C)), needs $((size(A,1),size(B,2)))"))
    end
    if isempty(A) || isempty(B)
        return fill!(C, zero(R))
    end

    function kernel(ctx, C, A, B)
        idx = @linearidx C
        i, j = Tuple(CartesianIndices(C)[idx])

        if i <= size(A,1) && j <= size(B,2)
            z2 = zero(A[i, 1]*B[1, j] + A[i, 1]*B[1, j])
            Ctmp = convert(promote_type(R, typeof(z2)), z2)
            for k in 1:size(A,2)
                Ctmp += A[i, k]*B[k, j]
            end
            C[i,j] = Ctmp
        end

        return
    end

    gpu_call(kernel, C, A, B)

    C
end

LinearAlgebra.mul!(C::AbstractGPUVecOrMat, A::AbstractGPUVecOrMat, B::AbstractGPUVecOrMat) = generic_matmatmul!(C, A, B)
LinearAlgebra.mul!(C::AbstractGPUVecOrMat, A::AbstractGPUVecOrMat, B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}) = generic_matmatmul!(C, A, B)
LinearAlgebra.mul!(C::AbstractGPUVecOrMat, A::AbstractGPUVecOrMat, B::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat}) = generic_matmatmul!(C, A, B)
LinearAlgebra.mul!(C::AbstractGPUVecOrMat, A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}, B::AbstractGPUVecOrMat) = generic_matmatmul!(C, A, B)
LinearAlgebra.mul!(C::AbstractGPUVecOrMat, A::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat}, B::AbstractGPUVecOrMat) = generic_matmatmul!(C, A, B)
LinearAlgebra.mul!(C::AbstractGPUVecOrMat, A::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat}, B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}) = generic_matmatmul!(C, A, B)
LinearAlgebra.mul!(C::AbstractGPUVecOrMat, A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}, B::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat}) = generic_matmatmul!(C, A, B)
LinearAlgebra.mul!(C::AbstractGPUVecOrMat, A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}, B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}) = generic_matmatmul!(C, A, B)
LinearAlgebra.mul!(C::AbstractGPUVecOrMat, A::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat}, B::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat}) = generic_matmatmul!(C, A, B)

function generic_rmul!(X::AbstractGPUArray, s::Number)
    function kernel(ctx, X, s)
        i = @linearidx X
        @inbounds X[i] *= s
        return
    end
    gpu_call(kernel, X, s)
    X
end

LinearAlgebra.rmul!(A::AbstractGPUArray, b::Number) = generic_rmul!(A, b)

function generic_lmul!(s::Number, X::AbstractGPUArray)
    function kernel(ctx, X, s)
        i = @linearidx X
        @inbounds X[i] = s*X[i]
        return
    end
    gpu_call(kernel, X, s)
    X
end

LinearAlgebra.lmul!(a::Number, B::AbstractGPUArray) = generic_lmul!(a, B)
