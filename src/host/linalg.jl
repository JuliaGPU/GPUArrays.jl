# integration with LinearAlgebra stdlib

## transpose and adjoint

function transpose_f!(f, At::AbstractGPUArray{T, 2}, A::AbstractGPUArray{T, 2}) where T
    gpu_call(At, A) do ctx, At, A
        idx = @cartesianidx A
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

function Base.copyto!(A::Array{T,N}, B::Adjoint{T, <:AbstractGPUArray{T,N}}) where {T,N}
    copyto!(A, Adjoint(Array(parent(B))))
end
function Base.copyto!(A::Array{T,N}, B::Transpose{T, <:AbstractGPUArray{T,N}}) where {T,N}
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
Base.copyto!(A::Array{T,N}, B::UpperTriangular{T, <:AbstractGPUArray{T,N}}) where {T,N} = copyto!(A, UpperTriangular(Array(parent(B))))
Base.copyto!(A::Array{T,N}, B::LowerTriangular{T, <:AbstractGPUArray{T,N}}) where {T,N} = copyto!(A, LowerTriangular(Array(parent(B))))

# GPU/GPU: B -> A
Base.copyto!(A::AbstractGPUArray{T,N}, B::UpperTriangular{T, <:AbstractGPUArray{T,N}}) where {T,N} = LinearAlgebra.triu!(copyto!(A, parent(B)))
Base.copyto!(A::AbstractGPUArray{T,N}, B::LowerTriangular{T, <:AbstractGPUArray{T,N}}) where {T,N} = LinearAlgebra.tril!(copyto!(A, parent(B)))
for T in (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular)
    @eval Base.copyto!(A::$T{T, <:AbstractGPUArray{T,N}}, B::$T{T, <:AbstractGPUArray{T,N}}) where {T,N} = $T(copyto!(parent(A), parent(B)))
end

function LinearAlgebra.tril!(A::AbstractGPUMatrix{T}, d::Integer = 0) where T
  gpu_call(A, d; name="tril!") do ctx, _A, _d
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
  gpu_call(A, d; name="triu!") do ctx, _A, _d
    I = @cartesianidx _A
    i, j = Tuple(I)
    if j < i + _d
      _A[i, j] = 0
    end
    return
  end
  return A
end


## diagonal

Base.copy(D::Diagonal{T, <:AbstractGPUArray{T, N}}) where {T, N} = Diagonal(copy(D.diag))

if VERSION <= v"1.8-"
    function LinearAlgebra.cholesky!(D::Diagonal{<:Any, <:AbstractGPUArray},
                                     ::Val{false} = Val(false); check::Bool = true)
        info = 0
        if mapreduce(x -> isreal(x) && isposdef(x), &, D.diag)
            D.diag .= sqrt.(D.diag)
        else
            info = findfirst(x -> !isreal(x) || !isposdef(x), collect(D.diag))
            check && throw(PosDefException(info))
        end
        Cholesky(D, 'U', convert(LinearAlgebra.BlasInt, info))
    end
else
    function LinearAlgebra.cholesky!(D::Diagonal{<:Any, <:AbstractGPUArray},
                                    ::NoPivot = NoPivot(); check::Bool = true)
        info = 0
        if mapreduce(x -> isreal(x) && isposdef(x), &, D.diag)
            D.diag .= sqrt.(D.diag)
        else
            info = findfirst(x -> !isreal(x) || !isposdef(x), collect(D.diag))
            check && throw(PosDefException(info))
        end
        Cholesky(D, 'U', convert(LinearAlgebra.BlasInt, info))
    end
end

Base.:\(D::Diagonal, b::AbstractGPUArray{<:Any, 1}) = cu(D.diag) .\ b

function LinearAlgebra.ldiv!(D::Diagonal{<:Any, <:AbstractGPUArray}, B::AbstractGPUVecOrMat)
    m, n = size(B, 1), size(B, 2)
    if m != length(D.diag)
        throw(DimensionMismatch("diagonal matrix is $(length(D.diag)) by $(length(D.diag)) but right hand side has $m rows"))
    end
    (m == 0 || n == 0) && return B
    z = D.diag .== 0
    if any(z)
        i = findfirst(z)
        throw(SingularException(i))
    else
        _ldiv_inner!(D, B)
    end
    return B
end

LinearAlgebra.ldiv!(D::Diagonal{<:Any, <:Any}, B::AbstractGPUArray) = LinearAlgebra.ldiv!(cu(D), B)

function _ldiv_inner!(D, B::AbstractGPUArray)
    B .= D.diag .\ B
    return B
end

function _ldiv_inner!(D, B)
    B .= collect(D.diag) .\ B
    return B
end

## matrix multiplication

function generic_matmatmul!(C::AbstractArray{R}, A::AbstractArray{T}, B::AbstractArray{S}, a::Number, b::Number) where {T,S,R}
    if size(A,2) != size(B,1)
        throw(DimensionMismatch("matrix A has dimensions $(size(A)), matrix B has dimensions $(size(B))"))
    end
    if size(C,1) != size(A,1) || size(C,2) != size(B,2)
        throw(DimensionMismatch("result C has dimensions $(size(C)), needs $((size(A,1),size(B,2)))"))
    end
    if isempty(A) || isempty(B)
        return fill!(C, zero(R))
    end

    gpu_call(C, A, B; name="matmatmul!") do ctx, C, A, B
        idx = @linearidx C
        i, j = @inbounds Tuple(CartesianIndices(C)[idx])..., 1

        @inbounds if i <= size(A,1) && j <= size(B,2)
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

# specificity hacks
LinearAlgebra.mul!(C::AbstractGPUVecOrMat, A::AbstractGPUVecOrMat, B::AbstractGPUVecOrMat, a::Real, b::Real) = generic_matmatmul!(C, A, B, a, b)
LinearAlgebra.mul!(C::AbstractGPUVecOrMat, A::AbstractGPUVecOrMat, B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}, a::Real, b::Real) = generic_matmatmul!(C, A, B, a, b)
LinearAlgebra.mul!(C::AbstractGPUVecOrMat, A::AbstractGPUVecOrMat, B::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat}, a::Real, b::Real) = generic_matmatmul!(C, A, B, a, b)
LinearAlgebra.mul!(C::AbstractGPUVecOrMat, A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}, B::AbstractGPUVecOrMat, a::Real, b::Real) = generic_matmatmul!(C, A, B, a, b)
LinearAlgebra.mul!(C::AbstractGPUVecOrMat, A::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat}, B::AbstractGPUVecOrMat, a::Real, b::Real) = generic_matmatmul!(C, A, B, a, b)
LinearAlgebra.mul!(C::AbstractGPUVecOrMat, A::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat}, B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}, a::Real, b::Real) = generic_matmatmul!(C, A, B, a, b)
LinearAlgebra.mul!(C::AbstractGPUVecOrMat, A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}, B::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat}, a::Real, b::Real) = generic_matmatmul!(C, A, B, a, b)
LinearAlgebra.mul!(C::AbstractGPUVecOrMat, A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}, B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}, a::Real, b::Real) = generic_matmatmul!(C, A, B, a, b)
LinearAlgebra.mul!(C::AbstractGPUVecOrMat, A::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat}, B::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat}, a::Real, b::Real) = generic_matmatmul!(C, A, B, a, b)


function generic_rmul!(X::AbstractArray, s::Number)
    gpu_call(X, s; name="rmul!") do ctx, X, s
        i = @linearidx X
        @inbounds X[i] *= s
        return
    end
    return X
end

LinearAlgebra.rmul!(A::AbstractGPUArray, b::Number) = generic_rmul!(A, b)

function generic_lmul!(s::Number, X::AbstractArray)
    gpu_call(X, s; name="lmul!") do ctx, X, s
        i = @linearidx X
        @inbounds X[i] = s*X[i]
        return
    end
    return X
end

LinearAlgebra.lmul!(a::Number, B::AbstractGPUArray) = generic_lmul!(a, B)


## permutedims
LinearAlgebra.permutedims!(dest::AbstractGPUArray, src::AbstractGPUArray, perm) =
    permutedims!(dest, src, Tuple(perm))

@inline @generated function permute_linearindex(size::NTuple{N,T}, l::T, strides::NTuple{N,T}) where {N,T}
    quote
        l -= one(T)
        res = one(T)
        Base.Cartesian.@nexprs $(N-1) i->begin
            assume(size[i] > 0)
            @inbounds l, s = divrem(l, size[i])
            @inbounds res += s * strides[i]
        end
        return @inbounds res + strides[N] * l
    end
end
function LinearAlgebra.permutedims!(dest::AbstractGPUArray, src::AbstractGPUArray,
                                    perm::NTuple{N}) where N
    length(dest) <= typemax(UInt32) ? _permutedims!(UInt32, dest, src, perm) : _permutedims!(UInt64, dest, src, perm)
end

function _permutedims!(::Type{IT}, dest::AbstractGPUArray, src::AbstractGPUArray,
                                    perm::NTuple{N}) where {IT,N}
    @assert length(src) <= typemax(IT)
    Base.checkdims_perm(dest, src, perm)
    dest_strides = ntuple(k->k==1 ? 1 : prod(i->size(dest, i), 1:k-1), N)
    dest_strides_perm = ntuple(i->IT(dest_strides[findfirst(==(i), perm)]), N)
    size_src = IT.(size(src))
    function permutedims_kernel(ctx, dest, src, size_src, dest_strides_perm)
        SLI = @linearidx dest
        assume(0 < SLI <= typemax(IT))
        LI = IT(SLI)
        dest_index = permute_linearindex(size_src, LI, dest_strides_perm)
        @inbounds dest[dest_index] = src[LI]
        return
    end
    gpu_call(permutedims_kernel, vec(dest), vec(src), size_src, dest_strides_perm)
    return dest
end

## norm

function LinearAlgebra.norm(v::AbstractGPUArray{T}, p::Real=2) where {T}
    norm_x = if p == Inf
        maximum(abs.(v))
    elseif p == -Inf
        minimum(abs.(v))
    else
        mapreduce(x->abs(x)^p, +, v; init=float(zero(T)))^(1/p)
    end
    return real(norm_x)
end


## symmetric

# prevent scalar indexing (upstream? this version is slower than a simple loop)
function Base.similar(A::Hermitian{<:Any,<:AbstractGPUArray}, ::Type{T}) where T
    B = similar(parent(A), T)
    fill!(view(B, diagind(B)), 0)
    return Hermitian(B, ifelse(A.uplo == 'U', :U, :L))
end
