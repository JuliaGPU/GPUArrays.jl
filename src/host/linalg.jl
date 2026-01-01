# integration with LinearAlgebra stdlib

using LinearAlgebra: MulAddMul, wrap, diagm

## transpose and adjoint

function LinearAlgebra.transpose!(B::AbstractGPUVector, A::AbstractGPUMatrix)
    axes(B,1) == axes(A,2) && axes(A,1) == 1:1 || throw(DimensionMismatch("transpose"))
    copyto!(B, A)
end
function LinearAlgebra.transpose!(B::AbstractGPUMatrix, A::AbstractGPUVector)
    axes(B,2) == axes(A,1) && axes(B,1) == 1:1 || throw(DimensionMismatch("transpose"))
    copyto!(B, A)
end
function LinearAlgebra.adjoint!(B::AbstractGPUVector, A::AbstractGPUMatrix)
    axes(B,1) == axes(A,2) && axes(A,1) == 1:1 || throw(DimensionMismatch("adjoint"))
    isempty(A) && return B
    @kernel function adjoint_kernel!(B, A)
        idx = @index(Global, Linear)
        @inbounds B[idx] = adjoint(A[1, idx])
    end
    adjoint_kernel!(get_backend(B))(B, A; ndrange = size(B))
    B
end
function LinearAlgebra.adjoint!(B::AbstractGPUMatrix, A::AbstractGPUVector)
    axes(B,2) == axes(A,1) && axes(B,1) == 1:1 || throw(DimensionMismatch("adjoint"))
    isempty(A) && return B
    @kernel function adjoint_kernel!(B, A)
        idx = @index(Global, Linear)
        @inbounds B[1, idx] = adjoint(A[idx])
    end
    adjoint_kernel!(get_backend(A))(B, A; ndrange = size(A))
    B
end

LinearAlgebra.transpose!(B::AnyGPUArray, A::AnyGPUArray) = transpose_f!(transpose, B, A)
LinearAlgebra.adjoint!(B::AnyGPUArray, A::AnyGPUArray) = transpose_f!(adjoint, B, A)
function transpose_f!(f, B::AnyGPUMatrix{T}, A::AnyGPUMatrix{T}) where T
    axes(B,1) == axes(A,2) && axes(B,2) == axes(A,1) || throw(DimensionMismatch(string(f)))
    # array with size zero dimension
    isempty(A) && return B
    @kernel function transpose_kernel!(B, A)
        idx = @index(Global, Cartesian)
        @inbounds B[idx[2], idx[1]] = f(A[idx[1], idx[2]])
    end
    transpose_kernel!(get_backend(B))(B, A; ndrange = size(A))
    B
end

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

## diagm

LinearAlgebra.diagm(kv::Pair{<:Integer,<:AbstractGPUVector}...) = _gpu_diagm(nothing, kv...)
LinearAlgebra.diagm(m::Integer, n::Integer, kv::Pair{<:Integer,<:AbstractGPUVector}...) = _gpu_diagm((Int(m),Int(n)), kv...)
LinearAlgebra.diagm(v::AbstractGPUVector) = LinearAlgebra.diagm(0 => v)
LinearAlgebra.diagm(m::Integer, n::Integer, v::AbstractGPUVector) = LinearAlgebra.diagm(m, n, 0 => v)

function _gpu_diagm(size, kv::Pair{<:Integer,<:AbstractGPUVector}...)
    A = LinearAlgebra.diagm_container(size, kv...)
    for p in kv
        inds = LinearAlgebra.diagind(A, p.first)
        copyto!(view(A, inds), p.second)
    end
    return A
end

function LinearAlgebra.diagm_container(size, kv::Pair{<:Integer,<:AbstractGPUVector}...)
    T = promote_type(map(x -> eltype(x.second), kv)...)
    U = promote_type(T, typeof(zero(T)))
    A = similar(kv[1].second, U, LinearAlgebra.diagm_size(size, kv...)...)
    fill!(A, zero(U))
    return A
end

function LinearAlgebra.diagm_size(size::Nothing, kv::Pair{<:Integer,<:AbstractGPUVector}...)
    mnmax = mapreduce(x -> length(x.second) + abs(Int(x.first)), max, kv; init=0)
    return mnmax, mnmax
end
function LinearAlgebra.diagm_size(size::Tuple{Int,Int}, kv::Pair{<:Integer,<:AbstractGPUVector}...)
    mmax = mapreduce(x -> length(x.second) - min(0,Int(x.first)), max, kv; init=0)
    nmax = mapreduce(x -> length(x.second) + max(0,Int(x.first)), max, kv; init=0)
    m, n = size
    (m ≥ mmax && n ≥ nmax) || throw(DimensionMismatch(lazy"invalid size=$size"))
    return m, n
end

## trace

function LinearAlgebra.tr(A::AnyGPUMatrix)
    LinearAlgebra.checksquare(A)
    sum(diag(A))
end

## copy upper triangle to lower and vice versa

function LinearAlgebra.copytri!(A::AbstractGPUMatrix, uplo::AbstractChar, conjugate::Bool=false)
  n = LinearAlgebra.checksquare(A)
  if uplo == 'U' && conjugate
      @kernel function U_conj!(_A)
        I = @index(Global, Cartesian)
        i, j = Tuple(I)
        if j > i
          @inbounds _A[j,i] = conj(_A[i,j])
        end
      end
      U_conj!(get_backend(A))(A; ndrange = size(A))
  elseif uplo == 'U' && !conjugate
      @kernel function U_noconj!(_A)
        I = @index(Global, Cartesian)
        i, j = Tuple(I)
        if j > i
          @inbounds _A[j,i] = _A[i,j]
        end
      end
      U_noconj!(get_backend(A))(A; ndrange = size(A))
  elseif uplo == 'L' && conjugate
      @kernel function L_conj!(_A)
        I = @index(Global, Cartesian)
        i, j = Tuple(I)
        if j > i
          @inbounds _A[i,j] = conj(_A[j,i])
        end
      end
      L_conj!(get_backend(A))(A; ndrange = size(A))
  elseif uplo == 'L' && !conjugate
      @kernel function L_noconj!(_A)
        I = @index(Global, Cartesian)
        i, j = Tuple(I)
        if j > i
          @inbounds _A[i,j] = _A[j,i]
        end
      end
      L_noconj!(get_backend(A))(A; ndrange = size(A))
  else
      throw(ArgumentError("uplo argument must be 'U' (upper) or 'L' (lower), got $uplo"))
  end
  A
end

## copy a triangular part of a matrix to another matrix

if isdefined(LinearAlgebra, :copytrito!)
    function LinearAlgebra.copytrito!(B::AbstractGPUMatrix{T}, A::AbstractGPUMatrix{T}, uplo::AbstractChar) where {T}
        LinearAlgebra.BLAS.chkuplo(uplo)
        m,n = size(A)
        m1,n1 = size(B)
        if uplo == 'U'
            if n < m
                (m1 < n || n1 < n) && throw(DimensionMismatch("B of size ($m1,$n1) should have at least size ($n,$n)"))
            else
                (m1 < m || n1 < n) && throw(DimensionMismatch("B of size ($m1,$n1) should have at least size ($m,$n)"))
            end
            length(A) == 0 && return B
            @kernel function U_kernel!(_A, _B)
                I = @index(Global, Cartesian)
                i, j = Tuple(I)
                if j >= i
                    @inbounds _B[i,j] = _A[i,j]
                end
            end
            U_kernel!(get_backend(B))(A, B; ndrange = size(A))
        else  # uplo == 'L'
            if m < n
                (m1 < m || n1 < m) && throw(DimensionMismatch("B of size ($m1,$n1) should have at least size ($m,$m)"))
            else
                (m1 < m || n1 < n) && throw(DimensionMismatch("B of size ($m1,$n1) should have at least size ($m,$n)"))
            end
            length(A) == 0 && return B
            @kernel function L_kernel!(_A, _B)
                I = @index(Global, Cartesian)
                i, j = Tuple(I)
                if j <= i
                    @inbounds _B[i,j] = _A[i,j]
                end
            end
            L_kernel!(get_backend(A))(A, B; ndrange = size(A))
        end
        return B
    end
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
    isempty(A) && return A
    @kernel function tril_kernel!(_A, _d)
        I = @index(Global, Cartesian)
        i, j = Tuple(I)
        if i < j - _d
            @inbounds _A[i, j] = zero(T)
        end
    end
    tril_kernel!(get_backend(A))(A, d; ndrange = size(A))
    return A
end

function LinearAlgebra.triu!(A::AbstractGPUMatrix{T}, d::Integer = 0) where T
    isempty(A) && return A
    @kernel function triu_kernel!(_A, _d)
        I = @index(Global, Cartesian)
        i, j = Tuple(I)
        if j < i + _d
            @inbounds _A[i, j] = zero(T)
        end
    end
    triu_kernel!(get_backend(A))(A, d; ndrange = size(A))
    return A
end

# check if upper triangular starting from the kth superdiagonal.
function LinearAlgebra.istriu(A::AbstractGPUMatrix, k::Integer = 0)
    function mapper(a, I)
        row, col = Tuple(I)
        if col < row + k
            return iszero(a)
        else
            true
        end
    end
    function reducer(a, b)
        a && b
    end
    mapreduce(mapper, reducer, A, eachindex(IndexCartesian(), A); init=true)
end

# check if lower triangular starting from the kth subdiagonal.
function LinearAlgebra.istril(A::AbstractGPUMatrix, k::Integer = 0)
    function mapper(a, I)
        row, col = Tuple(I)
        if col > row + k
            return iszero(a)
        else
            true
        end
    end
    function reducer(a, b)
        a && b
    end
    mapreduce(mapper, reducer, A, eachindex(IndexCartesian(), A); init=true)
end


## diagonal

Base.copy(D::Diagonal{T, <:AbstractGPUArray{T, N}}) where {T, N} = Diagonal(copy(D.diag))

_isrealandpositive(x) = isreal(x) && real(x) > 0

function LinearAlgebra.cholesky!(D::Diagonal{<:Any, <:AbstractGPUArray},
                                ::NoPivot = NoPivot(); check::Bool = true)
    info = findfirst(!_isrealandpositive, D.diag)
    if isnothing(info)
        D.diag .= sqrt.(D.diag)
        info = 0
    elseif check
        throw(PosDefException(info))
    else
        D.diag[begin:info-1] .= sqrt.(D.diag[begin:info-1])
    end
    return Cholesky(D, 'U', convert(LinearAlgebra.BlasInt, info))
end

function Base.:\(D::Diagonal{<:Any, <:AbstractGPUArray}, B::AbstractGPUVecOrMat)
    z = D.diag .== 0
    if any(z)
        i = findfirst(collect(z))
        throw(SingularException(i))
    else
        return D.diag .\ B
    end
end

function LinearAlgebra.mul!(C::Diagonal{<:Any, <:AbstractGPUArray},
                            A::Diagonal{<:Any, <:AbstractGPUArray},
                            B::Diagonal{<:Any, <:AbstractGPUArray})
    dc = C.diag
    da = A.diag
    db = B.diag
    d = length(dc)
    length(da) == d || throw(DimensionMismatch("right hand side has $(length(da)) rows but output is $d by $d"))
    length(db) == d || throw(DimensionMismatch("left hand side has $(length(db)) rows but output is $d by $d"))
    @. dc = da * db

    return C
end

function LinearAlgebra.mul!(C::Diagonal{<:Any, <:AbstractGPUArray},
                            A::Diagonal{<:Any, <:AbstractGPUArray},
                            B::Diagonal{<:Any, <:AbstractGPUArray},
                            α::Number,
                            β::Number)
    dc = C.diag
    da = A.diag
    db = B.diag
    d = length(dc)
    length(da) == d || throw(DimensionMismatch("right hand side has $(length(da)) rows but output is $d by $d"))
    length(db) == d || throw(DimensionMismatch("left hand side has $(length(db)) rows but output is $d by $d"))
    @. dc = α * da * db + β * dc

    return C
end
function LinearAlgebra.mul!(C::Diagonal{<:Any, <:AbstractGPUArray},
                            A::Union{AbstractGPUArray, Adjoint{T,<:AbstractGPUArray{T}}, Transpose{T,<:AbstractGPUArray{T}}},
                            B::Union{AbstractGPUArray, Adjoint{T,<:AbstractGPUArray{T}}, Transpose{T,<:AbstractGPUArray{T}}}) where {T}
    dc = C.diag
    d  = length(dc)
    m, n   = size(A, 1), size(A, 2)
    m′, n′ = size(B, 1), size(B, 2)
    m == d  || throw(DimensionMismatch("left hand side has $m rows but output is $d by $d"))
    n′ == d || throw(DimensionMismatch("right hand side has $n′ cols but output is $d by $d"))
    C_ = A * B
    isdiag(C_) || throw(ErrorException("output matrix must be diagonal"))
    dc .= diag(C_)
    return C
end

function LinearAlgebra.mul!(C::Diagonal{<:Any, <:AbstractGPUArray},
                            A::Union{AbstractGPUArray, Adjoint{T,<:AbstractGPUArray{T}}, Transpose{T,<:AbstractGPUArray{T}}},
                            B::Union{AbstractGPUArray, Adjoint{T,<:AbstractGPUArray{T}}, Transpose{T,<:AbstractGPUArray{T}}},
                            α::Number,
                            β::Number) where {T}
    dc = C.diag
    d  = length(dc)
    m, n   = size(A, 1), size(A, 2)
    m′, n′ = size(B, 1), size(B, 2)
    m == d  || throw(DimensionMismatch("left hand side has $m rows but output is $d by $d"))
    n′ == d || throw(DimensionMismatch("right hand side has $n′ cols but output is $d by $d"))
    C_ = @. α * A * B + β * C
    isdiag(C_) || throw(ErrorException("output matrix must be diagonal"))
    dc .= diag(C_)
    return C
end

function LinearAlgebra.mul!(B::AbstractGPUVecOrMat,
                            D::Diagonal{<:Any, <:AbstractGPUArray},
                            A::Union{AbstractGPUArray, Adjoint{T,<:AbstractGPUArray{T}}, Transpose{T,<:AbstractGPUArray{T}}}) where {T}
    dd = D.diag
    d = length(dd)
    m, n = size(A, 1), size(A, 2)
    m′, n′ = size(B, 1), size(B, 2)
    m == d || throw(DimensionMismatch("right hand side has $m rows but D is $d by $d"))
    (m, n) == (m′, n′) || throw(DimensionMismatch("expect output to be $m by $n, but got $m′ by $n′"))
    @. B = dd * A
    B
end

function LinearAlgebra.mul!(B::AbstractGPUVecOrMat,
                            D::Diagonal{<:Any, <:AbstractGPUArray},
                            A::Union{AbstractGPUArray, Adjoint{T,<:AbstractGPUArray{T}}, Transpose{T,<:AbstractGPUArray{T}}},
                            α::Number,
                            β::Number) where {T}
    dd = D.diag
    d = length(dd)
    m, n = size(A, 1), size(A, 2)
    m′, n′ = size(B, 1), size(B, 2)
    m == d || throw(DimensionMismatch("right hand side has $m rows but D is $d by $d"))
    (m, n) == (m′, n′) || throw(DimensionMismatch("expect output to be $m by $n, but got $m′ by $n′"))
    @. B = α * dd * A + β * B

    B
end

function LinearAlgebra.mul!(B::AbstractGPUVecOrMat,
                            A::Union{AbstractGPUArray, Adjoint{T,<:AbstractGPUArray{T}}, Transpose{T,<:AbstractGPUArray{T}}},
                            D::Diagonal{<:Any, <:AbstractGPUArray}) where {T}
    dd = D.diag
    d = length(dd)
    m, n = size(A, 1), size(A, 2)
    m′, n′ = size(B, 1), size(B, 2)
    n == d || throw(DimensionMismatch("left hand side has $n columns but D is $d by $d"))
    (m, n) == (m′, n′) || throw(DimensionMismatch("expect output to be $m by $n, but got $m′ by $n′"))
    ddT = transpose(dd)
    @. B = A * ddT

    B
end

function LinearAlgebra.mul!(B::AbstractGPUVecOrMat,
                            A::Union{AbstractGPUArray, Adjoint{T,<:AbstractGPUArray{T}}, Transpose{T,<:AbstractGPUArray{T}}},
                            D::Diagonal{<:Any, <:AbstractGPUArray},
                            α::Number,
                            β::Number) where {T}
    dd = D.diag
    d = length(dd)
    m, n = size(A, 1), size(A, 2)
    m′, n′ = size(B, 1), size(B, 2)
    n == d || throw(DimensionMismatch("left hand side has $n columns but D is $d by $d"))
    (m, n) == (m′, n′) || throw(DimensionMismatch("expect output to be $m by $n, but got $m′ by $n′"))
    ddT = transpose(dd)
    @. B = α * A * ddT + β * B

    B
end

function LinearAlgebra.ldiv!(B::AbstractGPUVecOrMat,
                              D::Diagonal{<:Any, <:AbstractGPUArray},
                              A::AbstractGPUVecOrMat)
    dd = D.diag
    d = length(dd)
    m, n = size(A, 1), size(A, 2)
    m′, n′ = size(B, 1), size(B, 2)
    m == d || throw(DimensionMismatch("right hand side has $m rows but D is $d by $d"))
    (m, n) == (m′, n′) || throw(DimensionMismatch("expect output to be $m by $n, but got $m′ by $n′"))
    z = dd .== 0
    if any(z)
        i = findfirst(collect(z))
        throw(SingularException(i))
    else
        B .= dd .\ A
    end
    B
end

# XXX: figure out how to do dynamically
MAX_TILE_DIM = 16

## matrix multiplication
# legacy method
generic_matmatmul!(C::AbstractArray, A::AbstractArray, B::AbstractArray, a::Number, b::Number) =
    generic_matmatmul!(C, A, B, MulAddMul(a, b))
function generic_matmatmul!(C::AbstractGPUMatrix{R}, A::AbstractGPUMatrix{T}, B::AbstractGPUMatrix{S}, add::MulAddMul) where {T<:Number,S<:Number,R<:Number}
    N = size(A,1)
    Q = size(A,2)
    M = size(B,2)
    if Q != size(B,1)
        throw(DimensionMismatch("matrix A has dimensions $(size(A)), matrix B has dimensions $(size(B))"))
    end
    if size(C,1) != N || size(C,2) != M
        throw(DimensionMismatch("result C has dimensions $(size(C)), needs $((N,M))"))
    end
    if isempty(A) || isempty(B)
        return fill!(C, zero(R))
    end

    @kernel unsafe_indices=true function coalesced_matmul_kernel!(
            output, @Const(input1), @Const(input2), N, Q, M,
            ::Val{BANK} = Val(1),
        ) where {BANK}
        grow, gcol = @index(Group, NTuple)
        tile_row, tile_col = @index(Local, NTuple)

        TILE_DIM = @uniform @groupsize()[1]

        # +1 to avoid bank conflicts on shared memory
        tile1 = @localmem(R, (TILE_DIM + BANK, TILE_DIM))
        tile2 = @localmem(R, (TILE_DIM + BANK, TILE_DIM))

        # private variable for tile output
        outval = @private R 1
        @inbounds outval[1] = -zero(R)

        # number of tiles depends on inner dimension
        @uniform NUM_TILES = div(Q + TILE_DIM - 1, TILE_DIM)

        # loop over all tiles needed for this calculation
        for t in 0:(NUM_TILES - 1)
            I = (grow - 1) * TILE_DIM + tile_row
            J = (gcol - 1) * TILE_DIM + tile_col

            # load inputs into tiles, with bounds checking for non-square matrices
            if I <= N && t * TILE_DIM + tile_col <= Q
                @inbounds tile1[tile_row, tile_col] = input1[I, t * TILE_DIM + tile_col]
            else
                @inbounds tile1[tile_row, tile_col] = zero(R)
            end
            if J <= M && t * TILE_DIM + tile_row <= Q
                @inbounds tile2[tile_row, tile_col] = input2[t * TILE_DIM + tile_row, J]
            else
                @inbounds tile2[tile_row, tile_col] = zero(R)
            end

            # wait for all tiles to be loaded
            @synchronize

            I = (grow - 1) * TILE_DIM + tile_row
            J = (gcol - 1) * TILE_DIM + tile_col

            # calculate value of spot in output, use temporary value to allow for vectorization
            out = zero(R)
            @simd for k in 1:TILE_DIM
                @inbounds out += tile1[tile_row, k] * tile2[k, tile_col]
            end
            outval[1] += out

            @synchronize
        end

        I = (grow - 1) * TILE_DIM + tile_row
        J = (gcol - 1) * TILE_DIM + tile_col

        # save if inbounds
        if I <= N && J <= M
            @inbounds output[I, J] = add(outval[1], output[I, J])
        end
    end

    coalesced_matmul_kernel!(get_backend(C), (MAX_TILE_DIM, MAX_TILE_DIM))(C, A, B, N, Q, M;ndrange=map(x -> ceil(Int,x/MAX_TILE_DIM)*MAX_TILE_DIM, size(C)))
    C
end
function generic_matmatmul!(C::AbstractArray{R}, A::AbstractArray{T}, B::AbstractArray{S}, add::MulAddMul) where {T,S,R}
    if size(A,2) != size(B,1)
        throw(DimensionMismatch("matrix A has dimensions $(size(A)), matrix B has dimensions $(size(B))"))
    end
    if size(C,1) != size(A,1) || size(C,2) != size(B,2)
        throw(DimensionMismatch("result C has dimensions $(size(C)), needs $((size(A,1),size(B,2)))"))
    end
    if isempty(A) || isempty(B)
        return fill!(C, zero(R))
    end

    @kernel function matmatmul_kernel!(C, A, B)
        assume.(size(C) .> 0)
        idx = @index(Global, Linear)
        i, j = @inbounds Tuple(CartesianIndices(C)[idx])..., 1

        @inbounds if i <= size(A,1) && j <= size(B,2)
            z2 = zero(A[i, 1]*B[1, j] + A[i, 1]*B[1, j])
            Cij = convert(promote_type(R, typeof(z2)), z2)
            for k in 1:size(A,2)
                Cij += A[i, k]*B[k, j]
            end
            C[i,j] = add(Cij, C[i,j])
        end
    end
    matmatmul_kernel!(get_backend(C))(C, A, B; ndrange = size(C))
    C
end

@static if !isdefined(LinearAlgebra, Symbol("@stable_muladdmul")) # @stable_muladdmul was added in 1.12
function LinearAlgebra.generic_matvecmul!(C::AbstractGPUVector, tA::AbstractChar, A::AbstractGPUMatrix, B::AbstractGPUVector, _add::MulAddMul = MulAddMul())
    generic_matmatmul!(C, wrap(A, tA), B, _add)
end

function LinearAlgebra.generic_matmatmul!(C::AbstractGPUVecOrMat, tA, tB, A::AbstractGPUVecOrMat, B::AbstractGPUVecOrMat, _add::MulAddMul=MulAddMul())
    generic_matmatmul!(C, wrap(A, tA), wrap(B, tB), _add)
end
else
function LinearAlgebra.generic_matvecmul!(C::AbstractGPUVector, tA::AbstractChar, A::AbstractGPUMatrix, B::AbstractGPUVector, a::Number, b::Number)
    LinearAlgebra.@stable_muladdmul generic_matmatmul!(C, wrap(A, tA), B, MulAddMul(a, b))
end

function LinearAlgebra.generic_matmatmul!(C::AbstractGPUVecOrMat, tA, tB, A::AbstractGPUVecOrMat, B::AbstractGPUVecOrMat, a::Number, b::Number)
    LinearAlgebra.@stable_muladdmul generic_matmatmul!(C, wrap(A, tA), wrap(B, tB), MulAddMul(a, b))
end
end
@static if VERSION ≥ v"1.12.0-rc"
    # we need to use the generic wrapper to avoid dispatch to the 2x2or3x3 method
    using LinearAlgebra: generic_matmatmul_wrapper!, BlasFlag
    function LinearAlgebra.generic_matmatmul_wrapper!(C::AbstractGPUMatrix{T}, tA::AbstractChar, tB::AbstractChar, A::AbstractGPUVecOrMat{T}, B::AbstractGPUVecOrMat{T}, alpha::Number, beta::Number, val::LinearAlgebra.BlasFlag.SyrkHerkGemm) where {T}
        LinearAlgebra.generic_matmatmul!(C, tA, tB, A, B, alpha, beta)
    end
    # Julia 1.12 introduced generic_mul! for scalar * array operations
    function LinearAlgebra.generic_mul!(C::AbstractGPUVecOrMat, X::AbstractGPUVecOrMat, s::Number, alpha::Number, beta::Number)
        if length(C) != length(X)
            throw(DimensionMismatch(lazy"first array has length $(length(C)) which does not match the length of the second, $(length(X))."))
        end
        @. C = X * s * alpha + C * beta
        return C
    end
    function LinearAlgebra.generic_mul!(C::AbstractGPUVecOrMat, s::Number, X::AbstractGPUVecOrMat, alpha::Number, beta::Number)
        if length(C) != length(X)
            throw(DimensionMismatch(lazy"first array has length $(length(C)) which does not match the length of the second, $(length(X))."))
        end
        @. C = s * X * alpha + C * beta
        return C
    end
end

function generic_trimatmul!(C::AbstractGPUVecOrMat{R}, uploc, isunitc, tfun::Function, A::AbstractGPUMatrix{T}, B::AbstractGPUVecOrMat{S}) where {T,S,R}
    if size(A,2) != size(B,1)
        throw(DimensionMismatch(lazy"matrix A has dimensions $(size(A)), matrix B has dimensions $(size(B))"))
    end
    if size(C,1) != size(A,1) || size(C,2) != size(B,2)
        throw(DimensionMismatch(lazy"result C has dimensions $(size(C)), needs $((size(A,1),size(B,2)))"))
    end
    if isempty(A) || isempty(B)
        return fill!(C, zero(R))
    end

    upper = tfun === identity ? uploc == 'U' :  uploc != 'U'
    unit  = isunitc == 'U'

    @kernel function trimatmul(C, A, B)
        idx = @index(Global, Linear)
        assume.(size(C) .> 0)
        i, j = @inbounds Tuple(CartesianIndices(C)[idx])..., 1
        l, m, n = size(A, 1), size(B, 1), size(B, 2)

        @inbounds if i <= l && j <= n
            z2 = zero(A[i,1] * B[1,j] + A[i,1] * B[1,j])
            Cij = convert(promote_type(R, typeof(z2)), z2)
            Cij += (unit ? one(Cij) : A[i,i]) * B[i,j]
            for k in (upper ? (i + 1) : 1):(upper ? m : (i - 1))
                Cij += A[i,k] * B[k,j]
            end
            C[i,j] += Cij
        end
    end

    @kernel function trimatmul_t(C, A, B)
        idx = @index(Global, Linear)
        assume.(size(C) .> 0)
        i, j = @inbounds Tuple(CartesianIndices(C)[idx])..., 1
        l, m, n = size(A, 1), size(B, 1), size(B, 2)

        @inbounds if i <= l && j <= n
            z2 = zero(A[i,1] * B[1,j] + A[i,1] * B[1,j])
            Cij = convert(promote_type(R, typeof(z2)), z2)
            Cij += (unit ? one(Cij) : transpose(A[i,i])) * B[i,j]
            for k in (upper ? (i + 1) : 1):(upper ? m : (i - 1))
                Cij += transpose(A[k,i]) * B[k,j]
            end
            C[i,j] += Cij
        end
    end

    @kernel function trimatmul_a(C, A, B)
        idx = @index(Global, Linear)
        assume.(size(C) .> 0)
        i, j = @inbounds Tuple(CartesianIndices(C)[idx])..., 1
        l, m, n = size(A, 1), size(B, 1), size(B, 2)

        @inbounds if i <= l && j <= n
            z2 = zero(A[i,1] * B[1,j] + A[i,1] * B[1,j])
            Cij = convert(promote_type(R, typeof(z2)), z2)
            Cij += (unit ? one(Cij) : adjoint(A[i,i])) * B[i,j]
            for k in (upper ? (i + 1) : 1):(upper ? m : (i - 1))
                Cij += adjoint(A[k,i]) * B[k,j]
            end
            C[i,j] += Cij
        end
    end

    if tfun === identity
        trimatmul(get_backend(C))(C, A, B; ndrange = length(C))
    elseif tfun == transpose
        trimatmul_t(get_backend(C))(C, A, B; ndrange = length(C))
    elseif tfun === adjoint
        trimatmul_a(get_backend(C))(C, A, B; ndrange = length(C))
    else
        error("Not supported")
    end

    C
end

function generic_mattrimul!(C::AbstractGPUVecOrMat{R}, uploc, isunitc, tfun::Function, A::AbstractGPUMatrix{T}, B::AbstractGPUVecOrMat{S}) where {T,S,R}
    if size(A,2) != size(B,1)
        throw(DimensionMismatch(lazy"matrix A has dimensions $(size(A)), matrix B has dimensions $(size(B))"))
    end
    if size(C,1) != size(A,1) || size(C,2) != size(B,2)
        throw(DimensionMismatch(lazy"result C has dimensions $(size(C)), needs $((size(A,1),size(B,2)))"))
    end
    if isempty(A) || isempty(B)
        return fill!(C, zero(R))
    end

    upper = tfun === identity ? uploc == 'U' :  uploc != 'U'
    unit  = isunitc == 'U'

    @kernel function mattrimul(C, A, B)
        idx = @index(Global, Linear)
        assume.(size(C) .> 0)
        i, j = @inbounds Tuple(CartesianIndices(C)[idx])..., 1
        l, m, n = size(A, 1), size(B, 1), size(B, 2)

        @inbounds if i <= l && j <= n
            z2 = zero(A[i,1] * B[1,j] + A[i,1] * B[1,j])
            Cij = convert(promote_type(R, typeof(z2)), z2)
            Cij += A[i,j] * (unit ? one(Cij) : B[j,j])
            for k in (upper ? 1 : (j + 1)):(upper ? (j - 1) : m)
                Cij += A[i,k] * B[k,j]
            end
            C[i,j] += Cij
        end
    end

    @kernel function mattrimul_t(C, A, B)
        idx = @index(Global, Linear)
        assume.(size(C) .> 0)
        i, j = @inbounds Tuple(CartesianIndices(C)[idx])..., 1
        l, m, n = size(A, 1), size(B, 1), size(B, 2)

        @inbounds if i <= l && j <= n
            z2 = zero(A[i,1] * B[1,j] + A[i,1] * B[1,j])
            Cij = convert(promote_type(R, typeof(z2)), z2)
            Cij += A[i,j] * (unit ? one(Cij) : transpose(B[j,j]))
            for k in (upper ? 1 : (j + 1) ):(upper ? (j - 1) : m)
                Cij += A[i,k] * transpose(B[j,k])
            end
            C[i,j] += Cij
        end
    end

    @kernel function mattrimul_a(C, A, B)
        idx = @index(Global, Linear)
        assume.(size(C) .> 0)
        i, j = @inbounds Tuple(CartesianIndices(C)[idx])..., 1
        l, m, n = size(A, 1), size(B, 1), size(B, 2)

        @inbounds if i <= l && j <= n
            z2 = zero(A[i,1] * B[1,j] + A[i,1] * B[1,j])
            Cij = convert(promote_type(R, typeof(z2)), z2)
            Cij += A[i,j] * (unit ? one(Cij) : adjoint(B[j,j]))
            for k in (upper ? 1 : (j + 1)):(upper ? (j - 1) : m)
                Cij += A[i,k] * adjoint(B[j,k])
            end
            C[i,j] += Cij
        end
    end

    if tfun === identity
        mattrimul(get_backend(C))(C, A, B; ndrange = length(C))
    elseif tfun == transpose
        mattrimul_t(get_backend(C))(C, A, B; ndrange = length(C))
    elseif tfun === adjoint
        mattrimul_a(get_backend(C))(C, A, B; ndrange = length(C))
    else
        error("Not supported")
    end

    C
end

function LinearAlgebra.generic_trimatmul!(C::AbstractGPUVecOrMat, uploc, isunitc, tfun::Function, A::AbstractGPUMatrix, B::AbstractGPUVecOrMat)
    generic_trimatmul!(C, uploc, isunitc, tfun, A, B)
end
function LinearAlgebra.generic_mattrimul!(C::AbstractGPUMatrix, uploc, isunitc, tfun::Function, A::AbstractGPUMatrix, B::AbstractGPUMatrix)
    generic_mattrimul!(C, uploc, isunitc, tfun, A, B)
end

function generic_rmul!(X::AbstractArray, s::Number)
    @kernel function rmul_kernel!(X, s)
        i = @index(Global, Linear)
        @inbounds X[i] *= s
    end
    rmul_kernel!(get_backend(X))(X, s; ndrange = size(X))
    return X
end

LinearAlgebra.rmul!(A::AbstractGPUArray, b::Number) = generic_rmul!(A, b)

function generic_lmul!(s::Number, X::AbstractArray)
    @kernel function lmul_kernel!(X, s)
        i = @index(Global, Linear)
        @inbounds X[i] = s*X[i]
    end
    lmul_kernel!(get_backend(X))(X, s; ndrange = size(X))
    return X
end

LinearAlgebra.lmul!(a::Number, B::AbstractGPUArray) = generic_lmul!(a, B)


## permutedims
LinearAlgebra.permutedims!(dest::AbstractGPUArray, src::AbstractGPUArray, perm) =
    permutedims!(dest, src, Tuple(perm))

@inline @generated function permute_linearindex(size::NTuple{N,T}, l::T,
                                                strides::NTuple{N,T}) where {N,T}
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

function LinearAlgebra.permutedims!(dest::AbstractGPUArray,
                                    src::AbstractGPUArray,
                                    perm::NTuple{N}) where N
    if length(dest) <= typemax(UInt32)
        _permutedims!(UInt32, dest, src, perm)
    else
        _permutedims!(UInt64, dest, src, perm)
    end
end

function _permutedims!(::Type{IT}, dest::AbstractGPUArray,
                       src::AbstractGPUArray, perm::NTuple{N}) where {IT,N}
    @assert length(src) <= typemax(IT)
    Base.checkdims_perm(dest, src, perm)
    dest_strides = ntuple(k->k==1 ? 1 : prod(i->size(dest, i), 1:k-1), N)
    dest_strides_perm = ntuple(i->IT(dest_strides[findfirst(==(i), perm)]), N)
    size_src = IT.(size(src))
    @kernel function permutedims_kernel!(dest, src, size_src, dest_strides_perm)
        SLI = @index(Global, Linear)
        assume(0 < SLI <= typemax(IT))
        LI = IT(SLI)
        dest_index = permute_linearindex(size_src, LI, dest_strides_perm)
        @inbounds dest[dest_index] = src[LI]
    end
    kernel = permutedims_kernel!(get_backend(dest))
    kernel(vec(dest), vec(src), size_src, dest_strides_perm; ndrange = size(dest))
    return dest
end

## norm

function LinearAlgebra.norm(v::AbstractGPUArray{T}, p::Real=2) where {T}
    result_type, sum_type, promote_ = _normtypes(T)
    isempty(v) && return zero(result_type)
    p == 0 && return convert(result_type, count(!iszero, v))
    spp = convert(sum_type, p)
    init = zero(sum_type)  # To set the accumulation type in `sum`
    # Rescaling heuristic similar to Base, see LinearAlgebra/src/generic.jl
    result = if p > 1 || p < -1  # May need rescaling
        infnorm = p > 1 ? maximum(norm, v) : minimum(norm, v)
        if isinf(p) || iszero(infnorm) || isinf(infnorm)
            return convert(result_type, infnorm)  # Return early to skip conversions
        end
        factor = convert(sum_type, infnorm)
        if p == 2
            if isfinite(length(v) * factor^2) && !iszero(factor^2)  # No rescaling
                sqrt(sum(x -> LinearAlgebra.norm_sqr(promote_(x)), v; init=init))
            else  # Rescaling
                factor * sqrt(sum(x -> (norm(promote_(x)) / factor)^2, v; init=init))
            end
        else
            if isfinite(length(v) * factor^spp) && !iszero(factor^spp)  # No rescaling
                sum(x -> norm(promote_(x))^spp, v; init=init)^inv(spp)
            else  # Rescaling
                factor * (sum(x -> (norm(promote_(x)) / factor)^spp, v; init=init)^inv(spp))
            end
        end
    elseif p == 1
        sum(x -> norm(promote_(x)), v; init=init)
    else
        sum(x -> norm(promote_(x))^spp, v; init=init)^inv(spp)
    end
    return convert(result_type, result)
end

function _normtypes(::Type{T}) where {T}
    result_type = typeof(float(norm(zero(T))))
    # Accumulate in at least Float32, like nrm2 in CUBLAS
    sum_type = promote_type(Float32, result_type)
    # If sum_type is wider than T, promote before applying other functions. To work in GPU
    # kernels this operation must close around a value, not a type, hence the prototype
    prototype = zero(promote_type(T, sum_type))
    promote_(x) = convert(typeof(prototype), x)
    return result_type, sum_type, promote_
end

## opnorm

function LinearAlgebra.opnorm1(A::AnyGPUArray{T,2}) where {T}
    result_type, sum_type, promote_ = _normtypes(T)
    result = maximum(sum(x -> norm(promote_(x)), A; dims=1); init=zero(sum_type))
    return convert(result_type, result)
end

function LinearAlgebra.opnormInf(A::AnyGPUArray{T,2}) where {T}
    result_type, sum_type, promote_ = _normtypes(T)
    result = maximum(sum(x -> norm(promote_(x)), A; dims=2); init=zero(sum_type))
    return convert(result_type, result)
end

## symmetric

# prevent scalar indexing (upstream? this version is slower than a simple loop)
function Base.similar(A::Hermitian{<:Any,<:AbstractGPUArray}, ::Type{T}) where T
    B = similar(parent(A), T)
    fill!(view(B, diagind(B)), 0)
    return Hermitian(B, ifelse(A.uplo == 'U', :U, :L))
end

## rotate

function LinearAlgebra.rotate!(x::AbstractGPUArray, y::AbstractGPUArray, c::Number, s::Number)
    @kernel function rotate_kernel!(x, y, c, s)
        i = @index(Global, Linear)
        @inbounds xi = x[i]
        @inbounds yi = y[i]
        @inbounds x[i] = s*yi +      c *xi
        @inbounds y[i] = c*yi - conj(s)*xi
    end
    rotate_kernel!(get_backend(x))(x, y, c, s; ndrange = size(x))
    return x, y
end

## reflect

function LinearAlgebra.reflect!(x::AbstractGPUArray, y::AbstractGPUArray, c::Number, s::Number)
    @kernel function  reflect_kernel!(x, y, c, s)
        i = @index(Global, Linear)
        @inbounds xi = x[i]
        @inbounds yi = y[i]
        @inbounds x[i] =      c  * xi + s * yi
        @inbounds y[i] = conj(s) * xi - c * yi
    end
    reflect_kernel!(get_backend(x))(x, y, c, s; ndrange = size(x))
    return x, y
end

## dot

LinearAlgebra.dot(x::AbstractGPUArray, y::AbstractGPUArray) = mapreduce(dot, +, x, y)

## axp{b}y

LinearAlgebra.axpby!(alpha::Number, x::AbstractGPUArray,
                     beta::Number,  y::AbstractGPUArray) = y .= x.*alpha .+ y.*beta

LinearAlgebra.axpy!(alpha::Number, x::AbstractGPUArray, y::AbstractGPUArray) = y .+= x.*alpha

## identity and zero equality check

Base.iszero(x::AbstractGPUMatrix{T}) where {T} = all(iszero, x)
function Base.isone(x::AbstractGPUMatrix{T}) where {T}
    n,m = size(x)
    m != n && return false

    # lazily perform `x-I`
    bc = Broadcast.broadcasted(x, CartesianIndices(x)) do _x, inds
        _x - (inds[1] == inds[2] ? one(_x) : zero(_x))
    end
    # call `GPUArrays.mapreducedim!` directly, which supports Broadcasted inputs
    y = similar(x, Bool, 1)
    GPUArrays.mapreducedim!(iszero, &, y, Broadcast.instantiate(bc); init=true)

    Array(y)[]
end

## Kronecker product

@kernel function kron_kernel_vec!(z, @Const(x), @Const(y))
    i, j = @index(Global, NTuple)

    @inbounds z[(i - 1) * length(y) + j] = x[i] * y[j]
end

function LinearAlgebra.kron!(z::AbstractGPUVector{T1}, x::AbstractGPUVector{T2}, y::AbstractGPUVector{T3}) where {T1,T2,T3}
    @assert length(z) == length(x) * length(y)

    backend = KernelAbstractions.get_backend(z)
    kernel = kron_kernel_vec!(backend)

    kernel(z, x, y, ndrange=(length(x), length(y)))

    return z
end

function LinearAlgebra.kron(x::AbstractGPUVector{T1}, y::AbstractGPUVector{T2}) where {T1,T2}
    T = promote_type(T1, T2)
    z = similar(x, T, length(x) * length(y))
    return kron!(z, x, y)
end

@kernel function kron_kernel!(C, @Const(A), @Const(B))
    ai, aj = @index(Global, NTuple)  # Indices in the result matrix

    # lb1, lb2 = size(B)  # Dimensions of B
    lb1 = size(B, 1)
    lb2 = size(B, 2)

    # Map global indices (ai, aj) to submatrices of the Kronecker product
    i_a = fld1(ai, lb1)  # Corresponding row index in A
    i_b = mod1(ai, lb1)  # Corresponding row index in B
    j_a = fld1(aj, lb2)  # Corresponding col index in A
    j_b = mod1(aj, lb2)  # Corresponding col index in B

    @inbounds C[ai, aj] = A[i_a, j_a] * B[i_b, j_b]
end

trans_adj_wrappers = (
    T -> :(AbstractGPUVecOrMat{$T}),
    T -> :(Transpose{$T, <:AbstractGPUVecOrMat{$T}}),
    T -> :(Adjoint{$T, <:AbstractGPUVecOrMat{$T}}),
)

for wrapa in trans_adj_wrappers, wrapb in trans_adj_wrappers
    TypeA = wrapa(:T1)
    TypeB = wrapb(:T2)
    TypeC = :(AbstractGPUVecOrMat{T3})

    @eval function LinearAlgebra.kron!(C::$TypeC, A::$TypeA, B::$TypeB) where {T1, T2, T3}
        @assert size(C, 1) == size(A, 1) * size(B, 1)
        @assert size(C, 2) == size(A, 2) * size(B, 2)

        backend = KernelAbstractions.get_backend(C)
        kernel = kron_kernel!(backend)

        kernel(C, A, B, ndrange=(size(C, 1), size(C, 2)))

        return C
    end

    @eval function LinearAlgebra.kron(A::$TypeA, B::$TypeB) where {T1, T2}
        T = promote_type(T1, T2)
        C = similar(A, T, size(A, 1) * size(B, 1), size(A, 2) * size(B, 2))
        return kron!(C, A, B)
    end
end
