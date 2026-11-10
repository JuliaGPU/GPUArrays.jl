# integration with LinearAlgebra stdlib

using LinearAlgebra: MulAddMul, wrap, diagm, BlasReal, AbstractTriangular, AdjOrTrans

const StridedGPUSubArray{T,N} = Base.StridedSubArray{T,N,<:AbstractGPUArray}
const AnyStridedGPUArray{T,N} = Union{AbstractGPUArray{T,N},StridedGPUSubArray{T,N}}
const AnyStridedGPUVector{T} = AnyStridedGPUArray{T,1}
const AnyStridedGPUMatrix{T} = AnyStridedGPUArray{T,2}
const AnyStridedGPUVecOrMat{T} = Union{AnyStridedGPUVector{T},AnyStridedGPUMatrix{T}}

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

function LinearAlgebra.copytri!(A::AbstractGPUMatrix, uplo::AbstractChar, conjugate::Bool = false, diag::Bool = false)
    n = LinearAlgebra.checksquare(A)
    if uplo == 'U' && conjugate
        @kernel function U_conj!(_A)
            I = @index(Global, Cartesian)
            i, j = Tuple(I)
            if j + diag > i
                @inbounds _A[j,i] = conj(_A[i,j])
          end
        end
        U_conj!(get_backend(A))(A; ndrange = size(A))
    elseif uplo == 'U' && !conjugate
        @kernel function U_noconj!(_A)
            I = @index(Global, Cartesian)
            i, j = Tuple(I)
            if j + diag > i
                @inbounds _A[j,i] = _A[i,j]
          end
        end
        U_noconj!(get_backend(A))(A; ndrange = size(A))
    elseif uplo == 'L' && conjugate
        @kernel function L_conj!(_A)
            I = @index(Global, Cartesian)
            i, j = Tuple(I)
            if j + diag > i
                @inbounds _A[i,j] = conj(_A[j,i])
            end
        end
        L_conj!(get_backend(A))(A; ndrange = size(A))
    elseif uplo == 'L' && !conjugate
        @kernel function L_noconj!(_A)
            I = @index(Global, Cartesian)
            i, j = Tuple(I)
            if j + diag > i
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
function LinearAlgebra.istriu(A::AnyStridedGPUMatrix, k::Integer = 0)
    function mapper(a, I)
        row, col = Tuple(I)
        if col < row + k
            return iszero(a)
        else
            true
        end
    end
    mapreduce(mapper, &, A, eachindex(IndexCartesian(), A); init=true)
end

# check if lower triangular starting from the kth subdiagonal.
function LinearAlgebra.istril(A::AnyStridedGPUMatrix, k::Integer = 0)
    function mapper(a, I)
        row, col = Tuple(I)
        if col > row + k
            return iszero(a)
        else
            true
        end
    end
    mapreduce(mapper, &, A, eachindex(IndexCartesian(), A); init=true)
end

# istriu/istril of triangular wrappers of the opposite triangularity: the generic
# fallbacks in LinearAlgebra scan the wrapper element-by-element, which requires scalar
# indexing. forwarding to the parent ends up in the mapreduce-based methods above, also
# for lazily wrapped parents (via LinearAlgebra's Adjoint/Transpose forwards). needed by
# e.g. 2-arg ldiv!/rdiv! on Julia 1.11.2+, which consult istriu/istril to select the
# solver; LinearAlgebra provides equivalent methods on Julia 1.13+
# (JuliaLang/LinearAlgebra.jl#1265), making these redundant there.
@static if VERSION < v"1.13.0-"
LinearAlgebra.istriu(A::LowerTriangular{<:Any, <:AnyGPUMatrix}, k::Integer = 0) =
    istriu(parent(A), min(k, 1))
LinearAlgebra.istriu(A::UnitLowerTriangular{<:Any, <:AnyGPUMatrix}, k::Integer = 0) =
    k <= 0 && istriu(parent(A), k)
LinearAlgebra.istril(A::UpperTriangular{<:Any, <:AnyGPUMatrix}, k::Integer = 0) =
    istril(parent(A), max(k, -1))
LinearAlgebra.istril(A::UnitUpperTriangular{<:Any, <:AnyGPUMatrix}, k::Integer = 0) =
    k >= 0 && istril(parent(A), k)
end

## uniformscaling

function LinearAlgebra.mul!(out::AnyGPUMatrix{T}, a::Number, B::LinearAlgebra.UniformScaling, α::Number, β::Number) where {T}
    LinearAlgebra.checksquare(out)
    if iszero(β)
        fill!(out, zero(T))
    elseif !isone(β)
        rmul!(out, β)
    end
    s = convert(T, a*B.λ*α)
    if !iszero(s)
        view(out, diagind(out)) .+= s
    end
    return out
end

@static if VERSION >= v"1.12.0-rc"
    import LinearAlgebra: _lscale_add!
    function LinearAlgebra._lscale_add!(C::AnyGPUMatrix, s::Number, X::AnyGPUMatrix, alpha::Number, beta::Number)
        if axes(C) == axes(X)
            if isone(alpha)
                if iszero(beta)
                    @. C = s * X
                else
                    @. C = s * X + C * beta
                end
            else
                if iszero(beta)
                    @. C = s * X * alpha
                else
                    @. C = s * X * alpha + C * beta
                end
            end
        else
            generic_mul!(C, s, X, alpha, beta)
        end
        return C
    end
end

## diagonal

Base.copy(D::Diagonal{T, <:AbstractGPUArray{T, N}}) where {T, N} = Diagonal(copy(D.diag))

# copy only the diagonal; the generic `AnyGPUArray` method would launch a kernel over all
# n^2 entries, calling `setindex!(::Diagonal, ...)` whose error path isn't GPU-compatible
function Base.copyto!(D1::Diagonal{T, <:AbstractGPUArray{T, N}},
                      D2::Diagonal{S, <:AbstractGPUArray{S, M}}) where {T, N, S, M}
    copyto!(D1.diag, D2.diag)
    return D1
end

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
function LinearAlgebra.mul!(C::AbstractGPUArray,
                            A::Diagonal{<:Any, <:AbstractGPUArray},
                            B::Diagonal{<:Any, <:AbstractGPUArray})
    C .= zero(eltype(C))
    dc = view(C, LinearAlgebra.diagind(C))
    da = A.diag
    db = B.diag
    d = length(dc)
    length(da) == d || throw(DimensionMismatch("right hand side has $(length(da)) rows but output is $d by $d"))
    length(db) == d || throw(DimensionMismatch("left hand side has $(length(db)) rows but output is $d by $d"))
    @. dc = da * db
    return C
end

function LinearAlgebra.mul!(C::AbstractGPUArray,
                            A::Diagonal{<:Any, <:AbstractGPUArray},
                            B::Diagonal{<:Any, <:AbstractGPUArray},
                            α::Number,
                            β::Number)
    dc = view(C, LinearAlgebra.diagind(C))
    da = A.diag
    db = B.diag
    d = length(dc)
    length(da) == d || throw(DimensionMismatch("right hand side has $(length(da)) rows but output is $d by $d"))
    length(db) == d || throw(DimensionMismatch("left hand side has $(length(db)) rows but output is $d by $d"))
    # C may be uninitialized, so a `β` that is zero has to fill rather than scale
    iszero(β) ? fill!(C, zero(eltype(C))) : rmul!(C, β)
    @. dc += α * da * db
    return C
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

## matrix multiplication

const AnyStridedGPUMatrixOperand{T} = Union{
    AnyStridedGPUMatrix{T},
    Adjoint{T,<:AnyStridedGPUMatrix{T}},
    Transpose{T,<:AnyStridedGPUMatrix{T}},
    Symmetric{T,<:AnyStridedGPUMatrix{T}},
    Hermitian{T,<:AnyStridedGPUMatrix{T}},
}
const AnyStridedGPUVecOrMatOperand{T} = Union{
    AnyStridedGPUVector{T},
    AnyStridedGPUMatrixOperand{T},
}

has_strided_gpu_view(As...) =
    any(A -> LinearAlgebra._unwrap(A) isa StridedGPUSubArray, As)

blas_compatible(X) =
    X isa AbstractMatrix && stride(X, 1) == 1 && stride(X, 2) >= max(1, size(X, 1))

function strided_view_matmatmul!(C, A, B, a, b)
    tA, tB = LinearAlgebra.wrapper_char(A), LinearAlgebra.wrapper_char(B)
    pA, pB = LinearAlgebra._unwrap(A), LinearAlgebra._unwrap(B)
    if tA in ('N', 'T', 'C') && tB in ('N', 'T', 'C') && all(blas_compatible, (C, pA, pB)) &&
       size(A, 2) == size(B, 1) && size(C) == (size(A, 1), size(B, 2))
        return LinearAlgebra.mul!(C, tA, tB, pA, pB, a, b)
    end
    return generic_matmatmul!(C, A, B, a, b)
end

# Intercept before LinearAlgebra unwraps operands and dispatches to backend BLAS methods.
@static if VERSION < v"1.11"
@inline function LinearAlgebra.mul!(C::AnyStridedGPUVector, A::AnyStridedGPUMatrixOperand,
                            B::AnyStridedGPUVector, a::Number, b::Number)
    if has_strided_gpu_view(C, A, B)
        return generic_matmatmul!(C, A, B, a, b)
    end
    invoke(LinearAlgebra.mul!,
           Tuple{AbstractVector,LinearAlgebra.AbstractVecOrMat,AbstractVector,Number,Number},
           C, A, B, a, b)
    return C
end

@inline function LinearAlgebra.mul!(C::AnyStridedGPUMatrix, A::AnyStridedGPUVecOrMatOperand,
                            B::AnyStridedGPUVecOrMatOperand, a::Number, b::Number)
    if has_strided_gpu_view(C, A, B)
        return strided_view_matmatmul!(C, A, B, a, b)
    end
    invoke(LinearAlgebra.mul!,
           Tuple{AbstractMatrix,LinearAlgebra.AbstractVecOrMat,
                 LinearAlgebra.AbstractVecOrMat,Number,Number},
           C, A, B, a, b)
    return C
end
else
@inline function LinearAlgebra._mul!(C::AnyStridedGPUVector, A::AnyStridedGPUMatrixOperand,
                             B::AnyStridedGPUVector, a::Number, b::Number)
    if has_strided_gpu_view(C, A, B)
        return generic_matmatmul!(C, A, B, a, b)
    end
    invoke(LinearAlgebra._mul!,
           Tuple{AbstractVector,LinearAlgebra.AbstractVecOrMat,AbstractVector,Number,Number},
           C, A, B, a, b)
    return C
end

@inline function LinearAlgebra._mul!(C::AnyStridedGPUMatrix, A::AnyStridedGPUVecOrMatOperand,
                             B::AnyStridedGPUVecOrMatOperand, a::Number, b::Number)
    if has_strided_gpu_view(C, A, B)
        return strided_view_matmatmul!(C, A, B, a, b)
    end
    invoke(LinearAlgebra._mul!,
           Tuple{AbstractMatrix,LinearAlgebra.AbstractVecOrMat,
                 LinearAlgebra.AbstractVecOrMat,Number,Number},
           C, A, B, a, b)
    return C
end
end

function LinearAlgebra.mul!(C::AnyStridedGPUVector, tA::AbstractChar, A::AnyStridedGPUMatrix, B::AnyStridedGPUVector, a::Number, b::Number)
    tA = Char(tA)
    mA, nA = tA == 'N' ? size(A) : reverse(size(A))
    nA == length(B) || throw(DimensionMismatch(lazy"second dimension of A, $nA, does not match length of B, $(length(B))"))
    mA == length(C) || throw(DimensionMismatch(lazy"first dimension of A, $mA, does not match length of C, $(length(C))"))
    mA == 0 && return C
    nA == 0 && return LinearAlgebra._rmul_or_fill!(C, b)

    if blas_scalars(eltype(C), a, b) && all(blas_layout, (A, B, C))
        if tA in ('N', 'T', 'C')
            return blas_gemv!(tA, a, A, B, b, C)
        elseif tA in ('S', 's')
            return blas_symv!(tA == 'S' ? 'U' : 'L', a, A, B, b, C)
        elseif tA in ('H', 'h')
            return blas_hemv!(tA == 'H' ? 'U' : 'L', a, A, B, b, C)
        end
    end

    Cm = reshape(C, length(C), 1)
    Bm = reshape(B, length(B), 1)
    if Cm isa AnyStridedGPUMatrix && Bm isa AnyStridedGPUMatrix
        LinearAlgebra.mul!(Cm, tA, 'N', A, Bm, a, b)
    else
        generic_matmatmul!(C, tA, 'N', A, B, a, b)
    end
    return C
end

function LinearAlgebra.mul!(C::AnyStridedGPUVecOrMat, tA, tB, A::AnyStridedGPUVecOrMat, B::AnyStridedGPUVecOrMat, a::Number, b::Number)
    tA, tB = Char(tA), Char(tB)
    mA, nA = size(A, tA == 'N' ? 1 : 2), size(A, tA == 'N' ? 2 : 1)
    mB, nB = size(B, tB == 'N' ? 1 : 2), size(B, tB == 'N' ? 2 : 1)
    nA == mB || throw(DimensionMismatch(lazy"A has dimensions ($mA,$nA) but B has dimensions ($mB,$nB)"))
    (C === A || B === C) && throw(ArgumentError("output matrix must not be aliased with input matrix"))
    if mA == 0 || nA == 0 || nB == 0
        size(C, 1) == mA && size(C, 2) == nB ||
            throw(DimensionMismatch(lazy"C has dimensions $(size(C)), should have ($mA,$nB)"))
        return LinearAlgebra._rmul_or_fill!(C, b)
    end

    if all(blas_matrix_layout, (A, B, C))
        if tA in ('N', 'T', 'C') && tB in ('N', 'T', 'C')
            return blas_gemm!(tA, tB, a, A, B, b, C)
        elseif blas_scalars(eltype(C), a, b)
            if (tA == 'S' || tA == 's') && tB == 'N'
                return blas_symm!('L', tA == 'S' ? 'U' : 'L', a, A, B, b, C)
            elseif (tB == 'S' || tB == 's') && tA == 'N'
                return blas_symm!('R', tB == 'S' ? 'U' : 'L', a, B, A, b, C)
            elseif (tA == 'H' || tA == 'h') && tB == 'N'
                return blas_hemm!('L', tA == 'H' ? 'U' : 'L', a, A, B, b, C)
            elseif (tB == 'H' || tB == 'h') && tA == 'N'
                return blas_hemm!('R', tB == 'H' ? 'U' : 'L', a, B, A, b, C)
            end
        end
    end

    generic_matmatmul!(C, tA, tB, A, B, a, b)
end

# Julia < 1.13 dispatches on the non-public `generic_matvecmul!` and `generic_matmatmul!`,
# which JuliaLang/LinearAlgebra.jl#1671 superseded by the `mul!` methods above. Forward from
# the old names, both the alpha/beta variants (1.12) and the ones taking a final MulAddMul
# (1.10 and 1.11).
@static if VERSION < v"1.13.0-rc4"
    LinearAlgebra.generic_matvecmul!(C::AnyStridedGPUVector, tA::AbstractChar, A::AnyStridedGPUMatrix, B::AnyStridedGPUVector, a::Number, b::Number) =
        LinearAlgebra.mul!(C, tA, A, B, a, b)
    LinearAlgebra.generic_matvecmul!(C::AnyStridedGPUVector, tA::AbstractChar, A::AnyStridedGPUMatrix, B::AnyStridedGPUVector, _add::MulAddMul = MulAddMul()) =
        LinearAlgebra.mul!(C, tA, A, B, _add.alpha, _add.beta)
    LinearAlgebra.generic_matmatmul!(C::AnyStridedGPUVecOrMat, tA, tB, A::AnyStridedGPUVecOrMat, B::AnyStridedGPUVecOrMat, a::Number, b::Number) =
        LinearAlgebra.mul!(C, tA, tB, A, B, a, b)
    LinearAlgebra.generic_matmatmul!(C::AnyStridedGPUVecOrMat, tA, tB, A::AnyStridedGPUVecOrMat, B::AnyStridedGPUVecOrMat, _add::MulAddMul = MulAddMul()) =
        LinearAlgebra.mul!(C, tA, tB, A, B, _add.alpha, _add.beta)
end

# triangular × triangular matmul: C = α·(A·B) + β·C, with both A and B triangular.
# Shared by the version-specific LinearAlgebra entry points below, which differ only in how
# α/β arrive (raw scalars on ≥1.12, a MulAddMul on older). The wrapper type — not the data —
# fixes which triangle holds the nonzeros, so we read it with a plain `isa` check.
function _triangular_matmatmul!(C::AbstractGPUVecOrMat{R}, A::AbstractTriangular, B::AbstractTriangular, alpha::Number, beta::Number) where {R}
    if size(A,2) != size(B,1)
        throw(DimensionMismatch(lazy"matrix A has dimensions $(size(A)), matrix B has dimensions $(size(B))"))
    end
    if size(C,1) != size(A,1) || size(C,2) != size(B,2)
        throw(DimensionMismatch(lazy"result C has dimensions $(size(C)), needs $((size(A,1),size(B,2)))"))
    end
    if isempty(A) || isempty(B)
        return fill!(C, zero(R))
    end
    upperA = A isa Union{UnitUpperTriangular, UpperTriangular}
    @kernel function trimatmul(C, A, B, alpha, beta)
        idx = @index(Global, Linear)
        assume.(size(C) .> 0)
        i, j = @inbounds Tuple(CartesianIndices(C)[idx])..., 1
        l, m, n = size(A, 1), size(B, 1), size(B, 2)

        @inbounds if i <= l && j <= n
            z2 = zero(A[i, 1]*B[1, j] + A[i, 1]*B[1, j])
            Cij = convert(promote_type(R, typeof(z2)), z2)
            Cij = muladd(A[i,i], B[i,j], Cij)
            for k in (upperA ? (i + 1) : 1):(upperA ? m : (i - 1))
                Cij = muladd(A[i,k], B[k,j], Cij)
            end
            # treat C as write-only when beta is zero (it may hold NaN/Inf)
            C[i,j] = iszero(beta) ? alpha * Cij : alpha * Cij + beta * C[i,j]
        end
    end
    trimatmul(get_backend(C))(C, A, B, alpha, beta; ndrange = length(C))
    return C
end

@static if VERSION ≥ v"1.12.0-rc"
    using LinearAlgebra: generic_matmatmul_wrapper!, BlasFlag
    @static if VERSION < v"1.13.0-rc4"
        storage_matmatmul!(C, tA, tB, A, B, alpha, beta) =
            LinearAlgebra.generic_matmatmul!(C, tA, tB, A, B, alpha, beta)
    else
        storage_matmatmul!(C, tA, tB, A, B, alpha, beta) =
            LinearAlgebra.mul!(C, tA, tB, A, B, alpha, beta)
    end
    function LinearAlgebra.generic_matmatmul_wrapper!(C::AnyStridedGPUMatrix{T}, tA::AbstractChar, tB::AbstractChar, A::AnyStridedGPUVecOrMat{T}, B::AnyStridedGPUVecOrMat{T}, alpha::Number, beta::Number, val::LinearAlgebra.BlasFlag.SyrkHerkGemm) where {T}
        storage_matmatmul!(C, tA, tB, A, B, alpha, beta)
    end
    function LinearAlgebra.generic_matmatmul_wrapper!(C::AnyStridedGPUMatrix{T}, tA::AbstractChar, tB::AbstractChar, A::AnyStridedGPUVecOrMat{T}, B::AnyStridedGPUVecOrMat{T}, alpha::Number, beta::Number, val::LinearAlgebra.BlasFlag.SymmHemmGeneric) where {T}
        storage_matmatmul!(C, tA, tB, A, B, alpha, beta)
    end
    for flag in (:SYMM, :HEMM)
        @eval LinearAlgebra._symm_hemm_generic!(C::AnyStridedGPUMatrix, tA::AbstractChar, tB::AbstractChar, A::AnyStridedGPUMatrix, B::AnyStridedGPUMatrix, alpha, beta, ::Val{BlasFlag.$flag}) =
            storage_matmatmul!(C, tA, tB, A, B, alpha, beta)
    end
    function LinearAlgebra.generic_matmatmul_wrapper!(C::AnyStridedGPUMatrix{Complex{T}}, tA::AbstractChar, tB::AbstractChar, A::AnyStridedGPUVecOrMat{Complex{T}}, B::AnyStridedGPUVecOrMat{T}, alpha::Number, beta::Number, val::Val{LinearAlgebra.BlasFlag.GEMM}) where T<:Union{Float32, Float64}
        storage_matmatmul!(C, tA, tB, A, B, alpha, beta)
    end
    function LinearAlgebra.generic_matmatmul_wrapper!(C::AnyStridedGPUMatrix{Complex{T}}, tA::AbstractChar, tB::AbstractChar, A::AnyStridedGPUVecOrMat{T}, B::AnyStridedGPUVecOrMat{Complex{T}}, alpha::Number, beta::Number, val::Val{LinearAlgebra.BlasFlag.GEMM}) where T<:Union{Float32, Float64}
        storage_matmatmul!(C, tA, tB, A, B, alpha, beta)
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
    LinearAlgebra._generic_matmatmul_nonadjtrans!(C::AbstractGPUVecOrMat, A::AbstractTriangular, B::AbstractTriangular, alpha::Number, beta::Number) =
        _triangular_matmatmul!(C, A, B, alpha, beta)
end

@static if VERSION < v"1.11.0-rc"
    LinearAlgebra._generic_matmatmul!(C::AbstractGPUVecOrMat, tA::Char, tB::Char, A::AbstractTriangular, B::AbstractTriangular, add::LinearAlgebra.MulAddMul) =
        _triangular_matmatmul!(C, A, B, add.alpha, add.beta)
end

@static if v"1.11.0-rc" < VERSION < v"1.12.0-rc"
    LinearAlgebra._generic_matmatmul!(C::AbstractGPUVecOrMat, A::AbstractTriangular, B::AbstractTriangular, add::LinearAlgebra.MulAddMul) =
        _triangular_matmatmul!(C, A, B, add.alpha, add.beta)
end

function generic_trimatmul!(C::AnyStridedGPUVecOrMat{R}, uploc, isunitc, tfun::Function, A::AnyStridedGPUMatrix{T}, B::AbstractVecOrMat{S}) where {T,S,R}
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
    oA = oneunit(eltype(A))

    @kernel function trimatmul(C, A, B)
        idx = @index(Global, Linear)
        assume.(size(C) .> 0)
        i, j = @inbounds Tuple(CartesianIndices(C)[idx])..., 1
        l, m, n = size(A, 1), size(B, 1), size(B, 2)

        @inbounds if i <= l && j <= n
            z2 = zero(A[i,1] * B[1,j] + A[i,1] * B[1,j])
            Cij = convert(promote_type(R, typeof(z2)), z2)
            Cij = muladd(unit ? oA : A[i,i], B[i,j], Cij)
            for k in (upper ? (i + 1) : 1):(upper ? m : (i - 1))
                Cij = muladd(A[i,k], B[k,j], Cij)
            end
            C[i,j] = Cij
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
            Cij = muladd(unit ? oA : transpose(A[i,i]), B[i,j], Cij)
            for k in (upper ? (i + 1) : 1):(upper ? m : (i - 1))
                Cij = muladd(transpose(A[k,i]), B[k,j], Cij)
            end
            C[i,j] = Cij
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
            Cij = muladd(unit ? oA : adjoint(A[i,i]), B[i,j], Cij)
            for k in (upper ? (i + 1) : 1):(upper ? m : (i - 1))
                Cij = muladd(adjoint(A[k,i]), B[k,j], Cij)
            end
            C[i,j] = Cij
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

function generic_mattrimul!(C::AnyStridedGPUVecOrMat{R}, uploc, isunitc, tfun::Function, A::AbstractVecOrMat{T}, B::AnyStridedGPUMatrix{S}) where {T,S,R}
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
    oB = oneunit(eltype(B))

    @kernel function mattrimul(C, A, B)
        idx = @index(Global, Linear)
        assume.(size(C) .> 0)
        i, j = @inbounds Tuple(CartesianIndices(C)[idx])..., 1
        l, m, n = size(A, 1), size(B, 1), size(B, 2)

        @inbounds if i <= l && j <= n
            z2 = zero(A[i,1] * B[1,j] + A[i,1] * B[1,j])
            Cij = convert(promote_type(R, typeof(z2)), z2)
            Cij = muladd(A[i,j], unit ? oB : B[j,j], Cij)
            for k in (upper ? 1 : (j + 1)):(upper ? (j - 1) : m)
                Cij = muladd(A[i,k], B[k,j], Cij)
            end
            C[i,j] = Cij
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
            Cij = muladd(A[i,j], unit ? oB : transpose(B[j,j]), Cij)
            for k in (upper ? 1 : (j + 1) ):(upper ? (j - 1) : m)
                Cij = muladd(A[i,k], transpose(B[j,k]), Cij)
            end
            C[i,j] = Cij
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
            Cij = muladd(A[i,j], unit ? oB : adjoint(B[j,j]), Cij)
            for k in (upper ? 1 : (j + 1)):(upper ? (j - 1) : m)
                Cij = muladd(A[i,k], adjoint(B[j,k]), Cij)
            end
            C[i,j] = Cij
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

# triangular solves parallelize only across right-hand sides; each one is a sequential
# forward/backward substitution. backends with a vendor BLAS override these with native
# trsm calls, so these fallbacks mainly serve backends without one. unlike LinearAlgebra's
# generic implementation, a zero diagonal element does not throw a SingularException but
# yields Infs/NaNs, like vendor BLAS libraries.
function generic_trimatdiv!(C::AnyStridedGPUVecOrMat{R}, uploc, isunitc, tfun::Function, A::AnyStridedGPUMatrix{T}, B::AbstractVecOrMat{S}) where {T,S,R}
    if size(A,2) != size(B,1)
        throw(DimensionMismatch(lazy"matrix A has dimensions $(size(A)), right hand side B has dimensions $(size(B))"))
    end
    if size(C) != size(B)
        throw(DimensionMismatch(lazy"result C has dimensions $(size(C)), needs $(size(B))"))
    end
    isempty(B) && return C

    upper = tfun === identity ? uploc == 'U' : uploc != 'U'
    unit  = isunitc == 'U'

    # C = tfun(A) \ B; safe for C === B, as B[i,j] is read before C[i,j] is written
    @kernel function trimatdiv(C, A, B, tf)
        j = @index(Global, Linear)
        m, n = size(C, 1), size(C, 2)

        @inbounds if j <= n
            for ii in 1:m
                i = upper ? m - ii + 1 : ii
                Cij = convert(promote_type(R, S), B[i,j])
                for k in (upper ? (i + 1) : 1):(upper ? m : (i - 1))
                    Aik = tf === identity ? A[i,k] : tf(A[k,i])
                    Cij -= Aik * C[k,j]
                end
                C[i,j] = unit ? Cij : tf(A[i,i]) \ Cij
            end
        end
    end

    trimatdiv(get_backend(C))(C, A, B, tfun; ndrange = size(C, 2))

    C
end

function generic_mattridiv!(C::AnyStridedGPUMatrix{R}, uploc, isunitc, tfun::Function, A::AbstractMatrix{T}, B::AnyStridedGPUMatrix{S}) where {T,S,R}
    if size(A,2) != size(B,1)
        throw(DimensionMismatch(lazy"left hand side A has dimensions $(size(A)), matrix B has dimensions $(size(B))"))
    end
    if size(C) != size(A)
        throw(DimensionMismatch(lazy"result C has dimensions $(size(C)), needs $(size(A))"))
    end
    isempty(A) && return C

    upper = tfun === identity ? uploc == 'U' : uploc != 'U'
    unit  = isunitc == 'U'

    # C = A / tfun(B); safe for C === A, as A[i,j] is read before C[i,j] is written
    @kernel function mattridiv(C, A, B, tf)
        i = @index(Global, Linear)
        m, n = size(C, 1), size(C, 2)

        @inbounds if i <= m
            for jj in 1:n
                j = upper ? jj : n - jj + 1
                Cij = convert(promote_type(R, T), A[i,j])
                for k in (upper ? 1 : (j + 1)):(upper ? (j - 1) : n)
                    Bkj = tf === identity ? B[k,j] : tf(B[j,k])
                    Cij -= C[i,k] * Bkj
                end
                C[i,j] = unit ? Cij : Cij / tf(B[j,j])
            end
        end
    end

    mattridiv(get_backend(C))(C, A, B, tfun; ndrange = size(C, 1))

    C
end

const AnyGPUOperand = Union{AnyStridedGPUVecOrMat, AdjOrTrans{<:Any,<:AnyStridedGPUVecOrMat}}

# copy `B` into `C`, for in-place BLAS routines, materializing a wrapped `B`
operand_copyto!(C, B) = C === B ? C : (C .= B)

function LinearAlgebra.generic_trimatmul!(C::AnyStridedGPUVecOrMat, uploc, isunitc, tfun::Function, A::AnyStridedGPUMatrix, B::AnyGPUOperand)
    if allequal(map(eltype, (C, A, B))) && size(C) == size(B) && blas_layout(A) && blas_layout(C)
        trans = trans_char(tfun)
        if C isa AbstractVector
            return blas_trmv!(uploc, trans, isunitc, A, operand_copyto!(C, B))
        elseif B isa AnyStridedGPUMatrix && blas_layout(B)
            return blas_trmm!('L', uploc, trans, isunitc, true, A, B, C)
        end
    end
    generic_trimatmul!(C, uploc, isunitc, tfun, A, B)
end
function LinearAlgebra.generic_mattrimul!(C::AnyStridedGPUMatrix, uploc, isunitc, tfun::Function, A::AnyGPUOperand, B::AnyStridedGPUMatrix)
    if allequal(map(eltype, (C, A, B))) && size(C) == size(A) && A isa AnyStridedGPUMatrix &&
       all(blas_layout, (A, B, C))
        return blas_trmm!('R', uploc, trans_char(tfun), isunitc, true, B, A, C)
    end
    generic_mattrimul!(C, uploc, isunitc, tfun, A, B)
end
function LinearAlgebra.generic_trimatdiv!(C::AnyStridedGPUVecOrMat, uploc, isunitc, tfun::Function, A::AnyStridedGPUMatrix, B::AnyGPUOperand)
    if allequal(map(eltype, (C, A, B))) && size(C) == size(B) && blas_layout(A) && blas_layout(C)
        trans = trans_char(tfun)
        X = operand_copyto!(C, B)
        if C isa AbstractVector
            return blas_trsv!(uploc, trans, isunitc, A, X)
        else
            return blas_trsm!('L', uploc, trans, isunitc, true, A, X)
        end
    end
    generic_trimatdiv!(C, uploc, isunitc, tfun, A, B)
end
function LinearAlgebra.generic_mattridiv!(C::AnyStridedGPUMatrix, uploc, isunitc, tfun::Function, A::AnyGPUOperand, B::AnyStridedGPUMatrix)
    if allequal(map(eltype, (C, A, B))) && size(C) == size(A) && blas_layout(B) && blas_layout(C)
        return blas_trsm!('R', uploc, trans_char(tfun), isunitc, true, B, operand_copyto!(C, A))
    end
    generic_mattridiv!(C, uploc, isunitc, tfun, A, B)
end

# LinearAlgebra's generic inverse of a triangular matrix uses scalar indexing
for t in (:LowerTriangular, :UnitLowerTriangular, :UpperTriangular, :UnitUpperTriangular)
    @eval function LinearAlgebra.inv(A::$t{T,<:AnyStridedGPUMatrix{T}}) where {T}
        S = typeof(inv(oneunit(T)))
        out = fill!(similar(parent(A), S, size(A)), zero(S))
        view(out, diagind(out)) .= one(S)
        $t(LinearAlgebra.ldiv!(A, out))
    end
end

function generic_rmul!(X::AbstractArray, s::Number)
    @kernel function rmul_kernel!(X, s)
        i = @index(Global, Linear)
        @inbounds X[i] *= s
    end
    rmul_kernel!(get_backend(X))(X, s; ndrange = size(X))
    return X
end

function LinearAlgebra.rmul!(A::AnyStridedGPUArray, b::Number)
    b === true && return A
    # false is a strong zero, which BLAS libraries don't implement
    b === false && return fill!(A, zero(eltype(A)))
    blas_scal!(b, A)
    return A
end
LinearAlgebra.rmul!(A::Diagonal{T, <:AbstractGPUArray}, b::Number) where {T} = generic_rmul!(A.diag, b)

function generic_lmul!(s::Number, X::AbstractArray)
    @kernel function lmul_kernel!(X, s)
        i = @index(Global, Linear)
        @inbounds X[i] = s*X[i]
    end
    lmul_kernel!(get_backend(X))(X, s; ndrange = size(X))
    return X
end

LinearAlgebra.lmul!(a::Number, B::AbstractGPUArray) = generic_lmul!(a, B)
LinearAlgebra.lmul!(a::Number, B::Diagonal{T, <:AbstractGPUArray}) where {T} = generic_lmul!(a, B.diag)


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
    # not every back-end supports launching an empty kernel (JuliaGPU/Metal.jl#980)
    isempty(dest) && return dest
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

LinearAlgebra.norm(v::AnyGPUArray, p::Real=2) =
    p == 2 && v isa AnyStridedGPUArray ? blas_nrm2(v) : generic_norm(v, p)

function LinearAlgebra.norm(D::Diagonal{<:Any,<:AnyGPUVector}, p::Real=2)
    r = norm(D.diag, p)
    # the off-diagonal zeros decide negative norms
    size(D, 1) > 1 && p < 0 ? zero(r) : r
end

function generic_norm(v::AbstractArray{T}, p::Real) where {T}
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

## normalize
@static if VERSION < v"1.14.0-DEV.2264" # no longer needed as of JuliaLang/LinearAlgebra#1634
# Avoid `first(a)` scalar indexing in LinearAlgebra.normalize (JuliaGPU/CUDA.jl#3097)
function LinearAlgebra.normalize(a::AnyGPUArray, p::Real=2)
    nrm = norm(a, p)
    if !isempty(a)
        T = typeof(zero(eltype(a))/nrm)
        aa = LinearAlgebra.copymutable_oftype(a, T)
        return LinearAlgebra.__normalize!(aa, nrm)
    else
        T = typeof(zero(eltype(a))/nrm)
        return T[]
    end
end
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

function LinearAlgebra.rotate!(x::AnyStridedGPUArray, y::AnyStridedGPUArray, c::Number, s::Number)
    length(x) == length(y) ||
        throw(DimensionMismatch(lazy"rotate arguments have lengths $(length(x)) and $(length(y))"))
    # false is a strong zero, which BLAS libraries don't implement
    c isa Bool || s isa Bool ? generic_rotate!(x, y, c, s) : blas_rot!(x, y, c, s)
    return x, y
end

function generic_rotate!(x, y, c, s)
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

function LinearAlgebra.reflect!(x::AnyStridedGPUArray, y::AnyStridedGPUArray, c::Number, s::Number)
    length(x) == length(y) ||
        throw(DimensionMismatch(lazy"reflect arguments have lengths $(length(x)) and $(length(y))"))
    @kernel function reflect_kernel!(x, y, c, s)
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

function gpu_dot(x, y)
    length(x) == length(y) ||
        throw(DimensionMismatch(lazy"dot product arguments have lengths $(length(x)) and $(length(y))"))
    isempty(x) && return dot(zero(eltype(x)), zero(eltype(y)))
    blas_dot(x, y)
end
LinearAlgebra.dot(x::AnyGPUArray, y::AnyGPUArray) = gpu_dot(x, y)
# more specific than LinearAlgebra's methods for wrappers
LinearAlgebra.dot(x::Adjoint{<:Union{Real,Complex},<:AbstractGPUArray}, y::Adjoint{<:Union{Real,Complex},<:AbstractGPUArray}) =
    conj(dot(parent(x), parent(y)))
for W in (:Symmetric, :Hermitian)
    @eval LinearAlgebra.dot(A::$W{<:Union{Real,Complex},<:AbstractGPUMatrix}, B::$W{<:Union{Real,Complex},<:AbstractGPUMatrix}) =
        gpu_dot(A, B)
end
# more specific than LinearAlgebra's methods for host BLAS
const FastGPUArray{T} = Union{AbstractGPUArray{T}, SubArray{T,<:Any,<:AbstractGPUArray,<:Any,true}}
LinearAlgebra.dot(x::FastGPUArray{T}, y::FastGPUArray{T}) where {T<:LinearAlgebra.BlasReal} = gpu_dot(x, y)
LinearAlgebra.dot(x::FastGPUArray{T}, y::FastGPUArray{T}) where {T<:LinearAlgebra.BlasComplex} = gpu_dot(x, y)

function Base.:(*)(x::Transpose{T,<:AnyStridedGPUVector{T}}, y::AnyStridedGPUVector{T}) where {T<:Complex}
    length(x) == length(y) ||
        throw(DimensionMismatch(lazy"dot product arguments have lengths $(length(x)) and $(length(y))"))
    isempty(y) && return zero(T)
    blas_dotu(parent(x), y)
end

LinearAlgebra.BLAS.asum(x::AnyStridedGPUArray{<:LinearAlgebra.BlasFloat}) = blas_asum(x)

## axp{b}y

function LinearAlgebra.axpby!(α::Number, x::AnyStridedGPUArray, β::Number, y::AnyStridedGPUArray)
    length(x) == length(y) ||
        throw(DimensionMismatch(lazy"axpby arguments have lengths $(length(x)) and $(length(y))"))
    α === false && return rmul!(y, β)
    β === false && return axpy!(α, x, fill!(y, zero(eltype(y))))
    blas_axpby!(α, x, β, y)
    return y
end

function LinearAlgebra.axpy!(α::Number, x::AnyStridedGPUArray, y::AnyStridedGPUArray)
    length(x) == length(y) ||
        throw(DimensionMismatch(lazy"axpy arguments have lengths $(length(x)) and $(length(y))"))
    α === false && return y
    blas_axpy!(α, x, y)
    return y
end

## Givens rotations

function LinearAlgebra.lmul!(G::LinearAlgebra.Givens, A::AnyStridedGPUVecOrMat)
    G.i2 > size(A, 1) && throw(DimensionMismatch("column indices for rotation are outside the matrix"))
    rotate!(view(A, G.i1, :), view(A, G.i2, :), real(G.c), G.s)
    return A
end

function LinearAlgebra.rmul!(A::AnyStridedGPUMatrix, G::LinearAlgebra.Givens)
    G.i2 > size(A, 2) && throw(DimensionMismatch("column indices for rotation are outside the matrix"))
    rotate!(view(A, :, G.i1), view(A, :, G.i2), real(G.c), -conj(G.s))
    return A
end

## identity and zero equality check

Base.iszero(x::AbstractGPUMatrix{T}) where {T} = all(iszero, x)
function Base.isone(x::AbstractGPUMatrix{T}) where {T}
    n,m = size(x)
    m != n && return false

    # lazily perform `x-I`
    bc = Broadcast.broadcasted(x, CartesianIndices(x)) do _x, inds
        _x - (inds[1] == inds[2] ? one(_x) : zero(_x))
    end
    # reduce the Broadcasted object directly
    _ak_mapreduce(iszero, &, Broadcast.instantiate(bc); backend=get_backend(x), init=true)
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

@kernel function kron_diag_dense_kernel!(C, @Const(a), @Const(B))
    ci, cj = @index(Global, NTuple)
    mb = size(B, 1)
    nb = size(B, 2)
    i = fld1(ci, mb)
    bi = mod1(ci, mb)
    j = fld1(cj, nb)
    bj = mod1(cj, nb)
    @inbounds C[ci, cj] = (i == j) ? a[i] * B[bi, bj] : zero(eltype(C))
end

@kernel function kron_dense_diag_kernel!(C, @Const(A), @Const(b))
    ci, cj = @index(Global, NTuple)
    nb = length(b)
    i = fld1(ci, nb)
    bi = mod1(ci, nb)
    j = fld1(cj, nb)
    bj = mod1(cj, nb)
    @inbounds C[ci, cj] = (bi == bj) ? A[i, j] * b[bi] : zero(eltype(C))
end

function LinearAlgebra.kron!(C::AbstractGPUMatrix, A::Diagonal{T1, <:AbstractGPUVector}, B::AbstractGPUMatrix{T2}) where {T1, T2}
    size(C) == (length(A.diag) * size(B, 1), length(A.diag) * size(B, 2)) || throw(DimensionMismatch())
    backend = KernelAbstractions.get_backend(C)
    kron_diag_dense_kernel!(backend)(C, A.diag, B, ndrange = size(C))
    return C
end

function LinearAlgebra.kron(A::Diagonal{T1, <:AbstractGPUVector}, B::AbstractGPUMatrix{T2}) where {T1, T2}
    T = promote_type(T1, T2)
    return kron!(similar(B, T, length(A.diag) * size(B, 1), length(A.diag) * size(B, 2)), A, B)
end

function LinearAlgebra.kron!(C::AbstractGPUMatrix, A::AbstractGPUMatrix{T1}, B::Diagonal{T2, <:AbstractGPUVector}) where {T1, T2}
    size(C) == (size(A, 1) * length(B.diag), size(A, 2) * length(B.diag)) || throw(DimensionMismatch())
    backend = KernelAbstractions.get_backend(C)
    kron_dense_diag_kernel!(backend)(C, A, B.diag, ndrange = size(C))
    return C
end

function LinearAlgebra.kron(A::AbstractGPUMatrix{T1}, B::Diagonal{T2, <:AbstractGPUVector}) where {T1, T2}
    T = promote_type(T1, T2)
    return kron!(similar(A, T, size(A, 1) * length(B.diag), size(A, 2) * length(B.diag)), A, B)
end

function LinearAlgebra.kron!(C::Diagonal{<:Any, <:AbstractGPUVector}, A::Diagonal{T1, <:AbstractGPUVector}, B::Diagonal{T2, <:AbstractGPUVector}) where {T1, T2}
    kron!(C.diag, A.diag, B.diag)
    return C
end

function LinearAlgebra.kron(A::Diagonal{T1, <:AbstractGPUVector}, B::Diagonal{T2, <:AbstractGPUVector}) where {T1, T2}
    Diagonal(kron(A.diag, B.diag))
end
