using Statistics

# Two passes, as Statistics.jl: `mean` first, then the sum of `abs2(a - m)`, fused into one
# reduction. The result has the real floating-point type of `A`'s elements, as Statistics.jl's
# `varm!`, whatever the type of the supplied means.
function Statistics.varm(A::AbstractGPUArray{<:Number}, M::AbstractArray{<:Number};
                         dims=:, corrected::Bool=true)
    # (Statistics.jl throws a MethodError here, from subtracting the array of means from each
    # element)
    dims isa Colon && throw(ArgumentError("`varm` with an array of means needs `dims`"))
    T = real(float(eltype(A)))
    R = Base.reduced_indices(A, dims)
    # an empty slice has no variance; Statistics.jl's `varm!` returns NaN, without checking `M`
    isempty(A) && return fill!(similar(A, T, map(length, R)), convert(T, NaN))
    for i in 1:max(length(R), ndims(M))
        Ri = i <= length(R) ? R[i] : Base.OneTo(1)
        axes(M, i) == Ri || throw(DimensionMismatch(
            "dimension $i of `mean` should have indices $Ri, but got $(axes(M, i))"))
    end
    # (trailing singleton dimensions of `M` don't extend the result)
    ndims(M) > ndims(A) && (M = reshape(M, map(length, R)))
    # (`abs2` rather than `^2`, which promotes to Float64 and uses power_by_squaring)
    # summed in the wider of the result type (so that integers don't overflow) and the type of
    # the deviations (so that wider means keep a Float16 sum from overflowing), and scaled
    # after summing, as by Statistics.jl (so that small Float16 terms don't vanish)
    D = Base.promote_op((a, m) -> abs2(a - m), eltype(A), eltype(M))
    z = zero(isconcretetype(D) ? promote_type(T, D) : T)
    B = Broadcast.broadcasted((a, m) -> oftype(z, abs2(a - m)), A, M)
    S = sum(Broadcast.instantiate(B); dims)
    λ = convert(T, inv(_mean_denom(A, dims) - corrected))
    return eltype(S) === T ? (S .*= λ) : broadcast(s -> oftype(λ, s) * λ, S)
end

Statistics.stdm(A::AbstractGPUArray{<:Number}, m::AbstractArray{<:Number}, dim::Int; corrected::Bool=true) =
    sqrt.(varm(A,m;dims=dim,corrected=corrected))

Statistics._std(A::AbstractGPUArray, corrected::Bool, mean, dims) =
    sqrt.(Statistics.var(A; corrected=corrected, mean=mean, dims=dims))

Statistics._std(A::AbstractGPUArray, corrected::Bool, mean, ::Colon) =
    sqrt.(Statistics.var(A; corrected=corrected, mean=mean, dims=:))

# Like Statistics.jl, accumulate in the floating-point type of the result and divide
# afterwards: summing integers overflows, and scaling each term first loses precision.
# The accumulator type is inferred, as computing it from `first(A)` would need scalar indexing.
function Statistics._mean(f, A::AbstractGPUArray, dims)
    isempty(A) && return sum(f, A; dims) / 0
    S = Base.promote_op(f, eltype(A))
    T = promote_type(Base.promote_op(/, S, Int), S)
    result = sum(_mean_convert(f, T), A; dims)
    if dims isa Colon
        result / length(A)
    else
        result ./= _mean_denom(A, dims)
    end
end

# `f` and `T` are static parameters rather than captured values, so the closure stays isbits
# when `f` is a type (e.g. `mean(Float32, A)`) and can be passed to a kernel.
_mean_convert(f::F, ::Type{T}) where {F,T} = x -> convert(T, f(x))
_mean_convert(::Type{F}, ::Type{T}) where {F,T} = x -> convert(T, F(x))

function Statistics.covzm(x::AbstractGPUMatrix, vardim::Int=1; corrected::Bool=true)
    C = Statistics.unscaled_covzm(x, vardim)
    T = promote_type(typeof(one(eltype(C)) / 1), eltype(C))
    A = convert(AbstractArray{T}, C)
    b = 1//(size(x, vardim) - corrected)
    A .*= b
    return A
end

# (also for two arguments: the type of the scaled result is inferred, not taken from `first(C)`,
# so that without variables the result is an empty matrix rather than a BoundsError)
function Statistics.covzm(x::AbstractGPUVecOrMat, y::AbstractGPUVecOrMat, vardim::Int;
                          corrected::Bool=true)
    C = Statistics.unscaled_covzm(x, y, vardim)
    T = promote_type(typeof(one(eltype(C)) / 1), eltype(C))
    A = convert(AbstractArray{T}, C)
    b = 1//(Statistics._getnobs(x, y, vardim) - corrected)
    A .*= b
    return A
end

function Statistics.cov2cor!(C::AbstractGPUMatrix{T}, xsd::AbstractGPUArray) where T
    nx = length(xsd)
    size(C) == (nx, nx) || throw(DimensionMismatch("inconsistent dimensions"))
    tril!(C, -1)
    C += adjoint(C)
    C = Statistics.clampcor.(C ./ (xsd * xsd'))
    C[diagind(C)] .= oneunit(T)
    return C
end

function Statistics.corzm(x::AbstractGPUMatrix, vardim::Int=1)
    c = Statistics.unscaled_covzm(x, vardim)
    return Statistics.cov2cor!(c, sqrt.(diag(c)))
end

# Statistics.jl's `cov2cor!` for two sets of standard deviations, as one broadcast
# (`xsd` and `ysd` hold one value per row and column of `C`, or are a number)
function _cov2cor!(C, xsd, ysd)
    xcol = xsd isa AbstractArray ? reshape(xsd, :, 1) : xsd
    yrow = ysd isa AbstractArray ? reshape(ysd, 1, :) : ysd
    C .= Statistics.clampcor.(C ./ (xcol .* yrow))
    return C
end

Statistics.corzm(x::AbstractGPUVector, y::AbstractGPUMatrix, vardim::Int=1) =
    _cov2cor!(Statistics.unscaled_covzm(x, y, vardim), sqrt(sum(abs2, x)),
              sqrt.(sum(abs2, y; dims=vardim)))
Statistics.corzm(x::AbstractGPUMatrix, y::AbstractGPUVector, vardim::Int=1) =
    _cov2cor!(Statistics.unscaled_covzm(x, y, vardim), sqrt.(sum(abs2, x; dims=vardim)),
              sqrt(sum(abs2, y)))
Statistics.corzm(x::AbstractGPUMatrix, y::AbstractGPUMatrix, vardim::Int=1) =
    _cov2cor!(Statistics.unscaled_covzm(x, y, vardim), sqrt.(sum(abs2, x; dims=vardim)),
              sqrt.(sum(abs2, y; dims=vardim)))

# The correlation of two vectors, from the three sums Statistics.jl accumulates in one loop,
# here in one reduction, and with its scaling (which keeps `xy / sqrt(xx * yy)` from overflowing)
function Statistics.corm(x::AbstractGPUVector{<:Number}, mx, y::AbstractGPUVector{<:Number}, my)
    n = length(x)
    length(y) == n || throw(DimensionMismatch("inconsistent lengths"))
    n > 0 || throw(ArgumentError("correlation only defined for non-empty vectors"))
    X, Y = eltype(x), eltype(y)
    dX, dY = Base.promote_op(-, X, typeof(mx)), Base.promote_op(-, Y, typeof(my))
    # (the accumulators of Statistics.jl, widened to the type of the terms)
    init = (zero(promote_type(typeof(sqrt(abs2(one(X)))), Base.promote_op(abs2, dX))),
            zero(promote_type(typeof(sqrt(abs2(one(Y)))), Base.promote_op(abs2, dY))),
            zero(promote_type(typeof(zero(X) * zero(Y)'), Base.promote_op((a, b) -> a * b', dX, dY))))
    xx, yy, xy = mapreduce(_corm_terms(mx, my, init), _add_terms, x, y; init)
    return Statistics.clampcor(xy / max(xx, yy) / sqrt(min(xx, yy) / max(xx, yy)))
end
_corm_terms(mx, my, ::Tuple{A,B,C}) where {A,B,C} = function (a, b)
    xi = a - mx
    yi = b - my
    (convert(A, abs2(xi)), convert(B, abs2(yi)), convert(C, xi * yi'))
end
_add_terms(s, t) = map(+, s, t)

_mean_denom(x::AbstractArray, dims::Integer) = size(x, dims)
_mean_denom(x::AbstractArray, dims::Colon) = length(x)
_mean_denom(x::AbstractArray, dims) = prod(size(x,d) for d in unique(dims); init=1)
