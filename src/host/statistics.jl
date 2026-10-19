using Statistics

function Statistics.varm(A::AbstractGPUArray{<:Real}, M::AbstractArray{<:Real};
                         dims, corrected::Bool=true)
    T = float(eltype(A))
    λ = convert(T, inv(_mean_denom(A, dims) - corrected))
    #B = (A .- M).^2
    # NOTE: the above broadcast promotes to Float64 and uses power_by_squaring...
    B = Broadcast.broadcasted(A, M) do a, m
        x = (a - m)
        λ * x * x
    end
    sum(Broadcast.instantiate(B); dims)
end

Statistics.stdm(A::AbstractGPUArray{<:Real},m::AbstractArray{<:Real}, dim::Int; corrected::Bool=true) =
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

_mean_denom(x::AbstractArray, dims::Integer) = size(x, dims)
_mean_denom(x::AbstractArray, dims::Colon) = length(x)
_mean_denom(x::AbstractArray, dims) = prod(size(x,d) for d in unique(dims); init=1)
