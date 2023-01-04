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

# Revert https://github.com/JuliaLang/Statistics.jl/pull/25
Statistics._mean(A::AbstractGPUArray, ::Colon)    = sum(A) / length(A)
Statistics._mean(f, A::AbstractGPUArray, ::Colon) = sum(f, A) / length(A)

function Statistics._mean(A::AbstractGPUArray, dims)
    T = float(eltype(A))
    λ = convert(T, inv(_mean_denom(A, dims)))
    sum(Base.Fix1(*,λ), A; dims)
end
function Statistics._mean(f, A::AbstractGPUArray, dims)
    T = float(eltype(A))
    λ = convert(T, inv(_mean_denom(A, dims)))
    sum(Base.Fix1(*,λ) ∘ f, A; dims)
end

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
