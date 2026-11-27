module StatisticsExt

using GPUArrays
using GPUArrays: AK
using LinearAlgebra
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


## order statistics

# The ranks a median of `n` values needs, the middle one or two, from which AcceleratedKernels
# selects by radix (for more than two ranks it sorts a copy). Whether there are NaNs, which make
# the median NaN, is checked separately: `max` returns a NaN if either argument is one, so the
# `maximum` of a slice is one of its NaNs if it has any. Integers have none, and skip that pass.
_median_ranks(n) = isodd(n) ? [(n + 1) ÷ 2] : [n ÷ 2, n ÷ 2 + 1]
_median_nans(A; dims=:) = eltype(A) <: Integer ? nothing : maximum(A; dims)
_median_value(lo, hi, nan) = ifelse(isnan(nan), oftype(Statistics.middle(lo, hi), nan),
                                    Statistics.middle(lo, hi))
_median_value(mid, nan) = ifelse(isnan(nan), oftype(Statistics.middle(mid), nan),
                                 Statistics.middle(mid))
_median_value(lo, hi, ::Nothing) = Statistics.middle(lo, hi)
_median_value(mid, ::Nothing) = Statistics.middle(mid)

_has_missing(v) = eltype(v) >: Missing && eltype(v) !== Any && any(ismissing, v)

# As Statistics.jl, except that the NaN returned for data containing NaNs is one of them, not
# necessarily the first. `v` is left as it is, which `median!` allows.
function Statistics.median!(v::AnyGPUVector)
    isempty(v) && throw(ArgumentError("median of an empty array is undefined, $(repr(v))"))
    _has_missing(v) && return missing
    nan = _median_nans(v)
    nan !== nothing && isnan(nan) && return nan
    return _median_value(Array(AK.select(v, _median_ranks(length(v))))..., nothing)
end

Statistics._median(A::AbstractGPUArray, ::Colon) = median!(vec(A))

# As Statistics.jl, which uses `mapslices`, except that:
# - an empty result is an empty array, where `mapslices` throws a BoundsError;
# - slices with a `missing` give `missing`, in a `Union{Missing,T}` array, where `mapslices`
#   fails to store it after a slice with a median;
# - as for `median!`, a slice with NaNs gives one of them, not necessarily the first.
#
# The reduced dimensions are first made one, by reshaping when they lead and by permuting the
# array otherwise, to select the same ranks from every slice.
function Statistics._median(A::AbstractGPUArray, dims)
    ds = dims isa Integer ? [dims] : applicable(iterate, dims) ? collect(dims) : nothing
    # (other dimensions get the errors of `mapslices`)
    if ds === nothing || isempty(ds) || !all(d -> d isa Integer && 1 <= d <= ndims(A), ds)
        return invoke(Statistics._median, Tuple{AbstractArray, Any}, A, dims)
    end
    rdims = Tuple(sort!(unique(Int.(ds))))
    kdims = Tuple(d for d in 1:ndims(A) if !(d in rdims))
    n = prod(d -> size(A, d), rdims; init=1)
    m = prod(d -> size(A, d), kdims; init=1)
    outsize = ntuple(d -> d in rdims ? 1 : size(A, d), ndims(A))

    m != 0 && n == 0 &&
        throw(ArgumentError("median of an empty array is undefined, $(repr(similar(A, eltype(A), 0)))"))
    nomissing = nonmissingtype(eltype(A))
    nomissing === Union{} && return fill!(similar(A, Missing, outsize), missing)
    missings = eltype(A) >: Missing && eltype(A) !== Any ? any(ismissing, A; dims=rdims) : nothing
    if missings !== nothing && !any(missings)
        missings = nothing
    end
    z = zero(nomissing)
    T = typeof(isodd(n) ? Statistics.middle(z) : Statistics.middle(z, z))
    m == 0 && return similar(A, missings === nothing ? T : Union{Missing, T}, outsize)

    # without `missing` (from `coalesce`, which unlike `convert` cannot throw) and with the
    # reduced dimensions combined
    src = nomissing === eltype(A) ? A : coalesce.(A, zero(nomissing))
    permute = length(rdims) > 1 && rdims != ntuple(identity, length(rdims))
    permute && (src = permutedims(src, (rdims..., kdims...)))
    sdims = length(rdims) == 1 ? only(rdims) : 1
    length(rdims) == 1 || (src = reshape(src, n, m))

    S = AK.select(src, _median_ranks(n); dims=sdims)
    slots = ntuple(i -> selectdim(S, sdims, i:i), size(S, sdims))
    R = reshape(broadcast(_median_value, slots..., _median_nans(src; dims=sdims)), outsize)
    return missings === nothing ? R : ifelse.(missings, missing, R)
end

# Quantiles use Statistics.jl's own arithmetic, `_quantile`, on stand-ins for the sorted data:
# `_RankProbe` records the order statistics it reads (they depend only on the length of the
# data and on `p`, `alpha` and `beta`, and it checks those), and once they have been gathered
# from the device, `_RankValues` answers with them.
struct _RankProbe{T} <: AbstractVector{T}
    n::Int
    ranks::Vector{Int}
end
Base.size(v::_RankProbe) = (v.n,)
Base.getindex(v::_RankProbe{T}, i::Int) where {T} = (push!(v.ranks, i); zero(T))

struct _RankValues{T} <: AbstractVector{T}
    n::Int
    values::Dict{Int,T}
end
Base.size(v::_RankValues) = (v.n,)
Base.getindex(v::_RankValues, i::Int) = v.values[i]

_isnan_or_missing(x) = ismissing(x) || (x isa Number && isnan(x))

# The quantiles of `v` at every probability of `ps` (a tuple or a host array), as a host
# collection like `ps`, with the bounds and NaN or missing checks of Statistics.jl in its
# order. The order statistics are read from `v` if it is already sorted (`how == :sorted`),
# after sorting it in place (`:sort`, for `quantile!`, which documents that it sorts `v`), or
# by selecting them, leaving `v` as it is (`:select`).
function _quantile_values!(v::AnyGPUVector, ps, how::Symbol, alpha::Real, beta::Real)
    isempty(v) && throw(ArgumentError("empty data vector"))
    n = length(v)
    T = nonmissingtype(eltype(v))
    if how !== :sorted
        # (the bounds of Statistics.jl's partial sort, which reject a NaN probability first)
        lo, hi = extrema(ps)
        floor(Int, lo * n), ceil(Int, 1 + hi * n)
    end
    # the ranks to read, and the error for an invalid probability, which is reported after
    # NaN or missing data
    ranks = Int[]
    err = try
        foreach(p -> Statistics._quantile(_RankProbe{T}(n, ranks), p; alpha, beta), ps)
        nothing
    catch err
        err isa InterruptException && rethrow()
        err
    end
    if how === :sorted
        values = Array(v[ranks])
        nan = any(_isnan_or_missing, v)
    else
        if how === :sort
            AK.sort!(v; alg=AK.Auto(stable=false))
            # (by `isless`, the last value is NaN or missing if any is)
            values = Array(v[[ranks; n]])
            nan = _isnan_or_missing(pop!(values))
        else
            # (AcceleratedKernels selects one or two distinct ranks, as for one probability, by
            # radix, and sorts a copy for more)
            ranks = unique(ranks)
            values = Array(AK.select(v, ranks))
            nan = any(_isnan_or_missing, v)
        end
    end
    nan && throw(ArgumentError("quantiles are undefined in presence of NaNs or missing values"))
    err === nothing || throw(err)
    x = _RankValues{T}(n, Dict{Int,T}(zip(ranks, values)))
    return map(p -> Statistics._quantile(x, p; alpha, beta), ps)
end

# The quantiles for a probability, a tuple of them, or an array of them, which as with
# Statistics.jl determines the container of the result: for an array, a GPU array if it is on
# the device, a host array otherwise.
_quantiles!(v, p::Real, how, alpha, beta) = only(_quantile_values!(v, (p,), how, alpha, beta))
_quantiles!(v, p::Tuple, how, alpha, beta) =
    isempty(p) ? () : _quantile_values!(v, p, how, alpha, beta)
function _quantiles!(v, p::AbstractArray, how, alpha, beta)
    # (the interpolation runs on the host)
    ps = p isa AnyGPUArray ? Array(p) : p
    q = if isempty(ps)
        # (Statistics.jl maps over `p`, without checking `v`)
        similar(ps, Base.promote_op(Statistics._quantile, Vector{eltype(v)}, eltype(ps)))
    else
        _quantile_values!(v, ps, how, alpha, beta)
    end
    return p isa AnyGPUArray ? copyto!(similar(p, eltype(q)), q) : q
end

Statistics.quantile!(v::AnyGPUVector, p::Real; sorted::Bool=false, alpha::Real=1.0,
                     beta::Real=alpha) =
    _quantiles!(v, p, sorted ? :sorted : :sort, alpha, beta)
Statistics.quantile!(v::AnyGPUVector, p::Union{AbstractArray, Tuple{Vararg{Real}}};
                     sorted::Bool=false, alpha::Real=1.0, beta::Real=alpha) =
    _quantiles!(v, p, sorted ? :sorted : :sort, alpha, beta)
function Statistics.quantile!(q::AbstractArray, v::AnyGPUVector, p::AbstractArray;
                              sorted::Bool=false, alpha::Real=1.0, beta::Real=alpha)
    Base.require_one_based_indexing(q, v, p)
    if size(p) != size(q)
        throw(DimensionMismatch("size of p, $(size(p)), must equal size of q, $(size(q))"))
    end
    isempty(q) && return q
    r = _quantile_values!(v, Array(p), sorted ? :sorted : :sort, alpha, beta)
    return copyto!(q, convert(Array{eltype(q)}, r))
end

# (Statistics.jl collects other arrays than vectors, which copies them to the host)
_quantiles(v, p, sorted, alpha, beta) = _quantiles!(v, p, sorted ? :sorted : :select, alpha, beta)
Statistics.quantile(v::AbstractGPUVector, p::Real; sorted::Bool=false, alpha::Real=1.0,
                    beta::Real=alpha) =
    _quantiles(v, p, sorted, alpha, beta)
Statistics.quantile(A::AbstractGPUArray, p::Union{AbstractArray, Tuple{Vararg{Real}}};
                    sorted::Bool=false, alpha::Real=1.0, beta::Real=alpha) =
    _quantiles(vec(A), p, sorted, alpha, beta)

# Overrides that only avoid scalar indexing in Statistics.jl's methods, until fixes upstream are
# released; the file says when each can go.
include("workarounds.jl")

end
