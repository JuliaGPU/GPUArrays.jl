# Workarounds for Statistics.jl methods that would work on GPU arrays as they are, but index
# into them (to take a type from `first(A)`, or in a loop over the elements), which GPU arrays
# only allow as slow scalar indexing. Each section names the Statistics.jl pull request that
# fixes its methods, and can be deleted as a whole once GPUArrays requires a Statistics.jl
# release with that fix. Julia 1.10 always loads the Statistics.jl it ships with, so that
# means dropping 1.10; until then, the sections can at most be skipped on newer Statistics.jl
# with `pkgversion(Statistics)`. The helpers defined here are only used here.


## JuliaStats/Statistics.jl#223: `mean` infers its accumulator type instead of calling `f` on
## `first(A)`

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


## JuliaStats/Statistics.jl#224: `cov` infers the type of its scaled result instead of taking it
## from `first(C)`

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


## JuliaStats/Statistics.jl#225: `cor` of two dense arrays normalizes with a broadcast instead of
## a loop

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
