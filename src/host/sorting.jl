# sorting and reversing, implemented by AcceleratedKernels
#
# AcceleratedKernels chooses the algorithm and its settings for the array's device. Base's
# algorithm objects become requirements (`_akalg`); an AcceleratedKernels algorithm passed as
# `alg` is used as given, e.g. `sort!(A; alg=AK.RadixSort())`. `scratch` is accepted and ignored:
# the equivalent for GPU arrays is a workspace, `AK.sort!(v; workspace=AK.workspace(AK.sort!, v))`.
#
# Vector and `dims` forms are separate methods, as in Base: vectors take no `dims`, other arrays
# require it.

# Base's sorting algorithms, as requirements on the algorithm AcceleratedKernels picks
_akalg(::Nothing) = AK.Auto()                   # Base's default is stable
_akalg(alg::AK.Algorithm) = alg
@static if isdefined(Base.Sort, :DefaultStable)     # (Julia 1.11)
    _akalg(::Base.Sort.DefaultStable) = AK.Auto(stable=true)
    _akalg(::Base.Sort.DefaultUnstable) = AK.Auto(stable=false)
else                                                # (`DEFAULT_UNSTABLE` is the same object)
    _akalg(::typeof(Base.Sort.DEFAULT_STABLE)) = AK.Auto(stable=true)
end
_akalg(::Base.Sort.MergeSortAlg) = AK.Auto(stable=true)
_akalg(::Base.Sort.InsertionSortAlg) = AK.Auto(stable=true)
_akalg(::Union{Base.Sort.QuickSortAlg, Base.Sort.PartialQuickSort}) = AK.Auto(stable=false)
_akalg(alg) = throw(ArgumentError(
    "sorting algorithm $alg is not supported on GPU arrays; omit `alg`, or pass one of Base's " *
    "`MergeSort`, `InsertionSort`, `QuickSort` or `PartialQuickSort`, or an AcceleratedKernels algorithm"))

# (vectors take Base's keywords only, so that `dims` is an error, as in Base)
function Base.sort!(v::AnyGPUVector; alg=nothing, lt=isless, by=identity, rev=nothing,
                    order::Base.Order.Ordering=Base.Order.Forward, scratch=nothing)
    AK.sort!(v; alg=_akalg(alg), lt, by, rev, order)
    return v
end
function Base.sort!(A::AnyGPUArray; dims::Integer, alg=nothing, scratch=nothing, kwargs...)
    AK.sort!(A; dims, alg=_akalg(alg), kwargs...)
    return A
end

Base.sort(v::AnyGPUVector; kwargs...) = sort!(copy(v); kwargs...)
Base.sort(A::AnyGPUArray; dims::Integer, kwargs...) = sort!(copy(A); dims, kwargs...)

# AcceleratedKernels always initialises `ix`; `initialized` is accepted and ignored
function Base.sortperm!(ix::AnyGPUArray{<:Integer}, v::AnyGPUVector; alg=nothing,
                        scratch=nothing, initialized::Bool=false, dims=nothing, kwargs...)
    dims === nothing || throw(ArgumentError("sortperm! of a vector does not accept `dims`"))
    axes(ix) == axes(v) ||
        throw(ArgumentError("index array must have the same axes as the source array"))
    AK.sortperm!(ix, v; alg=_akalg(alg), kwargs...)
    return ix
end
function Base.sortperm!(ix::AnyGPUArray{<:Integer}, A::AnyGPUArray; dims::Integer, alg=nothing,
                        scratch=nothing, initialized::Bool=false, kwargs...)
    axes(ix) == axes(A) ||
        throw(ArgumentError("index array must have the same axes as the source array"))
    AK.sortperm!(ix, A; dims, alg=_akalg(alg), kwargs...)
    return ix
end

Base.sortperm(v::AnyGPUVector; kwargs...) = sortperm!(similar(v, Int), v; kwargs...)
Base.sortperm(A::AnyGPUArray; dims::Integer, kwargs...) =
    sortperm!(similar(A, Int), A; dims, kwargs...)

# By sorting all of `v`; as Base, an integer `k` gives the element and a range a view
function Base.partialsort!(v::AnyGPUVector, k::Union{Integer, OrdinalRange}; kwargs...)
    sort!(v; kwargs...)
    return k isa Integer ? @allowscalar(v[k]) : view(v, k)
end
Base.partialsort!(v::AnyGPUVector, k::Union{Integer, OrdinalRange}, o::Base.Order.Ordering) =
    partialsort!(v, k; order=o)

function Base.reverse!(A::AnyGPUArray; dims=:)
    AK.reverse!(A; dims)
    return A
end
Base.reverse(A::AnyGPUArray; dims=:) = AK.reverse(A; dims)

# Vectors follow Base's rules for `dims`, and reverse through the ranged form
Base.reverse!(v::AnyGPUVector; dims=:) = invoke(reverse!, Tuple{AbstractVector}, v; dims)
Base.reverse(v::AnyGPUVector; dims=:) = invoke(reverse, Tuple{AbstractVector}, v; dims)
function Base.reverse!(v::AnyGPUVector, start::Integer, stop::Integer=lastindex(v))
    s, n = Int(start), Int(stop)
    n > s || return v       # as Base, a trivial interval is not checked
    checkbounds(v, s)
    checkbounds(v, n)
    AK.reverse!(view(v, s:n))
    return v
end
Base.reverse(v::AnyGPUVector, start::Integer, stop::Integer=lastindex(v)) =
    reverse!(copy(v), start, stop)
