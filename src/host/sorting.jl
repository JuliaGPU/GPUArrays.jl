
abstract type SortingAlgorithm end
struct MergeSortAlg <: SortingAlgorithm end

const MergeSort = MergeSortAlg()


function Base.sort!(c::AnyGPUVector, alg::MergeSortAlg; lt=isless, by=identity, rev=false)
    # for reverse sorting, invert the less-than function
    if rev
        lt = !lt
    end

    AK.merge_sort!(c; lt, by)
    return c
end

function Base.sort!(c::AnyGPUArray; alg::SortingAlgorithm = MergeSort, kwargs...)
    return sort!(c, alg; kwargs...)
end

function Base.sort(c::AnyGPUArray; kwargs...)
    return sort!(copy(c); kwargs...)
end

function Base.partialsort!(c::AnyGPUVector, k::Union{Integer, OrdinalRange},
                           alg::MergeSortAlg; lt=isless, by=identity, rev=false)

    sort!(c, alg; lt, by, rev)
    return @allowscalar copy(c[k])
end

function Base.partialsort!(c::AnyGPUArray, k::Union{Integer, OrdinalRange};
                           alg::SortingAlgorithm=MergeSort, kwargs...)
    return partialsort!(c, k, alg; kwargs...)
end

function Base.partialsort(c::AnyGPUArray, k::Union{Integer, OrdinalRange}; kwargs...)
    return partialsort!(copy(c), k; kwargs...)
end

function Base.sortperm!(ix::AnyGPUArray, A::AnyGPUArray; initialized=false, dims=nothing, kwargs...)
    if axes(ix) != axes(A)
        throw(ArgumentError("index array must have the same size/axes as the source array, $(axes(ix)) != $(axes(A))"))
    end
    if !isnothing(dims)
        throw(ArgumentError("GPUArrays sort with `dims` kwarg not yet implemented."))
    end

    AK.merge_sortperm!(ix, A; kwargs...)
    return ix
end

function Base.sortperm(c::AnyGPUVector; initialized=false, kwargs...)
    AK.merge_sortperm!(KA.allocate(get_backend(c), Int, length(c)), c; kwargs...)
end

function Base.sortperm(c::AnyGPUArray; dims, kwargs...)
    # Base errors for Matrices without dims arg, we should too
    error("GPU sort with `dims` kwarg not yet implemented.")
    # sortperm!(reshape(adapt(get_backend(c), collect(1:length(c))), size(c)), c; initialized=true, dims, kwargs...)
end
