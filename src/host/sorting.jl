# sorting

# GPUArrays owns the algorithm selection and delegates a concrete algorithm to AcceleratedKernels.
# AK is the algorithm library (merge / radix / bitonic sort as explicit kernels); this layer picks
# the one expected to be fastest for the input and passes it as `alg`. Passing `alg` explicitly skips
# the selection. `dims` is forwarded only when set, so the whole-array path keeps working against
# AcceleratedKernels releases that predate per-dimension sorting.

# Element types and orderings that AK's radix and bitonic sorts can handle. Everything else (custom
# comparators, non-bits eltypes) goes to merge sort, which is fully general.
@inline _bits_sortable(::Type{T}) where {T} =
    T === Int32 || T === UInt32 || T === Float32 ||
    T === Int64 || T === UInt64 || T === Float64
@inline _plain_order(lt, by, order) =
    lt === isless && by === identity && order === Base.Order.Forward

# Largest slice a single-workgroup bitonic sort can hold, from the device's max work-group size
# queried through KernelInterface (`KI.max_work_group_size`). This generalises the ceiling across
# devices with a real device query instead of a constant. Backends that do not yet implement
# KernelInterface fall back to AcceleratedKernels' shared-memory budget divided by the element size.
@inline function _bitonic_ceiling(v::AnyGPUArray)
    try
        return KI.max_work_group_size(KI.get_backend(v))
    catch
        return AK._prevpow2(AK._bs_shmem_bytes(get_backend(v)) ÷ sizeof(eltype(v)))
    end
end

# Flat sorts use radix (fastest at the sizes that matter); per-slice (`dims`) sorts use bitonic while
# a slice fits one workgroup and merge above that ceiling (segmented radix is not implemented yet).
# Anything radix/bitonic cannot express (custom comparator, non-bits eltype) falls back to merge.
function _select_sort_alg(v::AnyGPUArray, dims, lt, by, order)
    bits_plain = _bits_sortable(eltype(v)) && _plain_order(lt, by, order)
    if dims === Colon()
        return bits_plain ? AK.RadixSort() : AK.MergeSort()
    end
    if bits_plain && size(v, dims) <= _bitonic_ceiling(v)
        return AK.BitonicSort()
    end
    return AK.MergeSort()
end

function Base.sort!(v::AnyGPUArray; dims=:, alg=nothing, lt=isless, by=identity,
                    rev::Bool=false, order::Base.Order.Ordering=Base.Order.Forward)
    chosen = alg === nothing ? _select_sort_alg(v, dims, lt, by, order) : alg
    if dims === Colon()
        AK.sort!(v; alg=chosen, lt, by, rev, order)
    else
        AK.sort!(v; dims, alg=chosen, lt, by, rev, order)
    end
    return v
end

Base.sort(v::AnyGPUArray; kwargs...) = Base.sort!(copy(v); kwargs...)

function Base.sortperm!(ix::AnyGPUArray, v::AnyGPUArray; dims=:, alg=nothing, lt=isless,
                        by=identity, rev::Bool=false,
                        order::Base.Order.Ordering=Base.Order.Forward, initialized::Bool=false)
    axes(ix) == axes(v) ||
        throw(ArgumentError("index array must have the same axes as the array being sorted"))
    if dims === Colon()
        AK.sortperm!(ix, v; alg, lt, by, rev, order)
    else
        AK.sortperm!(ix, v; dims, alg, lt, by, rev, order)
    end
    return ix
end

function Base.sortperm(v::AnyGPUArray; kwargs...)
    ix = similar(v, Int)
    Base.sortperm!(ix, v; kwargs...)
end

# partialsort by sorting then reading back the requested position(s)
function Base.partialsort!(v::AnyGPUVector, k::Union{Integer, OrdinalRange}; kwargs...)
    Base.sort!(v; kwargs...)
    return @allowscalar copy(v[k])
end

Base.partialsort(v::AnyGPUVector, k::Union{Integer, OrdinalRange}; kwargs...) =
    Base.partialsort!(copy(v), k; kwargs...)
