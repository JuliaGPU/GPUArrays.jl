# sparse-specific primitives, as internal kernels
#
# These build on the Base verbs a back-end provides for its dense arrays (`sortperm`,
# `accumulate!`, `findall`, `mapreduce`, and gathers through `getindex`). They are
# candidates for AcceleratedKernels once proven.
#
# The sparse code never launches a kernel over an empty range, because not every back-end
# supports that (JuliaGPU/Metal.jl#980).

# release a temporary buffer early, without waiting for the GC
free_buffer!(x::AbstractGPUArray) = unsafe_free!(x)
free_buffer!(x) = nothing

## COV_EXCL_START
@kernel function expand_ptr_kernel(major, ptr)
    k = @index(Global, Linear)
    if k <= length(major)
        @inbounds major[k] = searchsortedlast(ptr, k % eltype(ptr)) % eltype(major)
    end
end

@kernel function compress_kernel(ptr, major)
    j = @index(Global, Linear)
    if j <= length(ptr)
        @inbounds ptr[j] = searchsortedfirst(major, j % eltype(major)) % eltype(ptr)
    end
end

@kernel function packed_key_kernel(keys, major, minor, nminor)
    k = @index(Global, Linear)
    if k <= length(keys)
        @inbounds keys[k] = (Int64(major[k]) - 1) * nminor + Int64(minor[k])
    end
end
## COV_EXCL_STOP

"""
    expand_ptr(ptr, nnz)

The slice index of each of the `nnz` stored entries of a compressed pointer buffer `ptr`
(e.g. the row of every entry of a CSR matrix), with the index type and storage of `ptr`.
Every stored entry is handled by its own thread, which finds its slice by binary search,
so that long slices do not serialize.
"""
function expand_ptr(ptr::AbstractVector{Ti}, nnz::Integer) where {Ti}
    major = similar(ptr, Ti, nnz)
    if nnz > 0
        expand_ptr_kernel(get_backend(ptr))(major, ptr; ndrange=nnz)
    end
    return major
end

"""
    compress(major, n)

The compressed pointer buffer, of length `n+1`, for stored entries whose sorted slice
indices are `major`: the inverse of [`expand_ptr`](@ref). Every pointer is found by binary
search; an atomic histogram followed by a scan would avoid the searches.
"""
function compress(major::AbstractVector{Ti}, n::Integer) where {Ti}
    ptr = similar(major, Ti, n + 1)
    compress_kernel(get_backend(ptr))(ptr, major; ndrange=n + 1)
    return ptr
end

# Sort keys ordering entries by (major, minor) index. They are packed into a signed
# `Int64`, which fits for all practical sizes and which every back-end can sort on the
# device (Metal's `sortperm` does not support unsigned integers).
function packed_keys(major::AbstractVector, minor::AbstractVector, dims::Dims{2})
    nmajor, nminor = dims
    widemul(nmajor, nminor) < typemax(Int64) ||
        throw(ArgumentError("sparse matrices with $nmajor × $nminor entries are too large to sort"))
    keys = similar(major, Int64, length(major))
    if !isempty(keys)
        packed_key_kernel(get_backend(keys))(keys, major, minor, Int64(nminor);
                                             ndrange=length(keys))
    end
    return keys
end

# Regroup the stored entries of a compressed matrix along its other dimension: the CSC
# buffers of a CSR matrix, and vice versa. `dims` is (major, minor) of the input.
#
# The entries are sorted by (minor, major) and gathered. An atomic counting sort over the
# minor indices (histogram, scan, scatter) would be much faster than the general sort.
function regroup(ptr::AbstractVector{Ti}, ind::AbstractVector{Ti}, val::AbstractVector,
                 dims::Dims{2}) where {Ti}
    nmajor, nminor = dims
    n = length(val)
    n == 0 && return empty_ptr(ptr, Ti, nminor), similar(ind, 0), similar(val, 0)

    major = expand_ptr(ptr, n)
    keys = packed_keys(ind, major, (nminor, nmajor))
    perm = sortperm(keys)
    free_buffer!(keys)
    new_major = ind[perm]
    new_ind = major[perm]
    new_val = val[perm]
    free_buffer!(major)
    free_buffer!(perm)
    new_ptr = compress(new_major, nminor)
    free_buffer!(new_major)
    return new_ptr, new_ind, new_val
end

## COV_EXCL_START
@kernel function run_heads_kernel(heads, keys)
    k = @index(Global, Linear)
    if k <= length(keys)
        @inbounds heads[k] = k == 1 || keys[k] != keys[k-1]
    end
end

# every thread folds one run of equal keys, in order
@kernel function reduce_runs_kernel(out_keys, out_vals, op, starts, keys, vals)
    r = @index(Global, Linear)
    if r <= length(starts)
        first = @inbounds starts[r]
        last = r == length(starts) ? length(vals) : @inbounds(starts[r+1]) - 1
        acc = @inbounds vals[first]
        for k in first+1:last
            acc = convert(eltype(out_vals), op(acc, @inbounds vals[k]))
        end
        @inbounds out_keys[r] = keys[first]
        @inbounds out_vals[r] = acc
    end
end
## COV_EXCL_STOP

# whether each of the sorted `keys` starts a run of equal keys
function run_heads(keys::AbstractVector)
    heads = similar(keys, Bool)
    isempty(keys) || run_heads_kernel(get_backend(keys))(heads, keys; ndrange=length(keys))
    return heads
end

"""
    reduce_by_key(op, keys, vals)

Combine the values of every run of equal, sorted `keys` with `op`, folding each run from
left to right. Returns the unique keys and the combined values. Every run is folded by
one thread, so many repetitions of a key serialize; a tiled segmented scan would not.
"""
function reduce_by_key(op, keys::AbstractVector, vals::AbstractVector)
    starts = findall(run_heads(keys))
    n = length(starts)
    out_keys = similar(keys, n)
    out_vals = similar(vals, n)
    if n > 0
        reduce_runs_kernel(get_backend(vals))(out_keys, out_vals, op, starts, keys, vals;
                                              ndrange=n)
    end
    free_buffer!(starts)
    return out_keys, out_vals
end
