# assembly of sparse arrays from coordinates

## COV_EXCL_START
# a sort key that is unique per input entry, ordering the runs of equal coordinates and,
# within each run, the entries by their position in the input
@kernel function positional_key_kernel(keys, runs, perm, n)
    k = @index(Global, Linear)
    if k <= length(keys)
        @inbounds keys[k] = (Int64(runs[k]) - 1) * n + (Int64(perm[k]) - 1)
    end
end

@kernel function unpack_key_kernel(major, minor, keys, nminor)
    k = @index(Global, Linear)
    if k <= length(keys)
        key = @inbounds keys[k] - 1
        @inbounds major[k] = (key ÷ nminor + 1) % eltype(major)
        @inbounds minor[k] = (key % nminor + 1) % eltype(minor)
    end
end
## COV_EXCL_STOP

# whether the order in which `combine` folds repeated entries of type `T` cannot change
# the result, so that the assembly does not need to restore the input order after an
# unstable sort. Floating-point `+` and `*` are commutative but not associative.
order_independent(combine, ::Type) = false
order_independent(::Union{typeof(max), typeof(min), typeof(|), typeof(&)}, ::Type) = true
order_independent(::Union{typeof(+), typeof(*)}, ::Type{<:Union{Integer,Complex{<:Integer}}}) = true

# the order in which to visit the entries so that equal keys are contiguous and appear in
# their input order. `sortperm` need not be stable (Metal's is not), so the input order is
# restored with a second sort on keys that are unique per entry.
function stable_sortperm(keys::AbstractVector{Int64}; stable::Bool)
    perm = sortperm(keys)
    stable || return perm
    n = length(keys)
    widemul(n, n) < typemax(Int64) ||
        throw(ArgumentError("cannot assemble $n entries in input order"))
    sorted = keys[perm]
    runs = accumulate(Base.add_sum, run_heads(sorted))
    free_buffer!(sorted)
    unique_keys = similar(keys)
    if !isempty(keys)
        positional_key_kernel(get_backend(keys))(unique_keys, runs, perm, n; ndrange=n)
    end
    free_buffer!(runs)
    perm2 = sortperm(unique_keys)
    free_buffer!(unique_keys)
    return perm[perm2]
end

function check_coordinates(I::AbstractVector, m::Integer, name)
    isempty(I) || all(i -> 1 <= i <= m, I) ||
        throw(ArgumentError("all $name indices must be in 1:$m"))
    return
end

"""
    GPUArrays.generic_assemble(I, J, V, m, n, combine; major=:col)

Assemble the entries `V[k]` at coordinates `(I[k], J[k])` of an `m`×`n` sparse matrix on
the device, combining the values of repeated coordinates with `combine`, from left to
right in input order (as `sparse(I, J, V, m, n, combine)` does). Returns the sorted, unique
major and minor indices of the entries (rows and columns for `major=:row`, columns and
rows for `major=:col`) and their values, from which any format can be built.

The coordinates are sorted on packed `Int64` keys; a second sort restores the input order
of repeated coordinates, unless the order cannot change the result (e.g. `+` on integers,
or `max`). `combine` must be associative.
"""
function generic_assemble(I::AbstractVector, J::AbstractVector, V::AbstractVector,
                          m::Integer, n::Integer, combine; major::Symbol=:col)
    length(I) == length(J) == length(V) ||
        throw(ArgumentError("the coordinate and value vectors must have the same length"))
    check_coordinates(I, m, "row")
    check_coordinates(J, n, "column")
    Ti = promote_type(eltype(I), eltype(J))
    check_sparse_dims(Ti, (Int(m), Int(n)))
    majors, minors, dims = major === :col ? (J, I, (n, m)) :
                           major === :row ? (I, J, (m, n)) :
                           throw(ArgumentError("major must be :row or :col"))

    keys = packed_keys(majors, minors, dims)
    perm = stable_sortperm(keys; stable=!order_independent(combine, eltype(V)))
    sorted_keys = keys[perm]
    sorted_vals = V[perm]
    free_buffer!(keys)
    free_buffer!(perm)
    unique_keys, vals = reduce_by_key(combine, sorted_keys, sorted_vals)
    free_buffer!(sorted_keys)
    free_buffer!(sorted_vals)

    out_major = similar(I, Ti, length(unique_keys))
    out_minor = similar(I, Ti, length(unique_keys))
    if !isempty(unique_keys)
        unpack_key_kernel(get_backend(unique_keys))(out_major, out_minor, unique_keys,
                                                    Int64(dims[2]); ndrange=length(unique_keys))
    end
    free_buffer!(unique_keys)
    return out_major, out_minor, vals
end

default_combine(::Type{Bool}) = |
default_combine(::Type) = +

const AnyGPUIndices = AnyGPUVector{<:Integer}

# `combine` can be any callable, with separate methods for functions to be more specific
# than SparseArrays' own
function assemble_matrix(I, J, V, m, n, combine, fmt::Symbol)
    S = sparse_format(fmt)
    major, minor, vals = generic_assemble(I, J, V, m, n, combine;
                                          major=S <: GPUSparseMatrixCSC ? :col : :row)
    if S <: GPUSparseMatrixCSC
        GPUSparseMatrixCSC(compress(major, n), minor, vals, (m, n))
    elseif S <: GPUSparseMatrixCSR
        GPUSparseMatrixCSR(compress(major, m), minor, vals, (m, n))
    else
        GPUSparseMatrixCOO(major, minor, vals, (m, n))
    end
end
for C in (:Function, :Any)
    @eval begin
        SparseArrays.sparse(I::AnyGPUIndices, J::AnyGPUIndices, V::AnyGPUVector, m::Integer,
                            n::Integer, combine::$C; fmt::Symbol=:csc) =
            assemble_matrix(I, J, V, m, n, combine, fmt)
        SparseArrays.sparse(I::AnyGPUIndices, J::AnyGPUIndices, v::Number, m::Integer,
                            n::Integer, combine::$C; fmt::Symbol=:csc) =
            assemble_matrix(I, J, fill!(similar(I, typeof(v)), v), m, n, combine, fmt)
    end
end

"""
    sparse(I, J, V, [m, n, combine]; fmt=:csc)

Assemble a GPU sparse matrix from GPU vectors of coordinates `I`, `J` and values `V` (or a
single value), like `SparseArrays.sparse`: repeated coordinates are combined with
`combine` (by default `+`, or `|` for `Bool` values) in input order, and explicit zeros are
kept. `fmt` selects the format: `:csc`, `:csr` or `:coo`.
"""
SparseArrays.sparse(::AnyGPUIndices, ::AnyGPUIndices, ::AnyGPUVector, ::Integer, ::Integer, ::Any)

# the defaults: `+`, or `|` for Bool values, and the dimensions that hold the coordinates.
# (Separate methods for Bool values keep these more specific than SparseArrays' own.)
for V in (:AnyGPUVector, :(AnyGPUVector{Bool}), :Number, :Bool)
    @eval begin
        SparseArrays.sparse(I::AnyGPUIndices, J::AnyGPUIndices, V::$V, m::Integer, n::Integer;
                            fmt::Symbol=:csc) =
            sparse(I, J, V, m, n, default_combine(eltype(V)); fmt)
        SparseArrays.sparse(I::AnyGPUIndices, J::AnyGPUIndices, V::$V; fmt::Symbol=:csc) =
            sparse(I, J, V, sparse_extent(I), sparse_extent(J); fmt)
    end
end

# the smallest dimension that holds the given indices
sparse_extent(I::AbstractVector) = isempty(I) ? 0 : Int(maximum(I))

function assemble_vector(I, V, n, combine)
    J = fill!(similar(I), one(eltype(I)))
    nzInd, _, vals = generic_assemble(I, J, V, n, 1, combine; major=:row)
    GPUSparseVector(nzInd, vals, n)
end
for C in (:Function, :Any)
    @eval begin
        SparseArrays.sparsevec(I::AnyGPUIndices, V::AnyGPUVector, n::Integer, combine::$C) =
            assemble_vector(I, V, n, combine)
        SparseArrays.sparsevec(I::AnyGPUIndices, v::Number, n::Integer, combine::$C) =
            assemble_vector(I, fill!(similar(I, typeof(v)), v), n, combine)
        SparseArrays.sparsevec(I::AnyGPUIndices, V::Union{AnyGPUVector,Number}, combine::$C) =
            sparsevec(I, V, sparse_extent(I), combine)
    end
end
"""
    sparsevec(I, V, [n, combine])

Assemble a GPU sparse vector from GPU vectors of indices `I` and values `V` (or a single
value), like `SparseArrays.sparsevec`.
"""
SparseArrays.sparsevec(::AnyGPUIndices, ::AnyGPUVector, ::Integer, ::Any)

for V in (:AnyGPUVector, :(AnyGPUVector{Bool}), :Number, :Bool)
    @eval begin
        SparseArrays.sparsevec(I::AnyGPUIndices, V::$V, n::Integer) =
            sparsevec(I, V, n, default_combine(eltype(V)))
        SparseArrays.sparsevec(I::AnyGPUIndices, V::$V) = sparsevec(I, V, sparse_extent(I))
    end
end

"""
    spdiagm(kv::Pair{<:Integer,<:AbstractGPUVector}...)
    spdiagm(m, n, kv...)

A GPU sparse matrix in CSC format with the vectors `kv[i].second` on the diagonals
`kv[i].first`, like `SparseArrays.spdiagm`. Values on the same diagonal are added.
"""
const GPUDiagonal = Pair{<:Integer,<:AnyGPUVector}
SparseArrays.spdiagm(kv::GPUDiagonal, kvs::GPUDiagonal...) = spdiagm_gpu(nothing, kv, kvs...)
SparseArrays.spdiagm(m::Integer, n::Integer, kv::GPUDiagonal, kvs::GPUDiagonal...) =
    spdiagm_gpu((Int(m), Int(n)), kv, kvs...)
SparseArrays.spdiagm(v::AnyGPUVector) = spdiagm(0 => v)
SparseArrays.spdiagm(m::Integer, n::Integer, v::AnyGPUVector) = spdiagm(m, n, 0 => v)

## COV_EXCL_START
@kernel function diagonal_coordinates_kernel(I, J, k)
    i = @index(Global, Linear)
    if i <= length(I)
        @inbounds I[i] = k >= 0 ? i : i - k
        @inbounds J[i] = k >= 0 ? i + k : i
    end
end
## COV_EXCL_STOP

function spdiagm_gpu(dims, kv::Pair{<:Integer,<:AnyGPUVector}...)
    # the extent that holds every diagonal, as in SparseArrays
    m, n = if dims === nothing
        extent = maximum(((k, v),) -> length(v) + abs(k), kv)
        (extent, extent)
    else
        dims
    end
    Tv = mapreduce(((_, v),) -> eltype(v), promote_type, kv)
    proto = last(first(kv))
    total = sum(((_, v),) -> length(v), kv)
    I = similar(proto, Int, total)
    J = similar(proto, Int, total)
    V = similar(proto, Tv, total)
    offset = 0
    for (k, v) in kv
        len = length(v)
        rows = k >= 0 ? len : len - k
        cols = k >= 0 ? len + k : len
        (rows <= m && cols <= n) ||
            throw(DimensionMismatch("diagonal $k of length $len does not fit in a $m×$n matrix"))
        if len > 0
            range = offset+1:offset+len
            diagonal_coordinates_kernel(get_backend(proto))(view(I, range), view(J, range), k;
                                                            ndrange=len)
            view(V, range) .= v
        end
        offset += len
    end
    return sparse(I, J, V, m, n, +)
end
