# assembly of sparse arrays from coordinates

## COV_EXCL_START
unpack_sparse_key(key::Integer, nminor) = ((key - 1) ÷ nminor + 1, (key - 1) % nminor + 1)
unpack_sparse_key(key::Tuple{Integer,Integer}, nminor) = key

@kernel function unpack_key_kernel(major, minor, keys, nminor)
    k = @index(Global, Linear)
    if k <= length(keys)
        i, j = unpack_sparse_key(@inbounds(keys[k]), nminor)
        @inbounds major[k] = i % eltype(major)
        @inbounds minor[k] = j % eltype(minor)
    end
end
## COV_EXCL_STOP

# AK's stable permutation sort preserves relative operand order within each run.
stable_sortperm(keys::AbstractVector) = sortperm(keys; alg=Base.Sort.MergeSort)

function check_coordinates(I::AbstractVector, m::Integer, name)
    isempty(I) || all(i -> 1 <= i <= m, I) ||
        throw(ArgumentError("all $name indices must be in 1:$m"))
    return
end

"""
    GPUArrays.generic_assemble(I, J, V, m, n, combine; major=:col)

Assemble the entries `V[k]` at coordinates `(I[k], J[k])` of an `m`×`n` sparse matrix on
the device, combining duplicate values with `combine` while preserving their relative input
order. Parenthesization is unspecified: custom combiners must be associative,
although commutativity is not required. Floating-point rounding may vary. Returns the sorted, unique
major and minor indices of the entries (rows and columns for `major=:row`, columns and
rows for `major=:col`) and their values, from which any format can be built.

Coordinates use packed Int64 keys where representable, otherwise lexicographic tuple
keys. A single stable permutation sort preserves duplicate order.
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
    perm = stable_sortperm(keys)
    sorted_keys = keys[perm]
    sorted_vals = V[perm]
    free_buffer!(keys)
    free_buffer!(perm)
    unique_keys, vals = reduce_by_key(combine, sorted_keys, sorted_vals)
    free_buffer!(sorted_keys)
    free_buffer!(sorted_vals)

    out_major = sparse_similar(I, Ti, length(unique_keys))
    out_minor = sparse_similar(I, Ti, length(unique_keys))
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
function assemble_matrix(I, J, V, m, n, combine, fmt::Symbol;
                         pointer_type::Type=promote_type(Int, eltype(I), eltype(J)))
    S = sparse_format(fmt)
    major, minor, vals = generic_assemble(I, J, V, m, n, combine;
                                          major=S <: GPUSparseMatrixCSC ? :col : :row)
    if S <: GPUSparseMatrixCSC
        GPUSparseMatrixCSC(compress(major, n, pointer_type), minor, vals, (m, n))
    elseif S <: GPUSparseMatrixCSR
        GPUSparseMatrixCSR(compress(major, m, pointer_type), minor, vals, (m, n))
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
            assemble_matrix(I, J, fill!(sparse_similar(I, typeof(v)), v), m, n, combine, fmt)
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
    J = fill!(sparse_similar(I), one(eltype(I)))
    nzInd, _, vals = generic_assemble(I, J, V, n, 1, combine; major=:row)
    GPUSparseVector(nzInd, vals, n)
end
for C in (:Function, :Any)
    @eval begin
        SparseArrays.sparsevec(I::AnyGPUIndices, V::AnyGPUVector, n::Integer, combine::$C) =
            assemble_vector(I, V, n, combine)
        SparseArrays.sparsevec(I::AnyGPUIndices, v::Number, n::Integer, combine::$C) =
            assemble_vector(I, fill!(sparse_similar(I, typeof(v)), v), n, combine)
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
        extent = maximum(((k, v),) -> Base.checked_add(length(v), Base.Checked.checked_abs(k)), kv)
        (extent, extent)
    else
        dims
    end
    Tv = mapreduce(((_, v),) -> eltype(v), promote_type, kv)
    proto = last(first(kv))
    total = foldl((n, pair) -> Base.checked_add(n, length(last(pair))), kv; init=0)
    I = sparse_similar(proto, Int, total)
    J = sparse_similar(proto, Int, total)
    V = sparse_similar(proto, Tv, total)
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
