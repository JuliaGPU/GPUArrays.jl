# indexing

# Scalar indexing reads a few entries of the buffers, so like dense GPU arrays it requires
# `@allowscalar`. The searches are binary searches over the stored indices of one slice.

# the value stored at `ind[lo:hi] == i` if any, else zero
function stored_value(A, ind, lo::Integer, hi::Integer, i::Integer)
    lo > hi && return zero(eltype(A))
    k = searchsortedfirst(ind, i, lo, hi, Base.Order.Forward)
    (k > hi || ind[k] != i) && return zero(eltype(A))
    return nonzeros(A)[k]
end

function Base.getindex(A::GPUSparseMatrixCSR, i::Integer, j::Integer)
    @boundscheck checkbounds(A, i, j)
    assertscalar("getindex")
    stored_value(A, A.colVal, Int(A.rowPtr[i]), Int(A.rowPtr[i+1]) - 1, j)
end

function Base.getindex(A::GPUSparseMatrixCSC, i::Integer, j::Integer)
    @boundscheck checkbounds(A, i, j)
    assertscalar("getindex")
    stored_value(A, A.rowVal, Int(A.colPtr[j]), Int(A.colPtr[j+1]) - 1, i)
end

function Base.getindex(A::GPUSparseMatrixCOO, i::Integer, j::Integer)
    @boundscheck checkbounds(A, i, j)
    assertscalar("getindex")
    # the entries of row `i` are contiguous, and sorted by column
    lo = searchsortedfirst(A.rowInd, i)
    hi = searchsortedlast(A.rowInd, i)
    stored_value(A, A.colInd, lo, hi, j)
end

function Base.getindex(x::GPUSparseVector, i::Integer)
    @boundscheck checkbounds(x, i)
    assertscalar("getindex")
    stored_value(x, x.nzInd, 1, nnz(x), i)
end

Base.getindex(A::GPUSparseMatrix, I::Tuple{Integer,Integer}) = getindex(A, I...)
Base.getindex(x::GPUSparseVector, ::Colon) = copy(x)
Base.getindex(A::GPUSparseMatrix, ::Colon, ::Colon) = copy(A)

function Base.setindex!(::GPUSparseArray, v, I...)
    throw(ArgumentError("""GPU sparse arrays do not support setting individual entries.
                           Assemble a new array with `sparse(I, J, V, m, n)` or `sparsevec(I, V, n)`, use broadcasting, or update stored values through `nonzeros(A)`."""))
end


## slices and ranges

# Indexing with ranges (and colons) keeps the stored entries that fall in them and
# renumbers them. Ranges with a positive step preserve the order of the entries, so the
# result needs no sorting; other index vectors are not supported.
const SparseRangeIndex = Union{AbstractRange{<:Integer}, Colon}

range_index(I::Colon, n) = Base.OneTo(n)
function range_index(I::AbstractRange{<:Integer}, n)
    (length(I) <= 1 || step(I) > 0) ||
        throw(ArgumentError("indexing GPU sparse arrays requires ranges with a positive step"))
    return I
end
range_index(i::Integer, n) = i:i

# the position of `i` in the range `I` (from `first`, with `step`, of `len` elements), or 0
@inline function range_position(i, first, step, len)
    d = i - first
    (d >= 0 && d % step == 0) || return 0
    p = d ÷ step + 1
    return 1 <= p <= len ? p : 0
end

## COV_EXCL_START
@kernel function select_entries_kernel(keep, new_rows, new_cols, rows, cols, I, J)
    k = @index(Global, Linear)
    if k <= length(keep)
        i = range_position(Int(@inbounds rows[k]), I...)
        j = range_position(Int(@inbounds cols[k]), J...)
        @inbounds keep[k] = i != 0 && j != 0
        @inbounds new_rows[k] = i % eltype(new_rows)
        @inbounds new_cols[k] = j % eltype(new_cols)
    end
end

@kernel function select_vector_entries_kernel(keep, new_inds, inds, I)
    k = @index(Global, Linear)
    if k <= length(keep)
        i = range_position(Int(@inbounds inds[k]), I...)
        @inbounds keep[k] = i != 0
        @inbounds new_inds[k] = i % eltype(new_inds)
    end
end
## COV_EXCL_STOP

# (a range of one element may have any step)
range_parameters(I::AbstractRange) = (Int(first(I)), length(I) > 1 ? Int(step(I)) : 1, length(I))

# the entries of `A[I, J]`, in the order they have in `A`: their new rows and columns, and
# their values
function select_entries(A::GPUSparseMatrix, I::AbstractRange, J::AbstractRange)
    rows, cols = entry_coordinates(A)
    n = nnz(A)
    keep = similar(A.nzVal, Bool, n)
    new_rows = similar(rows, n)
    new_cols = similar(cols, n)
    if n > 0
        select_entries_kernel(get_backend(A))(keep, new_rows, new_cols, rows, cols,
                                              range_parameters(I), range_parameters(J);
                                              ndrange=n)
    end
    kept = findall(keep)
    return new_rows[kept], new_cols[kept], A.nzVal[kept]
end

function sparse_getindex(A::GPUSparseMatrix, I, J)
    @boundscheck checkbounds(A, I, J)
    m, n = size(A)
    rows, cols, vals = select_entries(A, range_index(I, m), range_index(J, n))
    if I isa Integer || J isa Integer
        # entries of a single row or column are in order of the other index in every format
        return GPUSparseVector(I isa Integer ? cols : rows, vals,
                               I isa Integer ? length(range_index(J, n)) : length(range_index(I, m)))
    end
    dims = (length(range_index(I, m)), length(range_index(J, n)))
    if A isa GPUSparseMatrixCSR
        GPUSparseMatrixCSR(compress(rows, dims[1]), cols, vals, dims)
    elseif A isa GPUSparseMatrixCSC
        GPUSparseMatrixCSC(compress(cols, dims[2]), rows, vals, dims)
    else
        GPUSparseMatrixCOO(rows, cols, vals, dims)
    end
end

Base.getindex(A::GPUSparseMatrix, I::SparseRangeIndex, J::SparseRangeIndex) =
    sparse_getindex(A, I, J)
Base.getindex(A::GPUSparseMatrix, i::Integer, J::SparseRangeIndex) = sparse_getindex(A, i, J)
Base.getindex(A::GPUSparseMatrix, I::SparseRangeIndex, j::Integer) = sparse_getindex(A, I, j)

function sparse_getindex(x::GPUSparseVector, I::AbstractRange{<:Integer})
    @boundscheck checkbounds(x, I)
    I = range_index(I, length(x))
    keep = similar(x.nzVal, Bool, nnz(x))
    new_inds = similar(x.nzInd, nnz(x))
    if nnz(x) > 0
        select_vector_entries_kernel(get_backend(x))(keep, new_inds, x.nzInd,
                                                     range_parameters(I); ndrange=nnz(x))
    end
    kept = findall(keep)
    return GPUSparseVector(new_inds[kept], x.nzVal[kept], length(I))
end
# (separate methods to be more specific than SparseArrays' for any sparse vector)
Base.getindex(x::GPUSparseVector, I::AbstractUnitRange{<:Integer}) = sparse_getindex(x, I)
Base.getindex(x::GPUSparseVector, I::StepRange{<:Integer}) = sparse_getindex(x, I)
