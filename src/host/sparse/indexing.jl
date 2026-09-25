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
