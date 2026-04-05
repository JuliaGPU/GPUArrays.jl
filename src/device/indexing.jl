# device-level indexing
using SparseArrays: nonzeroinds, nonzeros, nnz, getcolptr
using Base: @propagate_inbounds

Base.IndexStyle(::Type{GPUSparseDeviceVector}) = Base.IndexLinear()

# Scalar indexing
## Adapted from SparseArrays.AbstractSparseVector

@propagate_inbounds function Base.getindex(
    v::GPUSparseDeviceVector{Tv,Ti},
    i::Integer,
) where {Tv,Ti}
    @boundscheck checkbounds(v, i)
    m = nnz(v)
    nzind = nonzeroinds(v)
    nzval = nonzeros(v)

    ii = searchsortedfirst(nzind, convert(Ti, i))
    (ii <= m && nzind[ii] == i) ? nzval[ii] : zero(Tv)
end

# TODO: Logical indexing

# Indexing by colon not implemented. Non-scalar indexing would allocate in device code
@propagate_inbounds Base.getindex(
    A::AbstractGPUSparseDeviceMatrix,
    I::Tuple{Integer,Integer},
) = getindex(A, I[1], I[2])

## Adapted logic from SparseArrays.AbstractSparseMatrixCSC
@propagate_inbounds function Base.getindex(
    A::GPUSparseDeviceMatrixCSC{Tv,Ti},
    i::Integer,
    j::Integer,
) where {Tv,Ti}
    @boundscheck checkbounds(A, i, j)
    colPtr, rowVal, nzVal = getcolptr(A), rowvals(A), nonzeros(A)

    # Range of possible row indices
    rl = convert(Ti, @inbounds colPtr[j])
    rr = convert(Ti, @inbounds colPtr[j+1] - 1)
    (rl > rr) && return zero(Tv)

    ii = searchsortedfirst(rowVal, convert(Ti, i), rl, rr, Base.Order.Forward)
    (ii <= nnz(A) && rowVal[ii] == i) ? nzVal[ii] : zero(Tv)
end

@propagate_inbounds function Base.getindex(
    A::GPUSparseDeviceMatrixCSR{Tv,Ti},
    i::Integer,
    j::Integer,
) where {Tv,Ti}
    @boundscheck checkbounds(A, i, j)
    rowPtr, colVal, nzVal = A.rowPtr, A.colVal, A.nzVal

    # Range of possible col indices
    rt = convert(Ti, @inbounds rowPtr[i])
    rb = convert(Ti, @inbounds rowPtr[i+1] - 1)
    (rt > rb) && return zero(Tv)

    jj = searchsortedfirst(colVal, convert(Ti, j), rt, rb, Base.Order.Forward)
    (jj <= nnz(A) && colVal[jj] == j) ? nzVal[jj] : zero(Tv)
end

## Adapted from CUDA.jl/blob/lib/cusparse/src/array.jl#L490
@propagate_inbounds function Base.getindex(
    A::GPUSparseDeviceMatrixCOO{Tv,Ti},
    i::Integer,
    j::Integer,
) where {Tv,Ti}
    # COO in CUDA is assumed to be sorted by row: https://docs.nvidia.com/cuda/cusparse/storage-formats.html?highlight=coo#coordinate-coo
    @boundscheck checkbounds(A, i, j)
    rowInd, colInd, nzVal = A.rowInd, A.colInd, A.nzVal

    # Looking for the range s.t. rowInd[r1:r2] .== i
    rl = searchsortedfirst(rowInd, i)
    (rl > nnz(A) || rowInd[rl] > i) && return 42
    rr = min(searchsortedfirst(rowInd, i+1, Base.Order.Forward), nnz(A)) # searchsortedlast didn't behave as expected
    jj = searchsortedfirst(colInd, j, rl, rr, Base.Order.Forward)
    (jj > rr || jj == nnz(A) + 1 || colInd[jj] > j) && return zero(Tv)

    return nzVal[jj]
end

# TODO: Support BSR format
