# structural operations

# the row and column of every stored entry, borrowing the index buffers of `A`
entry_coordinates(A::GPUSparseMatrixCSR) = expand_ptr(A.rowPtr, nnz(A)), A.colVal
entry_coordinates(A::GPUSparseMatrixCSC) = A.rowVal, expand_ptr(A.colPtr, nnz(A))
entry_coordinates(A::GPUSparseMatrixCOO) = A.rowInd, A.colInd

## COV_EXCL_START
@kernel function keep_matrix_entries_kernel(keep, f, rows, cols, vals)
    k = @index(Global, Linear)
    if k <= length(keep)
        @inbounds keep[k] = f(rows[k], cols[k], vals[k])
    end
end

@kernel function keep_vector_entries_kernel(keep, f, inds, vals)
    k = @index(Global, Linear)
    if k <= length(keep)
        @inbounds keep[k] = f(inds[k], vals[k])
    end
end
## COV_EXCL_STOP

# replace the contents of `dst` by `src`, which may have a different length
function replace_buffer!(dst::AbstractVector, src::AbstractVector)
    copyto!(resize!(dst, length(src)), src)
    return dst
end

"""
    fkeep!(f, A::GPUSparseMatrix)
    fkeep!(f, x::GPUSparseVector)

Keep the stored entries of `A` for which `f(i, j, v)` (or `f(i, v)` for a vector) is true,
on the device, as in SparseArrays. `f` runs in a kernel. The buffers of `A` are resized, so
`A` must not share them with another array.
"""
function SparseArrays.fkeep!(f::F, A::GPUSparseMatrix) where {F}
    nnz(A) == 0 && return A
    rows, cols = entry_coordinates(A)
    keep = similar(A.nzVal, Bool)
    keep_matrix_entries_kernel(get_backend(A))(keep, f, rows, cols, A.nzVal; ndrange=nnz(A))
    kept = findall(keep)
    free_buffer!(keep)
    new_rows, new_cols, new_vals = rows[kept], cols[kept], A.nzVal[kept]
    free_buffer!(kept)
    if A isa GPUSparseMatrixCSR
        replace_buffer!(A.rowPtr, compress(new_rows, size(A, 1)))
        replace_buffer!(A.colVal, new_cols)
    elseif A isa GPUSparseMatrixCSC
        replace_buffer!(A.colPtr, compress(new_cols, size(A, 2)))
        replace_buffer!(A.rowVal, new_rows)
    else
        replace_buffer!(A.rowInd, new_rows)
        replace_buffer!(A.colInd, new_cols)
    end
    replace_buffer!(A.nzVal, new_vals)
    return A
end
function SparseArrays.fkeep!(f::F, x::GPUSparseVector) where {F}
    nnz(x) == 0 && return x
    keep = similar(x.nzVal, Bool)
    keep_vector_entries_kernel(get_backend(x))(keep, f, x.nzInd, x.nzVal; ndrange=nnz(x))
    kept = findall(keep)
    free_buffer!(keep)
    new_inds, new_vals = x.nzInd[kept], x.nzVal[kept]
    free_buffer!(kept)
    replace_buffer!(x.nzInd, new_inds)
    replace_buffer!(x.nzVal, new_vals)
    return x
end

# the tolerance is captured by value, so that the predicate can run in a kernel
struct AboveTolerance{T}
    tol::T
end
(p::AboveTolerance)(i, j, x) = abs(x) > p.tol
(p::AboveTolerance)(i, x) = abs(x) > p.tol

SparseArrays.dropzeros!(A::GPUSparseMatrix) = fkeep!((i, j, x) -> !iszero(x), A)
SparseArrays.dropzeros!(x::GPUSparseVector) = fkeep!((i, x) -> !iszero(x), x)
SparseArrays.dropzeros(A::GPUSparseArray) = dropzeros!(copy(A))
# the tolerance is compared in the precision of the values, so that a Float64 tolerance
# (like the default of `exp`) does not end up in kernels on back-ends without Float64
droptol_tolerance(::Type{T}, tol) where {T<:Union{AbstractFloat,Complex{<:AbstractFloat}}} =
    convert(real(T), tol)
droptol_tolerance(::Type, tol) = tol
SparseArrays.droptol!(A::GPUSparseArray, tol::Real) =
    fkeep!(AboveTolerance(droptol_tolerance(eltype(A), tol)), A)


## triangular parts

struct BandSelection
    lo::Int
    hi::Int
end
(b::BandSelection)(i, j, x) = b.lo <= j - i <= b.hi

const GPUSparseMatrixOrTransposed =
    Union{GPUSparseMatrix, Transpose{<:Any,<:GPUSparseMatrix}, Adjoint{<:Any,<:GPUSparseMatrix}}

# a copy of the entries of `A` whose diagonal index `j - i` lies within `lo:hi`
band(A::GPUSparseMatrix, lo, hi) = fkeep!(BandSelection(lo, hi), copy(A))
band(A::Union{Transpose,Adjoint}, lo, hi) = fkeep!(BandSelection(lo, hi), copy(A))

LinearAlgebra.triu(A::GPUSparseMatrixOrTransposed, k::Integer=0) = band(A, k, typemax(Int))
LinearAlgebra.tril(A::GPUSparseMatrixOrTransposed, k::Integer=0) = band(A, typemin(Int), k)

## COV_EXCL_START
@kernel function outside_band_kernel(outside, rows, cols, vals, lo, hi)
    k = @index(Global, Linear)
    if k <= length(outside)
        d = @inbounds cols[k] - rows[k]
        @inbounds outside[k] = !iszero(vals[k]) && !(lo <= d <= hi)
    end
end
## COV_EXCL_STOP

# whether all nonzero values lie within the diagonals `lo:hi`
function within_band(A::GPUSparseMatrix, lo, hi)
    nnz(A) == 0 && return true
    rows, cols = entry_coordinates(A)
    outside = similar(A.nzVal, Bool)
    outside_band_kernel(get_backend(A))(outside, rows, cols, A.nzVal, lo, hi; ndrange=nnz(A))
    return !any(outside)
end
LinearAlgebra.istriu(A::GPUSparseMatrix, k::Integer=0) = within_band(A, k, typemax(Int))
LinearAlgebra.istril(A::GPUSparseMatrix, k::Integer=0) = within_band(A, typemin(Int), k)
LinearAlgebra.isdiag(A::GPUSparseMatrix) = within_band(A, 0, 0)

# the wrapper determines where the nonzeros can be; LinearAlgebra's generic methods would
# index the parent (and are broken on some versions, JuliaLang/julia#55547)
const GPUSparseUpperOrUnitUpperTriangular =
    LinearAlgebra.UpperOrUnitUpperTriangular{<:Any,<:GPUSparseMatrixOrTransposed}
const GPUSparseLowerOrUnitLowerTriangular =
    LinearAlgebra.LowerOrUnitLowerTriangular{<:Any,<:GPUSparseMatrixOrTransposed}
LinearAlgebra.istriu(::GPUSparseUpperOrUnitUpperTriangular) = true
LinearAlgebra.istril(U::GPUSparseUpperOrUnitUpperTriangular) =
    iszero(triu(materialize(parent(U)), 1))
LinearAlgebra.istril(::GPUSparseLowerOrUnitLowerTriangular) = true
LinearAlgebra.istriu(L::GPUSparseLowerOrUnitLowerTriangular) =
    iszero(tril(materialize(parent(L)), -1))

# `A` equals its transpose (or adjoint) when no element differs, which broadcasting over
# the union of both structures also checks for stored zeros on one side only
function LinearAlgebra.issymmetric(A::GPUSparseMatrix)
    size(A, 1) == size(A, 2) || return false
    return !any(A .!= copy(transpose(A)))
end
function LinearAlgebra.ishermitian(A::GPUSparseMatrix)
    size(A, 1) == size(A, 2) || return false
    return !any(A .!= copy(adjoint(A)))
end


## diagonals

## COV_EXCL_START
# every thread looks up one element of the diagonal `k` in its slice
@kernel function diagonal_kernel(d, ptr, ind, val, k, csr::Bool)
    t = @index(Global, Linear)
    if t <= length(d)
        # the diagonal element in row/column `major`, at column/row `minor`
        i, j = k >= 0 ? (t, t + k) : (t - k, t)
        major, minor = csr ? (i, j) : (j, i)
        lo = @inbounds ptr[major]
        hi = @inbounds ptr[major+1] - one(lo)
        p = searchsortedfirst(ind, minor % eltype(ind), lo, hi, Base.Order.Forward)
        @inbounds d[t] = (p <= hi && ind[p] == minor) ? val[p] : zero(eltype(d))
    end
end
## COV_EXCL_STOP

function LinearAlgebra.diag(A::GPUSparseMatrix, k::Integer=0)
    m, n = size(A)
    -m <= k <= n || throw(ArgumentError("requested diagonal, $k, must be at least $(-m) and at most $n for a $m×$n matrix"))
    len = k >= 0 ? min(m, n - k) : min(m + k, n)
    d = similar(A.nzVal, len)
    len == 0 && return d
    B = A isa GPUSparseMatrixCOO ? csr_view(A) : A
    diagonal_kernel(get_backend(B))(d, compressed_ptr(B), compressed_ind(B), B.nzVal, k,
                                    B isa GPUSparseMatrixCSR; ndrange=len)
    return d
end

function LinearAlgebra.tr(A::GPUSparseMatrix)
    LinearAlgebra.checksquare(A)
    return sum(diag(A))
end


## reshaping

## COV_EXCL_START
@kernel function reshape_kernel(rows, cols, m, m′)
    k = @index(Global, Linear)
    if k <= length(rows)
        l = (Int64(cols[k]) - 1) * m + Int64(rows[k]) - 1
        @inbounds rows[k] = (l % m′ + 1) % eltype(rows)
        @inbounds cols[k] = (l ÷ m′ + 1) % eltype(cols)
    end
end

@kernel function linear_index_kernel(inds, rows, cols, m)
    k = @index(Global, Linear)
    if k <= length(inds)
        @inbounds inds[k] = ((Int64(cols[k]) - 1) * m + rows[k]) % eltype(inds)
    end
end
## COV_EXCL_STOP

# Column-major order of the entries (that of CSC) is the order of their linear indices, so
# it is preserved by any reshape: new coordinates, in the same order, form the CSC buffers
# of the result.
function Base.reshape(A::GPUSparseMatrix, dims::Dims{2})
    prod(dims) == length(A) ||
        throw(DimensionMismatch("new dimensions $dims must be consistent with array size $(length(A))"))
    dims == size(A) && return copy(A)
    C = A isa GPUSparseMatrixCSC ? copy(A) : GPUSparseMatrixCSC(A)
    rows = C.rowVal
    cols = expand_ptr(C.colPtr, nnz(C))
    check_sparse_dims(indtype(A), dims)
    if nnz(C) > 0
        reshape_kernel(get_backend(C))(rows, cols, size(A, 1), dims[1]; ndrange=nnz(C))
    end
    B = GPUSparseMatrixCSC(compress(cols, dims[2]), rows, C.nzVal, dims)
    free_buffer!(cols)
    return convert(matrix_format(A), B)
end
Base.reshape(A::GPUSparseMatrix, dims::Tuple{Union{Int,Colon},Union{Int,Colon}}) =
    reshape(A, Base._reshape_uncolon(A, dims))

function Base.vec(A::GPUSparseMatrix)
    C = A isa GPUSparseMatrixCSC ? A : GPUSparseMatrixCSC(A)
    check_sparse_dims(indtype(A), (length(A),))
    inds = similar(C.rowVal, nnz(C))
    if nnz(C) > 0
        cols = expand_ptr(C.colPtr, nnz(C))
        linear_index_kernel(get_backend(C))(inds, C.rowVal, cols, size(C, 1); ndrange=nnz(C))
        free_buffer!(cols)
    end
    return GPUSparseVector(inds, C === A ? copy(C.nzVal) : C.nzVal, length(A))
end


## Kronecker products

## COV_EXCL_START
# every thread fills a column of `kron(A, B)`: the products of column `j1` of `A` and
# column `j2` of `B`, in column-major order
@kernel function kron_kernel(colPtr, rowVal, nzVal, Aptr, Aind, Aval, Bptr, Bind, Bval, m2, n2)
    c = @index(Global, Linear)
    if c < length(colPtr)
        j1 = (c - 1) ÷ n2 + 1
        j2 = (c - 1) % n2 + 1
        out = @inbounds colPtr[c]
        for p in @inbounds(Aptr[j1]):@inbounds(Aptr[j1+1] - one(eltype(Aptr)))
            for q in @inbounds(Bptr[j2]):@inbounds(Bptr[j2+1] - one(eltype(Bptr)))
                @inbounds rowVal[out] = ((Aind[p] - 1) * m2 + Bind[q]) % eltype(rowVal)
                @inbounds nzVal[out] = Aval[p] * Bval[q]
                out += one(out)
            end
        end
    end
end

@kernel function kron_counts_kernel(counts, Aptr, Bptr, n2)
    c = @index(Global, Linear)
    if c <= length(counts)
        j1 = (c - 1) ÷ n2 + 1
        j2 = (c - 1) % n2 + 1
        @inbounds counts[c] = (Aptr[j1+1] - Aptr[j1]) * (Bptr[j2+1] - Bptr[j2])
    end
end
## COV_EXCL_STOP

function generic_kron(A::GPUSparseMatrixCSC, B::GPUSparseMatrixCSC)
    m1, n1 = size(A)
    m2, n2 = size(B)
    Ti = promote_type(indtype(A), indtype(B))
    Tv = Base.promote_op(*, eltype(A), eltype(B))
    all(<=(typemax(Int)), (widemul(m1, m2), widemul(n1, n2))) ||
        throw(ArgumentError("the Kronecker product of $(size(A)) and $(size(B)) matrices is too large"))
    dims = (m1 * m2, n1 * n2)
    check_sparse_dims(Ti, dims)
    check_sparse_nnz(Ti, widemul(nnz(A), nnz(B)))
    n = nnz(A) * nnz(B)
    colPtr = similar(A.colPtr, Ti, dims[2] + 1)
    rowVal = similar(A.rowVal, Ti, n)
    nzVal = similar(A.nzVal, Tv, n)
    fill!(view(colPtr, 1:1), one(Ti))
    if dims[2] > 0
        kron_counts_kernel(get_backend(A))(view(colPtr, 2:dims[2]+1), A.colPtr, B.colPtr, n2;
                                           ndrange=dims[2])
        accumulate!(Base.add_sum, colPtr, colPtr)
        if n > 0
            kron_kernel(get_backend(A))(colPtr, rowVal, nzVal, A.colPtr, A.rowVal, A.nzVal,
                                        B.colPtr, B.rowVal, B.nzVal, m2, n2; ndrange=dims[2])
        end
    end
    return GPUSparseMatrixCSC(colPtr, rowVal, nzVal, dims)
end

# the result has the format of `A`
function LinearAlgebra.kron(A::GPUSparseMatrixOrTransposed, B::GPUSparseMatrixOrTransposed)
    C = generic_kron(convert(GPUSparseMatrixCSC, materialize(A)),
                     convert(GPUSparseMatrixCSC, materialize(B)))
    return convert(matrix_format(materialize_format(A)), C)
end
LinearAlgebra.kron(A::GPUSparseMatrixOrTransposed, D::Diagonal{<:Any,<:AnyGPUVector}) =
    kron(A, spdiagm(D.diag))
LinearAlgebra.kron(D::Diagonal{<:Any,<:AnyGPUVector}, B::GPUSparseMatrixOrTransposed) =
    convert(matrix_format(materialize_format(B)), kron(spdiagm(D.diag), B))

# a sparse matrix without a lazy wrapper, and the format that has
materialize(A::GPUSparseMatrix) = A
materialize(A::Union{Transpose,Adjoint}) = copy(A)
materialize_format(A::GPUSparseMatrix) = A
materialize_format(A::Union{Transpose,Adjoint}) = parent(A)
