# GPU sparse array types

using SparseArrays: indtype, getcolptr, nonzeroinds

"""
    AbstractGPUSparseArray{Tv,Ti,N}

Supertype of the sparse arrays whose storage lives on a GPU. GPUArrays provides the formats
[`GPUSparseVector`](@ref), [`GPUSparseMatrixCSR`](@ref), [`GPUSparseMatrixCSC`](@ref) and
[`GPUSparseMatrixCOO`](@ref); back-ends only subtype this directly for formats that
GPUArrays does not implement (e.g. block-sparse formats of a vendor library).
"""
abstract type AbstractGPUSparseArray{Tv,Ti,N} <: AbstractSparseArray{Tv,Ti,N} end
const AbstractGPUSparseVector{Tv,Ti} = AbstractGPUSparseArray{Tv,Ti,1}
const AbstractGPUSparseMatrix{Tv,Ti} = AbstractGPUSparseArray{Tv,Ti,2}
const AbstractGPUSparseVecOrMat = Union{AbstractGPUSparseVector,AbstractGPUSparseMatrix}

# the checks below only look at host metadata; structural invariants (sorted, unique and
# in-bounds indices, consistent pointers) are the caller's responsibility, as for
# SparseArrays' raw constructors. `check_structure` verifies them on the device.
function check_sparse_dims(::Type{Ti}, dims::Dims) where {Ti}
    all(>=(0), dims) ||
        throw(ArgumentError("dimensions must be non-negative, got $dims"))
    all(d -> d <= typemax(Ti), dims) ||
        throw(ArgumentError("dimensions $dims do not fit index type $Ti"))
    return
end
function check_sparse_nnz(::Type{Ti}, nnz::Integer) where {Ti}
    # pointers go up to nnz+1
    0 <= nnz < typemax(Ti) && nnz < typemax(Int) ||
        throw(ArgumentError("$nnz stored entries do not fit index type $Ti"))
    return
end
function check_sparse_entries(::Type{Ti}, inds::AbstractVector, nzVal::AbstractVector) where {Ti}
    length(inds) == length(nzVal) ||
        throw(ArgumentError("index and value buffers must have the same length, got $(length(inds)) and $(length(nzVal))"))
end
function check_sparse_ptr(ptr::AbstractVector, n::Integer)
    length(ptr) == Base.checked_add(n, 1) ||
        throw(ArgumentError("the pointer buffer must have length $(n + 1), got $(length(ptr))"))
    return
end

# Query only host GPU wrappers. Device representations are plain DenseArrays in modern
# backends and must be reconstructible without invoking host runtime APIs.
function check_sparse_storage(buffers...)
    if any(b -> b isa AnyGPUArray, buffers)
        all(b -> b isa AnyGPUArray, buffers) ||
            throw(ArgumentError("sparse component buffers must use compatible storage"))
        backend = get_backend(first(buffers))
        all(b -> get_backend(b) == backend, buffers) ||
            throw(ArgumentError("sparse component buffers must use the same backend"))
    end
    return nothing
end

"""
    GPUSparseMatrixCSR(rowPtr, colVal, nzVal, dims)

A sparse matrix in compressed sparse row format, stored in the dense vectors `rowPtr`
(length `m+1`), `colVal` and `nzVal` (length `nnz`). The stored entries of row `i` are at
positions `rowPtr[i]:rowPtr[i+1]-1`, with their columns in `colVal` and their values in
`nzVal`. The columns in a row must be sorted and unique; explicitly stored zeros are
allowed. Raw constructors borrow the supplied buffers, which must have compatible
storage and device placement. Coordinate storage `Vi` and compressed-pointer storage
`Vp` may use different integer types and wrappers. The stored count plus one must fit
the pointer type; both dimensions must fit the coordinate type.

The index and value buffers are typically GPU vectors, whose type is part of the matrix
type (`GPUSparseMatrixCSR{Tv,Ti,Vp,Vi,Vv}`), so that generic code allocates new buffers with
`similar` on the existing ones. Back-ends provide aliases for their storage, e.g.
`const MtlSparseMatrixCSR{Tv,Ti} = GPUSparseMatrixCSR{Tv,Ti,<:MtlVector{<:Integer},<:MtlVector{Ti},<:MtlVector{Tv}}`.

The same struct is used on the device: `adapt` converts the buffers, so kernels can read
the fields of a sparse matrix passed as an argument.
"""
struct GPUSparseMatrixCSR{Tv,Ti<:Integer,Vp<:AbstractVector{<:Integer},Vi<:AbstractVector{Ti},Vv<:AbstractVector{Tv}} <:
       AbstractGPUSparseMatrix{Tv,Ti}
    rowPtr::Vp
    colVal::Vi
    nzVal::Vv
    dims::Dims{2}

    function GPUSparseMatrixCSR{Tv,Ti,Vp,Vi,Vv}(rowPtr::Vp, colVal::Vi, nzVal::Vv,
                                            dims::NTuple{2,Integer}) where {Tv,Ti,Vp,Vi,Vv}
        dims = Dims{2}(dims)
        check_sparse_dims(Ti, dims)
        check_sparse_ptr(rowPtr, dims[1])
        check_sparse_entries(Ti, colVal, nzVal)
        check_sparse_nnz(eltype(rowPtr), length(nzVal))
        check_sparse_storage(rowPtr, colVal, nzVal)
        new{Tv,Ti,Vp,Vi,Vv}(rowPtr, colVal, nzVal, dims)
    end
end

"""
    GPUSparseMatrixCSC(colPtr, rowVal, nzVal, dims)

A sparse matrix in compressed sparse column format, the layout of `SparseMatrixCSC`: the
stored entries of column `j` are at positions `colPtr[j]:colPtr[j+1]-1`, with their rows
(sorted and unique) in `rowVal` and their values in `nzVal`. See
[`GPUSparseMatrixCSR`](@ref) for the storage type parameters.
"""
struct GPUSparseMatrixCSC{Tv,Ti<:Integer,Vp<:AbstractVector{<:Integer},Vi<:AbstractVector{Ti},Vv<:AbstractVector{Tv}} <:
       AbstractGPUSparseMatrix{Tv,Ti}
    colPtr::Vp
    rowVal::Vi
    nzVal::Vv
    dims::Dims{2}

    function GPUSparseMatrixCSC{Tv,Ti,Vp,Vi,Vv}(colPtr::Vp, rowVal::Vi, nzVal::Vv,
                                            dims::NTuple{2,Integer}) where {Tv,Ti,Vp,Vi,Vv}
        dims = Dims{2}(dims)
        check_sparse_dims(Ti, dims)
        check_sparse_ptr(colPtr, dims[2])
        check_sparse_entries(Ti, rowVal, nzVal)
        check_sparse_nnz(eltype(colPtr), length(nzVal))
        check_sparse_storage(colPtr, rowVal, nzVal)
        new{Tv,Ti,Vp,Vi,Vv}(colPtr, rowVal, nzVal, dims)
    end
end

"""
    GPUSparseMatrixCOO(rowInd, colInd, nzVal, dims)

A sparse matrix in coordinate format: stored entry `k` is at `(rowInd[k], colInd[k])` and
has value `nzVal[k]`. The entries must be sorted in row-major order without duplicate
coordinates, which makes the format a CSR matrix with explicit row indices. Unordered
coordinates with duplicates are the input of `sparse(I, J, V, m, n, combine)` instead.
See [`GPUSparseMatrixCSR`](@ref) for the storage type parameters.
"""
struct GPUSparseMatrixCOO{Tv,Ti<:Integer,Vi<:AbstractVector{Ti},Vv<:AbstractVector{Tv}} <:
       AbstractGPUSparseMatrix{Tv,Ti}
    rowInd::Vi
    colInd::Vi
    nzVal::Vv
    dims::Dims{2}

    function GPUSparseMatrixCOO{Tv,Ti,Vi,Vv}(rowInd::Vi, colInd::Vi, nzVal::Vv,
                                            dims::NTuple{2,Integer}) where {Tv,Ti,Vi,Vv}
        dims = Dims{2}(dims)
        check_sparse_dims(Ti, dims)
        check_sparse_entries(Ti, rowInd, nzVal)
        check_sparse_entries(Ti, colInd, nzVal)
        check_sparse_storage(rowInd, colInd, nzVal)
        new{Tv,Ti,Vi,Vv}(rowInd, colInd, nzVal, dims)
    end
end

"""
    GPUSparseVector(nzInd, nzVal, len)

A sparse vector of length `len` with stored entries at the sorted, unique indices `nzInd`
and values `nzVal`, the layout of `SparseVector`. See [`GPUSparseMatrixCSR`](@ref) for the
storage type parameters.
"""
struct GPUSparseVector{Tv,Ti<:Integer,Vi<:AbstractVector{Ti},Vv<:AbstractVector{Tv}} <:
       AbstractGPUSparseVector{Tv,Ti}
    nzInd::Vi
    nzVal::Vv
    len::Int

    function GPUSparseVector{Tv,Ti,Vi,Vv}(nzInd::Vi, nzVal::Vv, len::Integer) where {Tv,Ti,Vi,Vv}
        check_sparse_dims(Ti, (Int(len),))
        check_sparse_entries(Ti, nzInd, nzVal)
        check_sparse_storage(nzInd, nzVal)
        new{Tv,Ti,Vi,Vv}(nzInd, nzVal, len)
    end
end

for (S, args) in ((:GPUSparseMatrixCSR, (:rowPtr, :colVal)),
                  (:GPUSparseMatrixCSC, (:colPtr, :rowVal)))
    @eval $S($(args[1])::Vp, $(args[2])::Vi, nzVal::Vv, dims::NTuple{2,Integer}) where
            {Vp<:AbstractVector{<:Integer},Ti<:Integer,Vi<:AbstractVector{Ti},Tv,Vv<:AbstractVector{Tv}} =
        $S{Tv,Ti,Vp,Vi,Vv}($(args[1]), $(args[2]), nzVal, dims)
end
for (S, args) in ((:GPUSparseMatrixCOO, (:rowInd, :colInd)),)
    @eval $S($(args[1])::Vi, $(args[2])::Vi, nzVal::Vv, dims::NTuple{2,Integer}) where
            {Ti<:Integer,Vi<:AbstractVector{Ti},Tv,Vv<:AbstractVector{Tv}} =
        $S{Tv,Ti,Vi,Vv}($(args[1]), $(args[2]), nzVal, dims)
end
GPUSparseVector(nzInd::Vi, nzVal::Vv, len::Integer) where
        {Ti<:Integer,Vi<:AbstractVector{Ti},Tv,Vv<:AbstractVector{Tv}} =
    GPUSparseVector{Tv,Ti,Vi,Vv}(nzInd, nzVal, len)

const GPUSparseMatrixCompressed = Union{GPUSparseMatrixCSR,GPUSparseMatrixCSC}
const GPUSparseMatrix = Union{GPUSparseMatrixCSR,GPUSparseMatrixCSC,GPUSparseMatrixCOO}
const GPUSparseArray = Union{GPUSparseVector,GPUSparseMatrix}


## accessors

Base.size(A::GPUSparseMatrix) = A.dims
Base.length(A::GPUSparseMatrix) = Base.checked_mul(size(A, 1), size(A, 2))
Base.size(x::GPUSparseVector) = (x.len,)

SparseArrays.nnz(A::GPUSparseArray) = length(A.nzVal)
SparseArrays.nonzeros(A::GPUSparseArray) = A.nzVal

SparseArrays.getcolptr(A::GPUSparseMatrixCSC) = A.colPtr
SparseArrays.rowvals(A::GPUSparseMatrixCSC) = A.rowVal
SparseArrays.nzrange(A::GPUSparseMatrixCSC, col::Integer) =
    @inbounds A.colPtr[col]:(A.colPtr[col+1] - one(eltype(A.colPtr)))

SparseArrays.nonzeroinds(x::GPUSparseVector) = x.nzInd
SparseArrays.rowvals(x::GPUSparseVector) = x.nzInd

KernelAbstractions.get_backend(A::GPUSparseArray) = get_backend(A.nzVal)

# the pointer and index buffers of a compressed matrix, whatever its orientation
compressed_ptr(A::GPUSparseMatrixCSR) = A.rowPtr
compressed_ptr(A::GPUSparseMatrixCSC) = A.colPtr
compressed_ind(A::GPUSparseMatrixCSR) = A.colVal
compressed_ind(A::GPUSparseMatrixCSC) = A.rowVal

function promote_pointer_type(A, B)
    if A isa GPUSparseMatrixCompressed
        return B isa GPUSparseMatrixCompressed ?
            promote_type(eltype(compressed_ptr(A)), eltype(compressed_ptr(B))) :
            eltype(compressed_ptr(A))
    elseif B isa GPUSparseMatrixCompressed
        return eltype(compressed_ptr(B))
    end
    return promote_type(Int, indtype(A), indtype(B))
end

# the size of the dimension the pointers compress, and of the other one
major_dim(A::GPUSparseMatrixCSR) = size(A, 1)
major_dim(A::GPUSparseMatrixCSC) = size(A, 2)
minor_dim(A::GPUSparseMatrixCSR) = size(A, 2)
minor_dim(A::GPUSparseMatrixCSC) = size(A, 1)


## allocation

# Counts and byte sizes are host dimensions. Check arithmetic before asking a backend
# allocator, including allocations whose integer element type is narrower than Int.
function sparse_similar(proto, ::Type{T}, dims::Dims) where {T}
    count = foldl(Base.checked_mul, dims; init=1)
    count >= 0 || throw(ArgumentError("negative sparse allocation dimensions $dims"))
    Base.checked_mul(count, sizeof(T))
    return similar(proto, T, dims)
end
sparse_similar(proto, ::Type{T}, n::Integer) where {T} = sparse_similar(proto, T, (Int(n),))
sparse_similar(proto, ::Type{T}) where {T} = sparse_similar(proto, T, size(proto))
sparse_similar(proto, dims::Dims) = sparse_similar(proto, eltype(proto), dims)
sparse_similar(proto, n::Integer) = sparse_similar(proto, eltype(proto), (Int(n),))
sparse_similar(proto) = sparse_similar(proto, eltype(proto), size(proto))

# a copy of `src` converted to eltype `T`. Indices are converted without checks, as the
# callers have verified that the dimensions and number of entries fit.
function convert_buffer(::Type{T}, src::AbstractVector) where {T}
    dst = similar(src, T)
    dst .= src
    return dst
end

# Canonical dimensions/counts prove index representability before this conversion.
# Values use ordinary conversion semantics and must never take this unchecked path.
function convert_index_buffer(::Type{T}, src::AbstractVector) where {T<:Integer}
    dst = sparse_similar(src, T)
    dst .= src .% T
    return dst
end

function check_sparse_values(::Type{T}, values::AbstractVector{S}) where {T<:Integer,S<:Integer}
    if !all(v -> typemin(T) <= v <= typemax(T), values)
        throw(InexactError(:convert, T, "stored values outside the destination integer range"))
    end
    return nothing
end
check_sparse_values(::Type, ::AbstractVector) = nothing

# an empty compressed pointer buffer for `n` slices
empty_ptr(proto::AbstractVector, ::Type{Ti}, n::Integer) where {Ti} =
    fill!(sparse_similar(proto, Ti, Base.checked_add(n, 1)), one(Ti))

# SparseArrays semantics: `similar(A[, Tv[, Ti]])` copies the structure and leaves the
# values uninitialized, while `similar(A, Tv, dims)` returns an array without stored
# entries (even when `dims == size(A)`): a sparse matrix of the same format for two
# dimensions, a sparse vector for one, and a dense array for more.
Base.similar(A::GPUSparseArray) = similar(A, eltype(A))
Base.similar(A::GPUSparseArray, ::Type{Tv}) where {Tv} = similar(A, Tv, indtype(A))
Base.similar(A::GPUSparseMatrixCSR, ::Type{Tv}) where {Tv} =
    GPUSparseMatrixCSR(copy(A.rowPtr), copy(A.colVal), similar(A.nzVal, Tv), size(A))
Base.similar(A::GPUSparseMatrixCSC, ::Type{Tv}) where {Tv} =
    GPUSparseMatrixCSC(copy(A.colPtr), copy(A.rowVal), similar(A.nzVal, Tv), size(A))
function Base.similar(A::GPUSparseMatrixCSR, ::Type{Tv}, ::Type{Ti}) where {Tv,Ti}
    check_sparse_dims(Ti, size(A))
    check_sparse_nnz(Ti, nnz(A))
    GPUSparseMatrixCSR(convert_index_buffer(Ti, A.rowPtr), convert_index_buffer(Ti, A.colVal),
                       similar(A.nzVal, Tv), size(A))
end
function Base.similar(A::GPUSparseMatrixCSC, ::Type{Tv}, ::Type{Ti}) where {Tv,Ti}
    check_sparse_dims(Ti, size(A))
    check_sparse_nnz(Ti, nnz(A))
    GPUSparseMatrixCSC(convert_index_buffer(Ti, A.colPtr), convert_index_buffer(Ti, A.rowVal),
                       similar(A.nzVal, Tv), size(A))
end
function Base.similar(A::GPUSparseMatrixCOO, ::Type{Tv}, ::Type{Ti}) where {Tv,Ti}
    check_sparse_dims(Ti, size(A))
    GPUSparseMatrixCOO(convert_index_buffer(Ti, A.rowInd), convert_index_buffer(Ti, A.colInd),
                       similar(A.nzVal, Tv), size(A))
end
function Base.similar(x::GPUSparseVector, ::Type{Tv}, ::Type{Ti}) where {Tv,Ti}
    check_sparse_dims(Ti, size(x))
    GPUSparseVector(convert_index_buffer(Ti, x.nzInd), similar(x.nzVal, Tv), length(x))
end

Base.similar(A::GPUSparseArray, dims::Dims) = similar(A, eltype(A), dims)
Base.similar(A::GPUSparseArray, ::Type{Tv}, dims::Dims) where {Tv} =
    similar(A, Tv, indtype(A), dims)
Base.similar(A::Union{GPUSparseVector,GPUSparseMatrixCOO}, ::Type{Tv}, dims::Dims{2}) where {Tv} =
    empty_sparse(matrix_format(A), A.nzVal, Tv, indtype(A), dims, promote_type(Int, indtype(A)))
Base.similar(A::GPUSparseMatrixCompressed, ::Type{Tv}, dims::Dims{2}) where {Tv} =
    empty_sparse(matrix_format(A), A.nzVal, Tv, indtype(A), dims, eltype(compressed_ptr(A)))
Base.similar(A::GPUSparseArray, ::Type{Tv}, ::Type{Ti}, dims::Dims) where {Tv,Ti} =
    similar(A.nzVal, Tv, dims)
function Base.similar(A::GPUSparseArray, ::Type{Tv}, ::Type{Ti}, dims::Dims{1}) where {Tv,Ti}
    GPUSparseVector(similar(A.nzVal, Ti, 0), similar(A.nzVal, Tv, 0), dims[1])
end
function Base.similar(A::GPUSparseArray, ::Type{Tv}, ::Type{Ti}, dims::Dims{2}) where {Tv,Ti}
    empty_sparse(matrix_format(A), A.nzVal, Tv, Ti, dims)
end

# the matrix format that allocations derived from `A` use; vectors become CSC, as
# `similar(::SparseVector, T, (m, n))` does in SparseArrays
matrix_format(::GPUSparseMatrixCSR) = GPUSparseMatrixCSR
matrix_format(::GPUSparseMatrixCSC) = GPUSparseMatrixCSC
matrix_format(::GPUSparseMatrixCOO) = GPUSparseMatrixCOO
matrix_format(::GPUSparseVector) = GPUSparseMatrixCSC

# a matrix without stored entries, in format `S` and with the storage of `proto`
function empty_sparse(::Type{GPUSparseMatrixCSR}, proto, ::Type{Tv}, ::Type{Ti},
                      dims::Dims{2}, ::Type{Tp}=Ti) where {Tv,Ti,Tp}
    GPUSparseMatrixCSR(empty_ptr(proto, Tp, dims[1]), similar(proto, Ti, 0),
                       similar(proto, Tv, 0), dims)
end
function empty_sparse(::Type{GPUSparseMatrixCSC}, proto, ::Type{Tv}, ::Type{Ti},
                      dims::Dims{2}, ::Type{Tp}=Ti) where {Tv,Ti,Tp}
    GPUSparseMatrixCSC(empty_ptr(proto, Tp, dims[2]), similar(proto, Ti, 0),
                       similar(proto, Tv, 0), dims)
end
function empty_sparse(::Type{GPUSparseMatrixCOO}, proto, ::Type{Tv}, ::Type{Ti},
                      dims::Dims{2}, ::Type{Tp}=Ti) where {Tv,Ti,Tp}
    GPUSparseMatrixCOO(similar(proto, Ti, 0), similar(proto, Ti, 0),
                       similar(proto, Tv, 0), dims)
end

Base.zero(A::GPUSparseArray) = similar(A, eltype(A), size(A))

# Allocating format constructors and `copy` return independent storage. Raw buffer
# constructors borrow their arguments; identity `convert` and `adapt` may share.
Base.copy(A::GPUSparseMatrixCSR) =
    GPUSparseMatrixCSR(copy(A.rowPtr), copy(A.colVal), copy(A.nzVal), size(A))
Base.copy(A::GPUSparseMatrixCSC) =
    GPUSparseMatrixCSC(copy(A.colPtr), copy(A.rowVal), copy(A.nzVal), size(A))
Base.copy(A::GPUSparseMatrixCOO) =
    GPUSparseMatrixCOO(copy(A.rowInd), copy(A.colInd), copy(A.nzVal), size(A))
Base.copy(x::GPUSparseVector) =
    GPUSparseVector(copy(x.nzInd), copy(x.nzVal), length(x))

# Structural mutation requires resizable dense storage and exclusive access. Views are
# supported for reads and value updates, but cannot serve as structural destinations.
sparse_buffers(A::GPUSparseMatrixCSR) = (A.rowPtr, A.colVal, A.nzVal)
sparse_buffers(A::GPUSparseMatrixCSC) = (A.colPtr, A.rowVal, A.nzVal)
sparse_buffers(A::GPUSparseMatrixCOO) = (A.rowInd, A.colInd, A.nzVal)
sparse_buffers(A::GPUSparseVector) = (A.nzInd, A.nzVal)
Base.dataids(A::GPUSparseArray) = mapreduce(Base.dataids, (a, b) -> (a..., b...), sparse_buffers(A))
Base.unaliascopy(A::GPUSparseArray) = copy(A)

function check_sparse_mutation(A::GPUSparseArray)
    all(supports_resize, sparse_buffers(A)) ||
        throw(ArgumentError("structural mutation requires resizable dense buffers; raw views support reads and values-only updates"))
    return A
end

function copy_buffer!(dst::AbstractVector, src::AbstractVector)
    length(dst) == length(src) || resize!(dst, length(src))
    dst .= src
    return dst
end

function copy_sparse!(dst, src)
    size(dst) == size(src) ||
        throw(DimensionMismatch("destination has size $(size(dst)), source has size $(size(src))"))
    check_sparse_dims(indtype(dst), size(src))
    dst isa GPUSparseMatrixCompressed && check_sparse_nnz(eltype(compressed_ptr(dst)), nnz(src))
    check_sparse_mutation(dst)
    check_sparse_values(eltype(dst), nonzeros(src))
    dst === src && return dst
    # Copy before resizing/writing any component, including overlapping component views.
    src = Base.mightalias(dst, src) ? copy(src) : src
    foreach(copy_buffer!, sparse_buffers(dst), sparse_buffers(src))
    return dst
end
for S in (:GPUSparseMatrixCSR, :GPUSparseMatrixCSC, :GPUSparseMatrixCOO, :GPUSparseVector)
    @eval Base.copyto!(dst::$S, src::$S) = copy_sparse!(dst, src)
end


## structural checks

## COV_EXCL_START
# whether the indices in `ind[first:last]` are sorted, unique and within `1:n`
@inline function valid_slice(ind, first, last, n)
    prev = zero(eltype(ind))
    for k in first:last
        i = @inbounds ind[k]
        (i > prev && i <= n) || return false
        prev = i
    end
    return true
end

@kernel function check_compressed_kernel(valid, ptr, ind, n)
    j = @index(Global, Linear)
    if j < length(ptr)
        first = @inbounds ptr[j]
        last = @inbounds ptr[j+1] - one(first)
        @inbounds valid[j] = first >= 1 && last + 1 >= first && last <= length(ind) &&
                             valid_slice(ind, first, last, n)
    end
end

@kernel function check_coo_kernel(valid, rowInd, colInd, dims)
    k = @index(Global, Linear)
    if k <= length(rowInd)
        i = @inbounds rowInd[k]
        j = @inbounds colInd[k]
        ok = 1 <= i <= dims[1] && 1 <= j <= dims[2]
        if k > 1
            # row-major order without duplicates
            i′ = @inbounds rowInd[k-1]
            j′ = @inbounds colInd[k-1]
            ok &= i′ < i || (i′ == i && j′ < j)
        end
        @inbounds valid[k] = ok
    end
end
## COV_EXCL_STOP

"""
    GPUArrays.check_structure(A)

Verify on the device that the sparse array `A` satisfies the invariants of its format
(consistent pointers, sorted and unique indices within bounds), throwing an
`ArgumentError` otherwise. The constructors only check host metadata such as buffer
lengths, so this is useful for testing code that assembles sparse arrays from raw buffers.
"""
function check_structure(A::GPUSparseMatrixCompressed)
    ptr, ind = compressed_ptr(A), compressed_ind(A)
    n = major_dim(A)
    valid = similar(A.nzVal, Bool, n)
    if n > 0
        check_compressed_kernel(get_backend(A))(valid, ptr, ind, minor_dim(A); ndrange=n)
    end
    first, last = Array(ptr[[1, end]])
    (first == 1 && last == nnz(A) + 1 && all(valid)) ||
        throw(ArgumentError("invalid $(nameof(typeof(A))) structure"))
    return A
end
function check_structure(A::GPUSparseMatrixCOO)
    valid = similar(A.nzVal, Bool, nnz(A))
    if nnz(A) > 0
        check_coo_kernel(get_backend(A))(valid, A.rowInd, A.colInd, size(A); ndrange=nnz(A))
    end
    all(valid) || throw(ArgumentError("invalid GPUSparseMatrixCOO structure"))
    return A
end
function check_structure(x::GPUSparseVector)
    valid = similar(x.nzVal, Bool, 1)
    ptr = copyto!(similar(x.nzInd, Int, 2), Int[1, Base.checked_add(nnz(x), 1)])
    check_compressed_kernel(get_backend(x))(valid, ptr, x.nzInd, length(x); ndrange=1)
    all(valid) || throw(ArgumentError("invalid GPUSparseVector structure"))
    return x
end
