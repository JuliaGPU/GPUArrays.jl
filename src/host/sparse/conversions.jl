# conversions between sparse formats, and to and from the host

## to the host

SparseArrays.SparseMatrixCSC(A::GPUSparseMatrixCSC) =
    SparseMatrixCSC(size(A)..., Array(A.colPtr), Array(A.rowVal), Array(A.nzVal))
# the CSR buffers are those of the transpose in CSC
SparseArrays.SparseMatrixCSC(A::GPUSparseMatrixCSR) =
    copy(transpose(SparseMatrixCSC(reverse(size(A))..., Array(A.rowPtr), Array(A.colVal),
                                   Array(A.nzVal))))
function SparseArrays.SparseMatrixCSC(A::GPUSparseMatrixCOO)
    rowInd = Array(A.rowInd)
    rowPtr = zeros(eltype(rowInd), size(A, 1) + 1)
    rowPtr[1] = 1
    for i in rowInd
        rowPtr[i + 1] += one(eltype(rowPtr))
    end
    cumsum!(rowPtr, rowPtr)
    copy(transpose(SparseMatrixCSC(reverse(size(A))..., rowPtr, Array(A.colInd),
                                   Array(A.nzVal))))
end
SparseArrays.SparseMatrixCSC(x::GPUSparseVector) = SparseMatrixCSC(SparseVector(x))
SparseArrays.SparseMatrixCSC{Tv}(A::GPUSparseArray) where {Tv} =
    SparseMatrixCSC{Tv}(SparseMatrixCSC(A))
SparseArrays.SparseMatrixCSC{Tv,Ti}(A::GPUSparseArray) where {Tv,Ti} =
    SparseMatrixCSC{Tv,Ti}(SparseMatrixCSC(A))

SparseArrays.SparseVector(x::GPUSparseVector) =
    SparseVector(length(x), Array(x.nzInd), Array(x.nzVal))
SparseArrays.SparseVector{Tv}(x::GPUSparseVector) where {Tv} = SparseVector{Tv}(SparseVector(x))
SparseArrays.SparseVector{Tv,Ti}(x::GPUSparseVector) where {Tv,Ti} =
    SparseVector{Tv,Ti}(SparseVector(x))

host_sparse(A::GPUSparseMatrix) = SparseMatrixCSC(A)
host_sparse(x::GPUSparseVector) = SparseVector(x)

Base.Array(A::GPUSparseArray) = Array(host_sparse(A))
Base.Array{T}(A::GPUSparseArray) where {T} = Array{T}(host_sparse(A))
Base.Array{T,1}(x::GPUSparseVector) where {T} = Array{T,1}(host_sparse(x))
Base.Array{T,2}(A::GPUSparseMatrix) where {T} = Array{T,2}(host_sparse(A))
Base.collect(A::GPUSparseArray) = Array(A)


## `adapt`

# Kernel arguments and other adaptors: convert every buffer. With device vectors, the
# result is the isbits representation that kernels receive. Host buffers give SparseArrays'
# own types where there is one.
Adapt.adapt_structure(to, A::GPUSparseMatrixCSR) =
    GPUSparseMatrixCSR(adapt(to, A.rowPtr), adapt(to, A.colVal), adapt(to, A.nzVal), size(A))
Adapt.adapt_structure(to, A::GPUSparseMatrixCSC) =
    sparse_csc(adapt(to, A.colPtr), adapt(to, A.rowVal), adapt(to, A.nzVal), size(A))
Adapt.adapt_structure(to, A::GPUSparseMatrixCOO) =
    GPUSparseMatrixCOO(adapt(to, A.rowInd), adapt(to, A.colInd), adapt(to, A.nzVal), size(A))
Adapt.adapt_structure(to, x::GPUSparseVector) =
    sparse_vector(adapt(to, x.nzInd), adapt(to, x.nzVal), length(x))

sparse_csc(colPtr::Vector, rowVal::Vector, nzVal::Vector, dims) =
    SparseMatrixCSC(dims..., colPtr, rowVal, nzVal)
sparse_csc(colPtr, rowVal, nzVal, dims) = GPUSparseMatrixCSC(colPtr, rowVal, nzVal, dims)
sparse_vector(nzInd::Vector, nzVal::Vector, len) = SparseVector(len, nzInd, nzVal)
sparse_vector(nzInd, nzVal, len) = GPUSparseVector(nzInd, nzVal, len)

# Storage types (`adapt(Array, A)`, `adapt(MtlArray, S)`): move the buffers, keeping the
# format, element and index types. The index buffers are allocated like the adapted
# values, so that a target with an element type (`Array{Float32}`, or the
# `CuArray{Float32,1,M}` that `cu` adapts to) only converts the values.
move_buffer(proto::AbstractVector, src::AbstractVector) =
    copyto!(similar(proto, eltype(src), length(src)), src)

function Adapt.adapt_structure(to::Type{<:DenseArray}, A::GPUSparseMatrixCSR)
    nzVal = adapt(to, A.nzVal)
    nzVal === A.nzVal && return A
    GPUSparseMatrixCSR(move_buffer(nzVal, A.rowPtr), move_buffer(nzVal, A.colVal), nzVal,
                       size(A))
end
function Adapt.adapt_structure(to::Type{<:DenseArray}, A::GPUSparseMatrixCSC)
    nzVal = adapt(to, A.nzVal)
    colPtr, rowVal = nzVal === A.nzVal ? (A.colPtr, A.rowVal) :
                     (move_buffer(nzVal, A.colPtr), move_buffer(nzVal, A.rowVal))
    sparse_csc(colPtr, rowVal, nzVal, size(A))
end
function Adapt.adapt_structure(to::Type{<:DenseArray}, A::GPUSparseMatrixCOO)
    nzVal = adapt(to, A.nzVal)
    nzVal === A.nzVal && return A
    GPUSparseMatrixCOO(move_buffer(nzVal, A.rowInd), move_buffer(nzVal, A.colInd), nzVal,
                       size(A))
end
function Adapt.adapt_structure(to::Type{<:DenseArray}, x::GPUSparseVector)
    nzVal = adapt(to, x.nzVal)
    nzInd = nzVal === x.nzVal ? x.nzInd : move_buffer(nzVal, x.nzInd)
    sparse_vector(nzInd, nzVal, length(x))
end

# host sparse arrays to GPU storage, likewise keeping format and types: a CSC matrix stays
# CSC. Adapting never densifies.
function Adapt.adapt_structure(to::Type{<:AbstractGPUArray}, S::SparseMatrixCSC)
    nzVal = adapt(to, nonzeros(S))
    GPUSparseMatrixCSC(move_buffer(nzVal, getcolptr(S)), move_buffer(nzVal, rowvals(S)),
                       nzVal, size(S))
end
function Adapt.adapt_structure(to::Type{<:AbstractGPUArray}, x::SparseVector)
    nzVal = adapt(to, nonzeros(x))
    GPUSparseVector(move_buffer(nzVal, SparseArrays.nonzeroinds(x)), nzVal, length(x))
end

# explicit sparse types (`adapt(MtlSparseMatrixCSR{Float32,Int32}, S)`) are constructors
for S in (:SparseMatrixCSC, :SparseVector, :GPUSparseMatrixCSR, :GPUSparseMatrixCSC,
          :GPUSparseMatrixCOO, :GPUSparseVector)
    @eval Adapt.adapt_structure(::Type{T}, A::$S) where {T<:AbstractGPUSparseArray} = T(A)
end


## between formats

# the generic types do not determine a storage, so they cannot be constructed from host
# arrays: that needs a back-end alias, or `adapt` to a storage type
for S in (:GPUSparseMatrixCSR, :GPUSparseMatrixCSC, :GPUSparseMatrixCOO, :GPUSparseVector)
    @eval function $S(A::Union{SparseMatrixCSC,SparseVector})
        throw(ArgumentError("""$($S)(::$(nameof(typeof(A)))) does not know where to store the result.
                               Use a back-end alias (e.g. `MtlSparseMatrixCSR(S)`), `adapt(MtlArray, S)` to keep the format, or `mtl(S)`/`cu(S)`."""))
    end
end

# the same format: a copy
GPUSparseMatrixCSR(A::GPUSparseMatrixCSR) = copy(A)
GPUSparseMatrixCSC(A::GPUSparseMatrixCSC) = copy(A)
GPUSparseMatrixCOO(A::GPUSparseMatrixCOO) = copy(A)
GPUSparseVector(x::GPUSparseVector) = copy(x)

# typed forms: convert the format, then the element and index types
for S in (:GPUSparseMatrixCSR, :GPUSparseMatrixCSC, :GPUSparseMatrixCOO, :GPUSparseVector)
    @eval begin
        $S{Tv}(A::AbstractGPUSparseArray) where {Tv} = $S{Tv,indtype(A)}(A)
        $S{Tv,Ti}(A::AbstractGPUSparseArray) where {Tv,Ti} = with_eltypes($S(A), Tv, Ti)
        $S{Tv,Ti}(A::$S) where {Tv,Ti} = with_eltypes(A, Tv, Ti; copy=true)
    end
end

# `A` with the given element and index types, reusing its buffers where they already match
# unless `copy` is set
function with_eltypes(A::GPUSparseArray, ::Type{Tv}, ::Type{Ti}; copy::Bool=false) where {Tv,Ti}
    Tv == eltype(A) && Ti == indtype(A) && return copy ? Base.copy(A) : A
    check_sparse_dims(Ti, size(A))
    check_sparse_nnz(Ti, nnz(A))
    buffers = map(fieldnames(typeof(A))) do field
        buf = getfield(A, field)
        if field === :nzVal
            convert_buffer(Tv, buf)
        elseif buf isa AbstractVector
            convert_buffer(Ti, buf)
        else
            buf
        end
    end
    return sparse_format_type(A)(buffers...)
end
sparse_format_type(::GPUSparseMatrixCSR) = GPUSparseMatrixCSR
sparse_format_type(::GPUSparseMatrixCSC) = GPUSparseMatrixCSC
sparse_format_type(::GPUSparseMatrixCOO) = GPUSparseMatrixCOO
sparse_format_type(::GPUSparseVector) = GPUSparseVector

"""
    GPUArrays.generic_regroup(A::GPUSparseMatrixCSR)::GPUSparseMatrixCSC
    GPUArrays.generic_regroup(A::GPUSparseMatrixCSC)::GPUSparseMatrixCSR

Convert between CSR and CSC by regrouping the stored entries along the other dimension, on
the device. This is the generic implementation behind `GPUSparseMatrixCSC(::GPUSparseMatrixCSR)`
and vice versa, which back-ends can call when a vendor routine does not apply.
"""
function generic_regroup(A::GPUSparseMatrixCSR)
    colPtr, rowVal, nzVal = regroup(A.rowPtr, A.colVal, A.nzVal, size(A))
    GPUSparseMatrixCSC(colPtr, rowVal, nzVal, size(A))
end
function generic_regroup(A::GPUSparseMatrixCSC)
    rowPtr, colVal, nzVal = regroup(A.colPtr, A.rowVal, A.nzVal, reverse(size(A)))
    GPUSparseMatrixCSR(rowPtr, colVal, nzVal, size(A))
end
GPUSparseMatrixCSC(A::GPUSparseMatrixCSR) = generic_regroup(A)
GPUSparseMatrixCSR(A::GPUSparseMatrixCSC) = generic_regroup(A)

"""
    GPUArrays.generic_expand(A::GPUSparseMatrixCSR)::GPUSparseMatrixCOO
    GPUArrays.generic_compress(A::GPUSparseMatrixCOO)::GPUSparseMatrixCSR

Convert between CSR and COO on the device, which have the same order of stored entries:
expand the row pointers into a row index per entry, or compress them back.
"""
generic_expand(A::GPUSparseMatrixCSR) =
    GPUSparseMatrixCOO(expand_ptr(A.rowPtr, nnz(A)), copy(A.colVal), copy(A.nzVal), size(A))
generic_compress(A::GPUSparseMatrixCOO) =
    GPUSparseMatrixCSR(compress(A.rowInd, size(A, 1)), copy(A.colInd), copy(A.nzVal), size(A))
GPUSparseMatrixCOO(A::GPUSparseMatrixCSR) = generic_expand(A)
GPUSparseMatrixCSR(A::GPUSparseMatrixCOO) = generic_compress(A)

# CSC ↔ COO through CSR, borrowing the buffers of the intermediate
function GPUSparseMatrixCOO(A::GPUSparseMatrixCSC)
    B = GPUSparseMatrixCSR(A)
    GPUSparseMatrixCOO(expand_ptr(B.rowPtr, nnz(B)), B.colVal, B.nzVal, size(A))
end
function GPUSparseMatrixCSC(A::GPUSparseMatrixCOO)
    B = GPUSparseMatrixCSR(compress(A.rowInd, size(A, 1)), A.colInd, A.nzVal, size(A))
    generic_regroup(B)
end

# The CSR buffers of `A` are the CSC buffers of `transpose(A)`, and vice versa. This
# borrowed view shares the buffers of `A`; it is only used internally, to share kernels
# between formats, and never outlives the operation that creates it.
transpose_view(A::GPUSparseMatrixCSR) =
    GPUSparseMatrixCSC(A.rowPtr, A.colVal, A.nzVal, reverse(size(A)))
transpose_view(A::GPUSparseMatrixCSC) =
    GPUSparseMatrixCSR(A.colPtr, A.rowVal, A.nzVal, reverse(size(A)))

# materializing a transpose into the opposite format only copies the buffers
GPUSparseMatrixCSC(A::Transpose{<:Any,<:GPUSparseMatrixCSR}) = copy(transpose_view(parent(A)))
GPUSparseMatrixCSR(A::Transpose{<:Any,<:GPUSparseMatrixCSC}) = copy(transpose_view(parent(A)))
function GPUSparseMatrixCSC(A::Adjoint{<:Any,<:GPUSparseMatrixCSR})
    B = transpose_view(parent(A))
    GPUSparseMatrixCSC(copy(B.colPtr), copy(B.rowVal), conj.(B.nzVal), size(B))
end
function GPUSparseMatrixCSR(A::Adjoint{<:Any,<:GPUSparseMatrixCSC})
    B = transpose_view(parent(A))
    GPUSparseMatrixCSR(copy(B.rowPtr), copy(B.colVal), conj.(B.nzVal), size(B))
end

Base.convert(::Type{T}, A::AbstractArray) where {T<:AbstractGPUSparseArray} =
    A isa T ? A : T(A)


## to dense arrays

## COV_EXCL_START
# every thread scatters one slice of a compressed matrix into `dst`, indexed linearly
# as an `m`-row matrix
@kernel function densify_compressed_kernel(dst, ptr, ind, nzVal, m, csr::Bool)
    j = @index(Global, Linear)
    if j < length(ptr)
        for k in @inbounds(ptr[j]):@inbounds(ptr[j+1] - one(eltype(ptr)))
            i = @inbounds ind[k]
            row, col = csr ? (j, Int(i)) : (Int(i), j)
            @inbounds dst[(col - 1) * m + row] = nzVal[k]
        end
    end
end

@kernel function densify_coo_kernel(dst, rowInd, colInd, nzVal, m)
    k = @index(Global, Linear)
    if k <= length(nzVal)
        @inbounds dst[(Int(colInd[k]) - 1) * m + rowInd[k]] = nzVal[k]
    end
end

@kernel function densify_vector_kernel(dst, nzInd, nzVal)
    k = @index(Global, Linear)
    if k <= length(nzVal)
        @inbounds dst[nzInd[k]] = nzVal[k]
    end
end
## COV_EXCL_STOP

# densify on the device, into the first `length(src)` elements of `dst` (in column-major
# order, like `copyto!` between dense arrays). The formats are canonical, so every element
# is written at most once.
function Base.copyto!(dst::AbstractGPUArray, src::GPUSparseArray)
    length(dst) >= length(src) || throw(BoundsError(dst, length(src)))
    fill!(length(dst) == length(src) ? dst : view(dst, 1:length(src)), zero(eltype(dst)))
    nnz(src) == 0 && return dst
    backend = get_backend(dst)
    if src isa GPUSparseMatrixCompressed
        densify_compressed_kernel(backend)(dst, compressed_ptr(src), compressed_ind(src),
                                           src.nzVal, size(src, 1), src isa GPUSparseMatrixCSR;
                                           ndrange=major_dim(src))
    elseif src isa GPUSparseMatrixCOO
        densify_coo_kernel(backend)(dst, src.rowInd, src.colInd, src.nzVal, size(src, 1);
                                    ndrange=nnz(src))
    else
        densify_vector_kernel(backend)(dst, src.nzInd, src.nzVal; ndrange=nnz(src))
    end
    return dst
end

Base.copyto!(dst::Array, src::GPUSparseArray) = copyto!(dst, Array(src))

