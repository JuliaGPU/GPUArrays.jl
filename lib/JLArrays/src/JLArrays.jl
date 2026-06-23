# reference implementation on the CPU
# This acts as a wrapper around KernelAbstractions's parallel CPU
# functionality. It is useful for testing GPUArrays (and other packages)
# when no GPU is present.
# This file follows conventions from AMDGPU.jl

module JLArrays

export JLArray, JLVector, JLMatrix, jl, JLBackend, JLSparseVector,
       JLSparseMatrixCSC, JLSparseMatrixCSR, JLSparseMatrixCOO

using GPUArrays

using Adapt
using SparseArrays, LinearAlgebra

import KernelAbstractions
import KernelAbstractions: Adapt, StaticArrays, Backend, Kernel, StaticSize, DynamicSize, partition, blocks, workitems, launch_config

@static if isdefined(JLArrays.KernelAbstractions, :POCL) # KA v0.10
    import KernelAbstractions: POCL
end

module AS

const Generic  = 0

end

#
# Device functionality
#

const MAXTHREADS = 256

struct JLBackend <: KernelAbstractions.GPU
    static::Bool
    JLBackend(;static::Bool=false) = new(static)
end

struct Adaptor end
jlconvert(arg) = adapt(Adaptor(), arg)

# FIXME: add Ref to Adapt.jl (but make sure it doesn't cause ambiguities with CUDAnative's)
struct JlRefValue{T} <: Ref{T}
  x::T
end
Base.getindex(r::JlRefValue) = r.x
Adapt.adapt_structure(to::Adaptor, r::Base.RefValue) = JlRefValue(adapt(to, r[]))

## executed on-device

# array type
@static if !isdefined(JLArrays.KernelAbstractions, :POCL) # KA v0.9
    struct JLDeviceArray{T, N} <: AbstractDeviceArray{T, N}
        data::Vector{UInt8}
        offset::Int
        dims::Dims{N}
    end

    Base.elsize(::Type{<:JLDeviceArray{T}}) where {T} = sizeof(T)

    Base.size(x::JLDeviceArray) = x.dims
    Base.sizeof(x::JLDeviceArray) = Base.elsize(x) * length(x)

    Base.unsafe_convert(::Type{Ptr{T}}, x::JLDeviceArray{T}) where {T} =
        convert(Ptr{T}, pointer(x.data)) + x.offset

    # conversion of untyped data to a typed Array
    function typed_data(x::JLDeviceArray{T}) where {T}
        unsafe_wrap(Array, pointer(x), x.dims)
    end

    @inline Base.getindex(A::JLDeviceArray, index::Integer) = getindex(typed_data(A), index)
    @inline Base.setindex!(A::JLDeviceArray, x, index::Integer) = setindex!(typed_data(A), x, index)
    Base.pointer(x::JLDeviceArray{T}) where {T} =
        convert(Ptr{T}, pointer(x.data)) + x.offset
    @inline function Base.pointer(x::JLDeviceArray{T}, i::Integer) where T
        pointer(x) + (i - 1) * sizeof(T)
    end
end

#
# Host abstractions
#

function check_eltype(T)
  if !Base.allocatedinline(T)
    explanation = explain_allocatedinline(T)
    error("""
      JLArray only supports element types that are allocated inline.
      $explanation""")
  end
end

mutable struct JLArray{T, N} <: AbstractGPUArray{T, N}
    data::DataRef{Vector{UInt8}}

    offset::Int        # offset of the data in the buffer, in bytes

    dims::Dims{N}

    # allocating constructor
    function JLArray{T,N}(::UndefInitializer, dims::Dims{N}) where {T,N}
        check_eltype(T)
        maxsize = prod(dims) * sizeof(T)

        ref = GPUArrays.cached_alloc((JLArray, maxsize)) do
            data = Vector{UInt8}(undef, maxsize)
            DataRef(data) do data
                resize!(data, 0)
            end
        end

        obj = new{T, N}(ref, 0, dims)
        finalizer(unsafe_free!, obj)
        return obj
    end

    # low-level constructor for wrapping existing data
    function JLArray{T,N}(ref::DataRef{Vector{UInt8}}, dims::Dims{N};
                          offset::Int=0) where {T,N}
        check_eltype(T)
        obj = new{T,N}(ref, offset, dims)
        finalizer(unsafe_free!, obj)
        return obj
    end
end

mutable struct JLSparseVector{Tv, Ti} <: GPUArrays.AbstractGPUSparseVector{Tv, Ti}
    iPtr::JLArray{Ti, 1}
    nzVal::JLArray{Tv, 1}
    len::Int
    nnz::Ti

    function JLSparseVector{Tv, Ti}(iPtr::JLArray{<:Integer, 1}, nzVal::JLArray{Tv, 1},
                                    len::Integer) where {Tv, Ti <: Integer}
        new{Tv, Ti}(iPtr, nzVal, len, length(nzVal))
    end
end
SparseArrays.nnz(x::JLSparseVector)          = x.nnz
SparseArrays.nonzeroinds(x::JLSparseVector)  = x.iPtr
SparseArrays.nonzeros(x::JLSparseVector)     = x.nzVal

mutable struct JLSparseMatrixCSC{Tv, Ti} <: GPUArrays.AbstractGPUSparseMatrixCSC{Tv, Ti}
    colPtr::JLArray{Ti, 1}
    rowVal::JLArray{Ti, 1}
    nzVal::JLArray{Tv, 1}
    dims::NTuple{2,Int}
    nnz::Ti

    function JLSparseMatrixCSC{Tv, Ti}(colPtr::JLArray{<:Integer, 1}, rowVal::JLArray{<:Integer, 1},
                                       nzVal::JLArray{Tv, 1}, dims::NTuple{2,<:Integer}) where {Tv, Ti <: Integer}
        new{Tv, Ti}(colPtr, rowVal, nzVal, dims, length(nzVal))
    end
end
function JLSparseMatrixCSC(colPtr::JLArray{Ti, 1}, rowVal::JLArray{Ti, 1}, nzVal::JLArray{Tv, 1}, dims::NTuple{2,<:Integer}) where {Tv, Ti <: Integer}
    return JLSparseMatrixCSC{Tv, Ti}(colPtr, rowVal, nzVal, dims)
end
SparseArrays.SparseMatrixCSC(x::JLSparseMatrixCSC) = SparseMatrixCSC(size(x)..., Array(x.colPtr), Array(x.rowVal), Array(x.nzVal))

JLSparseMatrixCSC(A::JLSparseMatrixCSC) = A

function Base.getindex(A::JLSparseMatrixCSC{Tv, Ti}, i::Integer, j::Integer) where {Tv, Ti}
    @boundscheck checkbounds(A, i, j)
    r1 = Int(@inbounds A.colPtr[j])
    r2 = Int(@inbounds A.colPtr[j+1]-1)
    (r1 > r2) && return zero(Tv)
    r1 = searchsortedfirst(view(A.rowVal, r1:r2), i) + r1 - 1
    ((r1 > r2) || (A.rowVal[r1] != i)) ? zero(Tv) : A.nzVal[r1]
end

mutable struct JLSparseMatrixCSR{Tv, Ti} <: GPUArrays.AbstractGPUSparseMatrixCSR{Tv, Ti}
    rowPtr::JLArray{Ti, 1}
    colVal::JLArray{Ti, 1}
    nzVal::JLArray{Tv, 1}
    dims::NTuple{2,Int}
    nnz::Ti

    function JLSparseMatrixCSR{Tv, Ti}(rowPtr::JLArray{<:Integer, 1}, colVal::JLArray{<:Integer, 1},
                                       nzVal::JLArray{Tv, 1}, dims::NTuple{2,<:Integer}) where {Tv, Ti<:Integer}
        new{Tv, Ti}(rowPtr, colVal, nzVal, dims, length(nzVal))
    end
end
function JLSparseMatrixCSR(rowPtr::JLArray{Ti, 1}, colVal::JLArray{Ti, 1}, nzVal::JLArray{Tv, 1}, dims::NTuple{2,<:Integer}) where {Tv, Ti <: Integer}
    return JLSparseMatrixCSR{Tv, Ti}(rowPtr, colVal, nzVal, dims)
end
function SparseArrays.SparseMatrixCSC(x::JLSparseMatrixCSR)
    x_transpose = SparseMatrixCSC(size(x, 2), size(x, 1), Array(x.rowPtr), Array(x.colVal), Array(x.nzVal))
    return SparseMatrixCSC(transpose(x_transpose))
end

JLSparseMatrixCSC(Mat::Union{Transpose{Tv, <:SparseMatrixCSC}, Adjoint{Tv, <:SparseMatrixCSC}}) where {Tv} = JLSparseMatrixCSC(JLSparseMatrixCSR(Mat))

mutable struct JLSparseMatrixCOO{Tv, Ti} <: GPUArrays.AbstractGPUSparseMatrixCOO{Tv, Ti}
    rowInd::JLArray{Ti, 1}
    colInd::JLArray{Ti, 1}
    nzVal::JLArray{Tv, 1}
    dims::NTuple{2,Int}
    nnz::Ti

    function JLSparseMatrixCOO{Tv, Ti}(rowInd::JLArray{Ti, 1}, colInd::JLArray{Ti, 1},
                                       nzVal::JLArray{Tv, 1}, dims::NTuple{2,<:Integer},
                                       nnz::Integer=length(nzVal)) where {Tv, Ti <: Integer}
        length(rowInd) == length(colInd) == length(nzVal) ||
            throw(DimensionMismatch("COO row, column, and value arrays have different lengths"))
        new{Tv, Ti}(rowInd, colInd, nzVal, (Int(dims[1]), Int(dims[2])), Ti(nnz))
    end
end
JLSparseMatrixCOO(rowInd::JLArray{Ti, 1}, colInd::JLArray{Ti, 1},
                  nzVal::JLArray{Tv, 1}, dims::NTuple{2,<:Integer},
                  nnz::Integer=length(nzVal)) where {Tv, Ti <: Integer} =
    JLSparseMatrixCOO{Tv, Ti}(rowInd, colInd, nzVal, dims, nnz)

Base.length(x::JLSparseMatrixCOO) = prod(x.dims)
Base.size(x::JLSparseMatrixCOO) = x.dims
SparseArrays.nnz(x::JLSparseMatrixCOO) = x.nnz
SparseArrays.nonzeros(x::JLSparseMatrixCOO) = x.nzVal

function SparseArrays.SparseMatrixCSC(x::JLSparseMatrixCOO)
    return sparse(Array(x.rowInd), Array(x.colInd), Array(x.nzVal), size(x)...)
end

function Base.getindex(A::JLSparseMatrixCOO{Tv}, i::Integer, j::Integer) where {Tv}
    @boundscheck checkbounds(A, i, j)
    acc = zero(Tv)
    for k in 1:Int(nnz(A))
        @inbounds if A.rowInd[k] == i && A.colInd[k] == j
            acc += A.nzVal[k]
        end
    end
    return acc
end

function Base.size(g::JLSparseMatrixCSR, d::Integer)
    if 1 <= d <= 2
        return g.dims[d]
    elseif d > 1
        return 1
    else
        throw(ArgumentError("dimension must be ≥ 1, got $d"))
    end
end

JLSparseMatrixCSR(Mat::Transpose{Tv, <:SparseMatrixCSC}) where {Tv} =
    JLSparseMatrixCSR(JLVector{Cint}(parent(Mat).colptr), JLVector{Cint}(parent(Mat).rowval),
                         JLVector(parent(Mat).nzval), size(Mat))
JLSparseMatrixCSR(Mat::Adjoint{Tv, <:SparseMatrixCSC}) where {Tv} =
    JLSparseMatrixCSR(JLVector{Cint}(parent(Mat).colptr), JLVector{Cint}(parent(Mat).rowval),
                         JLVector(conj.(parent(Mat).nzval)), size(Mat))

JLSparseMatrixCSR(A::JLSparseMatrixCSR) = A

function Base.getindex(A::JLSparseMatrixCSR{Tv, Ti}, i0::Integer, i1::Integer) where {Tv, Ti}
    @boundscheck checkbounds(A, i0, i1)
    c1 = Int(A.rowPtr[i0])
    c2 = Int(A.rowPtr[i0+1]-1)
    (c1 > c2) && return zero(Tv)
    c1 = searchsortedfirst(A.colVal, i1, c1, c2, Base.Order.Forward)
    (c1 > c2 || A.colVal[c1] != i1) && return zero(Tv)
    nonzeros(A)[c1]
end

GPUArrays.storage(a::JLArray) = a.data
# format-conversion verbs (cross-format only; the identity cases are handled generically
# by GPUArrays). these reuse the format constructors defined below.
GPUArrays.sparse_csc(A::Union{JLSparseMatrixCSR,JLSparseMatrixCOO}) = JLSparseMatrixCSC(A)
GPUArrays.sparse_csr(A::Union{JLSparseMatrixCSC,JLSparseMatrixCOO}) = JLSparseMatrixCSR(A)
GPUArrays.sparse_coo(A::Union{JLSparseMatrixCSC,JLSparseMatrixCSR}) = JLSparseMatrixCOO(A)

Base.similar(Mat::JLSparseMatrixCSR) = JLSparseMatrixCSR(copy(Mat.rowPtr), copy(Mat.colVal), similar(nonzeros(Mat)), size(Mat))
Base.similar(Mat::JLSparseMatrixCSR, T::Type) = JLSparseMatrixCSR(copy(Mat.rowPtr), copy(Mat.colVal), similar(nonzeros(Mat), T), size(Mat))
Base.similar(Mat::JLSparseMatrixCOO) = JLSparseMatrixCOO(copy(Mat.rowInd), copy(Mat.colInd), similar(nonzeros(Mat)), size(Mat), nnz(Mat))
Base.similar(Mat::JLSparseMatrixCOO, T::Type) = JLSparseMatrixCOO(copy(Mat.rowInd), copy(Mat.colInd), similar(nonzeros(Mat), T), size(Mat), nnz(Mat))

# CSC and vector structure-preserving `similar` (mirror the CSR/COO methods above; these
# used to come for free from GPUArrays via a type-reconstruction helper).
Base.similar(Mat::JLSparseMatrixCSC{<:Any, Ti}, ::Type{T}) where {T, Ti} =
    JLSparseMatrixCSC{T, Ti}(copy(Mat.colPtr), copy(Mat.rowVal), similar(nonzeros(Mat), T), size(Mat))
Base.similar(Vec::JLSparseVector{<:Any, Ti}, ::Type{T}) where {T, Ti} =
    JLSparseVector{T, Ti}(copy(Vec.iPtr), similar(nonzeros(Vec), T), length(Vec))

# empty sparse arrays of a given shape, used by the generic broadcast machinery as the
# output container before its storage is sized and filled. the index type is preserved.
Base.similar(Mat::JLSparseMatrixCSC{<:Any, Ti}, ::Type{T}, N::Int, M::Int) where {T, Ti} =
    JLSparseMatrixCSC{T, Ti}(JLVector(ones(Ti, M + 1)), JLVector{Ti}(undef, 0), JLVector{T}(undef, 0), (N, M))
Base.similar(Mat::JLSparseMatrixCSR{<:Any, Ti}, ::Type{T}, N::Int, M::Int) where {T, Ti} =
    JLSparseMatrixCSR{T, Ti}(JLVector(ones(Ti, N + 1)), JLVector{Ti}(undef, 0), JLVector{T}(undef, 0), (N, M))
Base.similar(Vec::JLSparseVector{<:Any, Ti}, ::Type{T}, dims::Dims{1}) where {T, Ti} =
    JLSparseVector{T, Ti}(JLVector{Ti}(undef, 0), JLVector{T}(undef, 0), dims[1])

Base.similar(Mat::JLSparseMatrixCSC{Tv, Ti}, N::Int, M::Int) where {Tv, Ti} = similar(Mat, Tv, N, M)
Base.similar(Mat::JLSparseMatrixCSR{Tv, Ti}, N::Int, M::Int) where {Tv, Ti} = similar(Mat, Tv, N, M)

Base.similar(Mat::JLSparseMatrixCSC, T::Type, dims::Tuple{Int, Int}) = similar(Mat, T, dims...)
Base.similar(Mat::JLSparseMatrixCSR, T::Type, dims::Tuple{Int, Int}) = similar(Mat, T, dims...)

Base.similar(Mat::JLSparseMatrixCSC, dims::Tuple{Int, Int}) = similar(Mat, dims...)
Base.similar(Mat::JLSparseMatrixCSR, dims::Tuple{Int, Int}) = similar(Mat, dims...)

JLArray(x::JLSparseVector)    = GPUArrays.to_dense(x)
JLArray(x::JLSparseMatrixCSC) = GPUArrays.to_dense(x)
JLArray(x::JLSparseMatrixCSR) = GPUArrays.to_dense(x)
JLArray(x::JLSparseMatrixCOO) = GPUArrays.to_dense(x)

# conversion of untyped data to a typed Array
function typed_data(x::JLArray{T}) where {T}
    unsafe_wrap(Array, pointer(x), x.dims)
end

function GPUArrays.derive(::Type{T}, a::JLArray, dims::Dims{N}, offset::Int) where {T,N}
    ref = copy(a.data)
    offset = a.offset + offset * sizeof(T)
    JLArray{T,N}(ref, dims; offset)
end


## convenience constructors

const JLVector{T} = JLArray{T,1}
const JLMatrix{T} = JLArray{T,2}
const JLVecOrMat{T} = Union{JLVector{T},JLMatrix{T}}

# type and dimensionality specified
JLArray{T,N}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N} =
    JLArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))
JLArray{T,N}(::UndefInitializer, dims::Vararg{Integer,N}) where {T,N} =
    JLArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))

# type but not dimensionality specified
JLArray{T}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N} =
  JLArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))
JLArray{T}(::UndefInitializer, dims::Vararg{Integer,N}) where {T,N} =
  JLArray{T,N}(undef, convert(Dims{N}, dims))

# empty vector constructor
JLArray{T,1}() where {T} = JLArray{T,1}(undef, 0)

Base.similar(a::JLArray{T,N}) where {T,N} = JLArray{T,N}(undef, size(a))
Base.similar(a::JLArray{T}, dims::Base.Dims{N}) where {T,N} = JLArray{T,N}(undef, dims)
Base.similar(a::JLArray, ::Type{T}, dims::Base.Dims{N}) where {T,N} = JLArray{T,N}(undef, dims)

function Base.copy(a::JLArray{T,N}) where {T,N}
    b = similar(a)
    @inbounds copyto!(b, a)
end


## derived types

export DenseJLArray, DenseJLVector, DenseJLMatrix, DenseJLVecOrMat,
       StridedJLArray, StridedJLVector, StridedJLMatrix, StridedJLVecOrMat,
       AnyJLArray, AnyJLVector, AnyJLMatrix, AnyJLVecOrMat

# dense arrays: stored contiguously in memory
DenseJLArray{T,N} = JLArray{T,N}
DenseJLVector{T} = DenseJLArray{T,1}
DenseJLMatrix{T} = DenseJLArray{T,2}
DenseJLVecOrMat{T} = Union{DenseJLVector{T}, DenseJLMatrix{T}}

# strided arrays
StridedSubJLArray{T,N,I<:Tuple{Vararg{Union{Base.RangeIndex, Base.ReshapedUnitRange,
                                            Base.AbstractCartesianIndex}}}} =
  SubArray{T,N,<:JLArray,I}
StridedJLArray{T,N} = Union{JLArray{T,N}, StridedSubJLArray{T,N}}
StridedJLVector{T} = StridedJLArray{T,1}
StridedJLMatrix{T} = StridedJLArray{T,2}
StridedJLVecOrMat{T} = Union{StridedJLVector{T}, StridedJLMatrix{T}}

# Pointer access is only available for callers that explicitly want a pointer.
Base.pointer(x::JLArray{T}) where {T} =
    convert(Ptr{T}, pointer(x.data[])) + x.offset
@inline function Base.pointer(x::JLArray{T}, i::Integer) where T
    pointer(x) + (i - 1) * sizeof(T)
end

# anything that's (secretly) backed by a JLArray
AnyJLArray{T,N} = Union{JLArray{T,N}, WrappedArray{T,N,JLArray,JLArray{T,N}}}
AnyJLVector{T} = AnyJLArray{T,1}
AnyJLMatrix{T} = AnyJLArray{T,2}
AnyJLVecOrMat{T} = Union{AnyJLVector{T}, AnyJLMatrix{T}}


## array interface

Base.elsize(::Type{<:JLArray{T}}) where {T} = sizeof(T)

Base.size(x::JLArray) = x.dims
Base.sizeof(x::JLArray) = Base.elsize(x) * length(x)

# Reject implicit conversions to a pointer, i.e., by passing to a CPU ccall.
function Base.unsafe_convert(::Type{Ptr{T}}, x::JLArray{T}) where {T}
    error("Illegal conversion of a JLArray to a Ptr")
end


## interop with Julia arrays

function JLArray{T,N}(xs::AbstractArray{<:Any,N}) where {T,N}
    A = JLArray{T,N}(undef, size(xs))
    copyto!(A, convert(Array{T}, xs))
    return A
end

# underspecified constructors
JLArray{T}(xs::AbstractArray{S,N}) where {T,N,S} = JLArray{T,N}(xs)
(::Type{JLArray{T,N} where T})(x::AbstractArray{S,N}) where {S,N} = JLArray{S,N}(x)
JLArray(A::AbstractArray{T,N}) where {T,N} = JLArray{T,N}(A)

function JLSparseVector(xs::SparseVector{Tv, Ti}) where {Ti, Tv}
    iPtr  = JLVector{Ti}(undef, length(xs.nzind))
    nzVal = JLVector{Tv}(undef, length(xs.nzval))
    copyto!(iPtr, convert(Vector{Ti}, xs.nzind))
    copyto!(nzVal, convert(Vector{Tv}, xs.nzval))
    return JLSparseVector{Tv, Ti}(iPtr, nzVal, length(xs),)
end
SparseArrays.SparseVector(xs::JLArray{<:Any,1}) = SparseVector(JLSparseVector(xs))
JLSparseVector(xs::JLArray{T,1}) where {T} =
    GPUArrays.to_sparse(JLSparseVector{T,Int}, xs)
Base.length(x::JLSparseVector) = x.len
Base.size(x::JLSparseVector) = (x.len,)

function JLSparseMatrixCSC(xs::SparseMatrixCSC{Tv, Ti}) where {Ti, Tv}
    colPtr = JLVector{Ti}(undef, length(xs.colptr))
    rowVal = JLVector{Ti}(undef, length(xs.rowval))
    nzVal  = JLVector{Tv}(undef, length(xs.nzval))
    copyto!(colPtr, convert(Vector{Ti}, xs.colptr))
    copyto!(rowVal, convert(Vector{Ti}, xs.rowval))
    copyto!(nzVal,  convert(Vector{Tv}, xs.nzval))
    return JLSparseMatrixCSC{Tv, Ti}(colPtr, rowVal, nzVal, (xs.m, xs.n))
end
SparseArrays.SparseMatrixCSC(xs::JLArray{<:Any,2}) = SparseMatrixCSC(JLSparseMatrixCSC(xs))
JLSparseMatrixCSC(xs::SparseVector) = JLSparseMatrixCSC(SparseMatrixCSC(xs))
JLSparseMatrixCSC(xs::JLArray{T,2}) where {T} = JLSparseMatrixCSC(JLSparseMatrixCOO(xs))
Base.length(x::JLSparseMatrixCSC) = prod(x.dims)
Base.size(x::JLSparseMatrixCSC) = x.dims

function JLSparseMatrixCSR(xs::SparseMatrixCSC{Tv, Ti}) where {Ti, Tv}
    csr_xs = SparseMatrixCSC(transpose(xs))
    rowPtr = JLVector{Ti}(undef, length(csr_xs.colptr))
    colVal = JLVector{Ti}(undef, length(csr_xs.rowval))
    nzVal  = JLVector{Tv}(undef, length(csr_xs.nzval))
    copyto!(rowPtr, convert(Vector{Ti}, csr_xs.colptr))
    copyto!(colVal, convert(Vector{Ti}, csr_xs.rowval))
    copyto!(nzVal,  convert(Vector{Tv}, csr_xs.nzval))
    return JLSparseMatrixCSR{Tv, Ti}(rowPtr, colVal, nzVal, (xs.m, xs.n))
end
JLSparseMatrixCSR(xs::SparseVector{Tv, Ti}) where {Ti, Tv} = JLSparseMatrixCSR(SparseMatrixCSC(xs))
JLSparseMatrixCSR(xs::JLArray{T,2}) where {T} = JLSparseMatrixCSR(JLSparseMatrixCOO(xs))
function JLSparseMatrixCOO(xs::SparseMatrixCSC{Tv, Ti}) where {Ti, Tv}
    rowInd, colInd, nzVal = findnz(xs)
    return JLSparseMatrixCOO{Tv, Ti}(JLVector{Ti}(rowInd), JLVector{Ti}(colInd),
                                     JLVector{Tv}(nzVal), (xs.m, xs.n))
end
JLSparseMatrixCOO(xs::SparseVector) = JLSparseMatrixCOO(SparseMatrixCSC(xs))
JLSparseMatrixCOO(xs::JLArray{T,2}) where {T} =
    GPUArrays.to_sparse(JLSparseMatrixCOO{T,Int}, xs)
# Cross-format device→device conversions (CSC↔CSR↔COO): the back-end owns the constructors;
# the on-device conversion algorithm is GPUArrays' generic `convert` (replacing the former
# host round-trips through `SparseMatrixCSC`). The `sparse_csc/csr/coo` verbs above route to
# these. The COO→CSR/CSC path needs `sortperm`, provided below.
JLSparseMatrixCOO(A::Union{JLSparseMatrixCSC,JLSparseMatrixCSR}) =
    convert(JLSparseMatrixCOO, A)
JLSparseMatrixCSR(A::Union{JLSparseMatrixCSC,JLSparseMatrixCOO}) =
    convert(JLSparseMatrixCSR, A)
JLSparseMatrixCSC(A::Union{JLSparseMatrixCSR,JLSparseMatrixCOO}) =
    convert(JLSparseMatrixCSC, A)

# `sortperm` on a JLArray vector, required by the generic COO→CSR/CSC conversions (and by
# `findnz`). Computed on the host and wrapped back into a device vector so that the gather
# kernel can index the permutation on-device (mirrors Metal's `sortperm`, which also returns
# a device `Int` vector).
Base.sortperm(x::JLVector; kwargs...) = JLArray(sortperm(Array(x); kwargs...))

SparseArrays.sparse(xs::JLArray{T,1}) where {T} =
    GPUArrays.to_sparse(JLSparseVector{T,Int}, xs)

function SparseArrays.sparse(xs::JLArray{T,2}; fmt=:csc) where {T}
    if fmt == :csc
        return JLSparseMatrixCSC(xs)
    elseif fmt == :csr
        return JLSparseMatrixCSR(xs)
    elseif fmt == :coo
        return JLSparseMatrixCOO(xs)
    else
        throw(ArgumentError("format :$fmt not available, use :csc, :csr, or :coo"))
    end
end

function Base.copyto!(dst::JLSparseMatrixCSR, src::JLSparseMatrixCSR)
    if size(dst) != size(src)
        throw(ArgumentError("Inconsistent Sparse Matrix size"))
    end
    resize!(dst.rowPtr, length(src.rowPtr))
    resize!(dst.colVal, length(src.colVal))
    resize!(SparseArrays.nonzeros(dst), length(SparseArrays.nonzeros(src)))
    copyto!(dst.rowPtr, src.rowPtr)
    copyto!(dst.colVal, src.colVal)
    copyto!(SparseArrays.nonzeros(dst), SparseArrays.nonzeros(src))
    dst.nnz = src.nnz
    dst
end
Base.length(x::JLSparseMatrixCSR) = prod(x.dims)
Base.size(x::JLSparseMatrixCSR) = x.dims

function GPUArrays._spadjoint(A::JLSparseMatrixCSR)
    Aᴴ = JLSparseMatrixCSC(A.rowPtr, A.colVal, conj(A.nzVal), reverse(size(A)))
    JLSparseMatrixCSR(Aᴴ)
end
function GPUArrays._sptranspose(A::JLSparseMatrixCSR)
    Aᵀ = JLSparseMatrixCSC(A.rowPtr, A.colVal, A.nzVal, reverse(size(A)))
    JLSparseMatrixCSR(Aᵀ)
end
function _spadjoint(A::JLSparseMatrixCSC)
    Aᴴ = JLSparseMatrixCSR(A.colPtr, A.rowVal, conj(A.nzVal), reverse(size(A)))
    JLSparseMatrixCSC(Aᴴ)
end
function _sptranspose(A::JLSparseMatrixCSC)
    Aᵀ = JLSparseMatrixCSR(A.colPtr, A.rowVal, A.nzVal, reverse(size(A)))
    JLSparseMatrixCSC(Aᵀ)
end

# idempotency
JLArray{T,N}(xs::JLArray{T,N}) where {T,N} = xs

# adapt for the GPU
jl(xs) = adapt(JLArray, xs)
## don't convert isbits types since they are already considered GPU-compatible
Adapt.adapt_storage(::Type{JLArray}, xs::AbstractArray) =
  isbits(xs) ? xs : convert(JLArray, xs)
## if an element type is specified, convert to it
Adapt.adapt_storage(::Type{<:JLArray{T}}, xs::AbstractArray) where {T} =
  isbits(xs) ? xs : convert(JLArray{T}, xs)

# adapt back to the CPU
Adapt.adapt_storage(::Type{Array}, xs::JLArray) = convert(Array, xs)

## conversions

Base.convert(::Type{T}, x::T) where T <: JLArray = x


## indexing

Base.findall(bools::JLArray{Bool}) = JLArray(findall(Array(bools)))
Base.findall(f::Function, A::JLArray) = JLArray(findall(f, Array(A)))
Base.findall(f::Base.Fix2{typeof(in)}, A::JLArray) = JLArray(findall(f, Array(A)))


## broadcast

import Base.Broadcast: BroadcastStyle, Broadcasted

struct JLArrayStyle{N} <: AbstractGPUArrayStyle{N} end
JLArrayStyle{M}(::Val{N}) where {N,M} = JLArrayStyle{N}()

# identify the broadcast style of a (wrapped) array
BroadcastStyle(::Type{<:JLArray{T,N}}) where {T,N} = JLArrayStyle{N}()
BroadcastStyle(::Type{<:AnyJLArray{T,N}}) where {T,N} = JLArrayStyle{N}()

# allocation of output arrays
Base.similar(bc::Broadcasted{JLArrayStyle{N}}, ::Type{T}, dims) where {T,N} =
    similar(JLArray{T}, dims)


## memory operations

function Base.copyto!(dest::Array{T}, d_offset::Integer,
                      source::DenseJLArray{T}, s_offset::Integer,
                      amount::Integer) where T
    amount==0 && return dest
    @boundscheck checkbounds(dest, d_offset)
    @boundscheck checkbounds(dest, d_offset+amount-1)
    @boundscheck checkbounds(source, s_offset)
    @boundscheck checkbounds(source, s_offset+amount-1)
    GC.@preserve dest source Base.unsafe_copyto!(pointer(dest, d_offset),
                                                 pointer(source, s_offset), amount)
    return dest
end

Base.copyto!(dest::Array{T}, source::DenseJLArray{T}) where {T} =
    copyto!(dest, 1, source, 1, length(source))

function Base.copyto!(dest::DenseJLArray{T}, d_offset::Integer,
                      source::Array{T}, s_offset::Integer,
                      amount::Integer) where T
    amount==0 && return dest
    @boundscheck checkbounds(dest, d_offset)
    @boundscheck checkbounds(dest, d_offset+amount-1)
    @boundscheck checkbounds(source, s_offset)
    @boundscheck checkbounds(source, s_offset+amount-1)
    GC.@preserve dest source Base.unsafe_copyto!(pointer(dest, d_offset),
                                                 pointer(source, s_offset), amount)
    return dest
end

Base.copyto!(dest::DenseJLArray{T}, source::Array{T}) where {T} =
    copyto!(dest, 1, source, 1, length(source))

function Base.copyto!(dest::DenseJLArray{T}, d_offset::Integer,
                      source::DenseJLArray{T}, s_offset::Integer,
                      amount::Integer) where T
    amount==0 && return dest
    @boundscheck checkbounds(dest, d_offset)
    @boundscheck checkbounds(dest, d_offset+amount-1)
    @boundscheck checkbounds(source, s_offset)
    @boundscheck checkbounds(source, s_offset+amount-1)
    GC.@preserve dest source Base.unsafe_copyto!(pointer(dest, d_offset),
                                                 pointer(source, s_offset), amount)
    return dest
end

Base.copyto!(dest::DenseJLArray{T}, source::DenseJLArray{T}) where {T} =
    copyto!(dest, 1, source, 1, length(source))

function Base.resize!(a::DenseJLVector{T}, nl::Integer) where {T}
    # JLArrays aren't performance critical, so simply allocate a new one
    # instead of duplicating the underlying data allocation from the ctor.
    b = JLVector{T}(undef, nl)
    copyto!(b, 1, a, 1, min(length(a), nl))

    # replace the data, freeing the old one and increasing the refcount of the new one
    # to avoid it from being freed when we leave this function.
    unsafe_free!(a)
    a.data = copy(b.data)

    a.offset = b.offset
    a.dims = b.dims
    return a
end

## random number generation

using Random


## GPUArrays interfaces

@static if !isdefined(JLArrays.KernelAbstractions, :POCL) # KA v0.9
    Adapt.adapt_storage(::Adaptor, x::JLArray{T,N}) where {T,N} =
      JLDeviceArray{T,N}(x.data[], x.offset, x.dims)
else
    function Adapt.adapt_storage(::Adaptor, x::JLArray{T,N}) where {T,N}
        arr = typed_data(x)
        Adapt.adapt_storage(POCL.KernelAdaptor([pointer(arr)]), arr)
    end
end

function GPUArrays.mapreducedim!(f, op, R::AnyJLArray, A::Union{AbstractArray,Broadcast.Broadcasted};
                                 init=nothing)
    if init !== nothing
        fill!(R, init)
    end
    @allowscalar Base.reducedim!(op, typed_data(R), map(f, A))
    R
end

Adapt.adapt_structure(to::Adaptor, x::JLSparseMatrixCSC{Tv,Ti}) where {Tv,Ti} =
GPUSparseDeviceMatrixCSC{Tv,Ti,JLDeviceArray{Ti, 1}, JLDeviceArray{Tv, 1}, AS.Generic}(adapt(to, x.colPtr), adapt(to, x.rowVal), adapt(to, x.nzVal), x.dims, x.nnz)
Adapt.adapt_structure(to::Adaptor, x::JLSparseMatrixCSR{Tv,Ti}) where {Tv,Ti} =
GPUSparseDeviceMatrixCSR{Tv,Ti,JLDeviceArray{Ti, 1}, JLDeviceArray{Tv, 1}, AS.Generic}(adapt(to, x.rowPtr), adapt(to, x.colVal), adapt(to, x.nzVal), x.dims, x.nnz)
Adapt.adapt_structure(to::Adaptor, x::JLSparseMatrixCOO{Tv,Ti}) where {Tv,Ti} =
GPUSparseDeviceMatrixCOO{Tv,Ti,JLDeviceArray{Ti, 1}, JLDeviceArray{Tv, 1}, AS.Generic}(adapt(to, x.rowInd), adapt(to, x.colInd), adapt(to, x.nzVal), x.dims, x.nnz)
Adapt.adapt_structure(to::Adaptor, x::JLSparseVector{Tv,Ti}) where {Tv,Ti} =
GPUSparseDeviceVector{Tv,Ti,JLDeviceArray{Ti, 1}, JLDeviceArray{Tv, 1}, AS.Generic}(adapt(to, x.iPtr), adapt(to, x.nzVal), x.len, x.nnz)

## KernelAbstractions interface

KernelAbstractions.get_backend(a::JLA) where JLA <: JLArray = JLBackend()
KernelAbstractions.get_backend(a::JLA) where JLA <: Union{JLSparseMatrixCSC, JLSparseMatrixCSR, JLSparseMatrixCOO, JLSparseVector} = JLBackend()

function KernelAbstractions.mkcontext(kernel::Kernel{JLBackend}, I, _ndrange, iterspace, ::Dynamic) where Dynamic
    return KernelAbstractions.CompilerMetadata{KernelAbstractions.ndrange(kernel), Dynamic}(I, _ndrange, iterspace)
end

KernelAbstractions.allocate(::JLBackend, ::Type{T}, dims::Tuple) where T = JLArray{T}(undef, dims)

@inline function launch_config(kernel::Kernel{JLBackend}, ndrange, workgroupsize)
    if ndrange isa Integer
        ndrange = (ndrange,)
    end
    if workgroupsize isa Integer
        workgroupsize = (workgroupsize, )
    end

    if KernelAbstractions.workgroupsize(kernel) <: DynamicSize && workgroupsize === nothing
        workgroupsize = (MAXTHREADS,) # Vectorization, 4x unrolling, minimal grain size
    end
    iterspace, dynamic = partition(kernel, ndrange, workgroupsize)
    # partition checked that the ndrange's agreed
    if KernelAbstractions.ndrange(kernel) <: StaticSize
        ndrange = nothing
    end

    return ndrange, workgroupsize, iterspace, dynamic
end

@static if isdefined(JLArrays.KernelAbstractions, :isgpu) # KA v0.9
    KernelAbstractions.isgpu(b::JLBackend) = false
end

@static if !isdefined(JLArrays.KernelAbstractions, :POCL) # KA v0.9
    function convert_to_cpu(obj::Kernel{JLBackend, W, N, F}) where {W, N, F}
        return Kernel{typeof(KernelAbstractions.CPU(; static = obj.backend.static)), W, N, F}(KernelAbstractions.CPU(; static = obj.backend.static), obj.f)
    end
else
    function convert_to_cpu(obj::Kernel{JLBackend, W, N, F}) where {W, N, F}
        return Kernel{typeof(KernelAbstractions.POCLBackend()), W, N, F}(KernelAbstractions.POCLBackend(), obj.f)
    end
end

function (obj::Kernel{JLBackend})(args...; ndrange=nothing, workgroupsize=nothing)
    ndrange, workgroupsize, _, _ = launch_config(obj, ndrange, workgroupsize)
    # Use `map` rather than `jlconvert.(args)` to skip the broadcast
    # machinery (broadcasted/materialize/ntuple) that would otherwise
    # be specialized per unique arg-tuple type.
    device_args = map(jlconvert, args)
    new_obj = convert_to_cpu(obj)
    new_obj(device_args...; ndrange, workgroupsize)
end

Adapt.adapt_storage(::JLBackend, a::Array) = Adapt.adapt(JLArrays.JLArray, a)
Adapt.adapt_storage(::JLBackend, a::JLArrays.JLArray) = a

@static if !isdefined(JLArrays.KernelAbstractions, :POCL) # KA v0.9
    Adapt.adapt_storage(::KernelAbstractions.CPU, a::JLArrays.JLArray) = convert(Array, a)
else
    Adapt.adapt_storage(::KernelAbstractions.POCLBackend, a::JLArrays.JLArray) = convert(Array, a)
end

end
