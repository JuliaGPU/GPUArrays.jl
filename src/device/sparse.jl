# on-device sparse array types
# should be excluded from coverage counts
# COV_EXCL_START
using SparseArrays

# NOTE: this functionality is currently very bare-bones, only defining the array types
#       without any device-compatible sparse array functionality


# core types

export GPUSparseDeviceVector, GPUSparseDeviceMatrixCSC, GPUSparseDeviceMatrixCSR,
       GPUSparseDeviceMatrixBSR, GPUSparseDeviceMatrixCOO

abstract type AbstractGPUSparseDeviceMatrix{Tv, Ti} <: AbstractSparseMatrix{Tv, Ti} end


struct GPUSparseDeviceVector{Tv,Ti,Vi,Vv, A} <: AbstractSparseVector{Tv,Ti}
    iPtr::Vi
    nzVal::Vv
    len::Int
    nnz::Ti
end

Base.length(g::GPUSparseDeviceVector) = g.len
Base.size(g::GPUSparseDeviceVector) = (g.len,)
SparseArrays.nnz(g::GPUSparseDeviceVector) = g.nnz
SparseArrays.nonzeroinds(g::GPUSparseDeviceVector) = g.iPtr
SparseArrays.nonzeros(g::GPUSparseDeviceVector) = g.nzVal

struct GPUSparseDeviceMatrixCSC{Tv,Ti,Vi,Vv,A} <: AbstractGPUSparseDeviceMatrix{Tv, Ti}
    colPtr::Vi
    rowVal::Vi
    nzVal::Vv
    dims::NTuple{2,Int}
    nnz::Ti
end

Base.length(g::GPUSparseDeviceMatrixCSC) = prod(g.dims)
Base.size(g::GPUSparseDeviceMatrixCSC) = g.dims
SparseArrays.nnz(g::GPUSparseDeviceMatrixCSC) = g.nnz
SparseArrays.rowvals(g::GPUSparseDeviceMatrixCSC) = g.rowVal
SparseArrays.getcolptr(g::GPUSparseDeviceMatrixCSC) = g.colPtr
SparseArrays.nzrange(g::GPUSparseDeviceMatrixCSC, col::Integer) = @inbounds SparseArrays.getcolptr(g)[col]:(SparseArrays.getcolptr(g)[col+1]-1)
SparseArrays.nonzeros(g::GPUSparseDeviceMatrixCSC) = g.nzVal

const GPUSparseDeviceColumnView{Tv, Ti, Vi, Vv, A} = SubArray{Tv, 1, GPUSparseDeviceMatrixCSC{Tv, Ti, Vi, Vv, A}, Tuple{Base.Slice{Base.OneTo{Int}}, Int}}
function SparseArrays.nonzeros(x::GPUSparseDeviceColumnView)
    rowidx, colidx = parentindices(x)
    A = parent(x)
    @inbounds y = view(SparseArrays.nonzeros(A), SparseArrays.nzrange(A, colidx))
    return y
end

function SparseArrays.nonzeroinds(x::GPUSparseDeviceColumnView)
    rowidx, colidx = parentindices(x)
    A = parent(x)
    @inbounds y = view(SparseArrays.rowvals(A), SparseArrays.nzrange(A, colidx))
    return y
end
SparseArrays.rowvals(x::GPUSparseDeviceColumnView) = SparseArrays.nonzeroinds(x)

function SparseArrays.nnz(x::GPUSparseDeviceColumnView)
    rowidx, colidx = parentindices(x)
    A = parent(x)
    return length(SparseArrays.nzrange(A, colidx))
end

struct GPUSparseDeviceMatrixCSR{Tv,Ti,Vi,Vv,A} <: AbstractGPUSparseDeviceMatrix{Tv,Ti}
    rowPtr::Vi
    colVal::Vi
    nzVal::Vv
    dims::NTuple{2, Int}
    nnz::Ti
end

@inline function _getindex(arg::Union{GPUSparseDeviceMatrixCSR{Tv},
                                      GPUSparseDeviceMatrixCSC{Tv},
                                      GPUSparseDeviceVector{Tv}}, I, ptr)::Tv where {Tv}
    if ptr == 0
        return zero(Tv)
    else
        return @inbounds arg.nzVal[ptr]::Tv
    end
end

struct GPUSparseDeviceMatrixBSR{Tv,Ti,Vi,Vv,A} <: AbstractGPUSparseDeviceMatrix{Tv,Ti}
    rowPtr::Vi
    colVal::Vi
    nzVal::Vv
    dims::NTuple{2,Int}
    blockDim::Ti
    dir::Char
    nnz::Ti
end

struct GPUSparseDeviceMatrixCOO{Tv,Ti,Vi,Vv, A} <: AbstractGPUSparseDeviceMatrix{Tv,Ti}
    rowInd::Vi
    colInd::Vi
    nzVal::Vv
    dims::NTuple{2,Int}
    nnz::Ti
end

Base.length(g::AbstractGPUSparseDeviceMatrix) = prod(g.dims)
Base.size(g::AbstractGPUSparseDeviceMatrix) = g.dims
SparseArrays.nnz(g::AbstractGPUSparseDeviceMatrix) = g.nnz
SparseArrays.getnzval(g::AbstractGPUSparseDeviceMatrix) = g.nzVal

struct GPUSparseDeviceArrayCSR{Tv, Ti, Vi, Vv, N, M, A} <: AbstractSparseArray{Tv, Ti, N}
    rowPtr::Vi
    colVal::Vi
    nzVal::Vv
    dims::NTuple{N, Int}
    nnz::Ti
end

function GPUSparseDeviceArrayCSR{Tv, Ti, Vi, Vv, N}(rowPtr::Vi, colVal::Vi, nzVal::Vv, dims::NTuple{N,<:Integer}) where {Tv, Ti<:Integer, M, Vi<:AbstractDeviceArray{<:Integer,M}, Vv<:AbstractDeviceArray{Tv, M}, N}
    @assert M == N - 1 "GPUSparseDeviceArrayCSR requires ndims(rowPtr) == ndims(colVal) == ndims(nzVal) == length(dims) - 1"
    GPUSparseDeviceArrayCSR{Tv, Ti, Vi, Vv, N, M}(rowPtr, colVal, nzVal, dims, length(nzVal))
end

Base.length(g::GPUSparseDeviceArrayCSR) = prod(g.dims)
Base.size(g::GPUSparseDeviceArrayCSR) = g.dims
SparseArrays.nnz(g::GPUSparseDeviceArrayCSR) = g.nnz
SparseArrays.getnzval(g::GPUSparseDeviceArrayCSR) = g.nzVal

# input/output

function Base.show(io::IO, ::MIME"text/plain", A::GPUSparseDeviceVector)
    println(io, "$(length(A))-element device sparse vector at:")
    println(io, "  iPtr: $(A.iPtr)")
    print(io,   "  nzVal: $(A.nzVal)")
end

function Base.show(io::IO, ::MIME"text/plain", A::GPUSparseDeviceMatrixCSR)
    println(io, "$(length(A))-element device sparse matrix CSR at:")
    println(io, "  rowPtr: $(A.rowPtr)")
    println(io, "  colVal: $(A.colVal)")
    print(io,   "  nzVal:  $(A.nzVal)")
end

function Base.show(io::IO, ::MIME"text/plain", A::GPUSparseDeviceMatrixCSC)
    println(io, "$(length(A))-element device sparse matrix CSC at:")
    println(io, "  colPtr: $(A.colPtr)")
    println(io, "  rowVal: $(A.rowVal)")
    print(io,   "  nzVal:  $(A.nzVal)")
end

function Base.show(io::IO, ::MIME"text/plain", A::GPUSparseDeviceMatrixBSR)
    println(io, "$(length(A))-element device sparse matrix BSR at:")
    println(io, "  rowPtr: $(A.rowPtr)")
    println(io, "  colVal: $(A.colVal)")
    print(io,   "  nzVal:  $(A.nzVal)")
end

function Base.show(io::IO, ::MIME"text/plain", A::GPUSparseDeviceMatrixCOO)
    println(io, "$(length(A))-element device sparse matrix COO at:")
    println(io, "  rowPtr: $(A.rowPtr)")
    println(io, "  colInd: $(A.colInd)")
    print(io,   "  nzVal:  $(A.nzVal)")
end

function Base.show(io::IO, ::MIME"text/plain", A::GPUSparseDeviceArrayCSR)
    println(io, "$(length(A))-element device sparse array CSR at:")
    println(io, "  rowPtr: $(A.rowPtr)")
    println(io, "  colVal: $(A.colVal)")
    print(io,   "  nzVal:  $(A.nzVal)")
end

# COV_EXCL_STOP
