# sparse array types

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

GPUArrays.sparse_array_type(sa::JLSparseMatrixCSC) = JLSparseMatrixCSC
GPUArrays.sparse_array_type(::Type{<:JLSparseMatrixCSC}) = JLSparseMatrixCSC
GPUArrays.sparse_array_type(sa::JLSparseMatrixCSR) = JLSparseMatrixCSR
GPUArrays.sparse_array_type(::Type{<:JLSparseMatrixCSR}) = JLSparseMatrixCSR
GPUArrays.sparse_array_type(sa::JLSparseVector) = JLSparseVector
GPUArrays.sparse_array_type(::Type{<:JLSparseVector}) = JLSparseVector

GPUArrays.dense_array_type(sa::JLSparseVector) = JLArray
GPUArrays.dense_array_type(::Type{<:JLSparseVector}) = JLArray
GPUArrays.dense_array_type(sa::JLSparseMatrixCSC) = JLArray
GPUArrays.dense_array_type(::Type{<:JLSparseMatrixCSC}) = JLArray
GPUArrays.dense_array_type(sa::JLSparseMatrixCSR) = JLArray
GPUArrays.dense_array_type(::Type{<:JLSparseMatrixCSR}) = JLArray

GPUArrays.csc_type(::Type{<:JLSparseMatrixCSR}) = JLSparseMatrixCSC
GPUArrays.csr_type(::Type{<:JLSparseMatrixCSC}) = JLSparseMatrixCSR

Base.similar(Mat::JLSparseMatrixCSR) = JLSparseMatrixCSR(copy(Mat.rowPtr), copy(Mat.colVal), similar(nonzeros(Mat)), size(Mat))
Base.similar(Mat::JLSparseMatrixCSR, T::Type) = JLSparseMatrixCSR(copy(Mat.rowPtr), copy(Mat.colVal), similar(nonzeros(Mat), T), size(Mat))

Base.similar(Mat::JLSparseMatrixCSC, T::Type, N::Int, M::Int) =  JLSparseMatrixCSC(JLVector([zero(Int32)]), JLVector{Int32}(undef, 0), JLVector{T}(undef, 0), (N, M))
Base.similar(Mat::JLSparseMatrixCSR, T::Type, N::Int, M::Int) =  JLSparseMatrixCSR(JLVector([zero(Int32)]), JLVector{Int32}(undef, 0), JLVector{T}(undef, 0), (N, M))

Base.similar(Mat::JLSparseMatrixCSC{Tv, Ti}, N::Int, M::Int) where {Tv, Ti} = similar(Mat, Tv, N, M)
Base.similar(Mat::JLSparseMatrixCSR{Tv, Ti}, N::Int, M::Int) where {Tv, Ti} = similar(Mat, Tv, N, M)

Base.similar(Mat::JLSparseMatrixCSC, T::Type, dims::Tuple{Int, Int}) = similar(Mat, T, dims...)
Base.similar(Mat::JLSparseMatrixCSR, T::Type, dims::Tuple{Int, Int}) = similar(Mat, T, dims...)

Base.similar(Mat::JLSparseMatrixCSC, dims::Tuple{Int, Int}) = similar(Mat, dims...)
Base.similar(Mat::JLSparseMatrixCSR, dims::Tuple{Int, Int}) = similar(Mat, dims...)

JLArray(x::JLSparseVector)    = JLArray(collect(SparseVector(x)))
JLArray(x::JLSparseMatrixCSC) = JLArray(collect(SparseMatrixCSC(x)))
JLArray(x::JLSparseMatrixCSR) = JLArray(collect(SparseMatrixCSC(x)))


## interop with SparseArrays

function JLSparseVector(xs::SparseVector{Tv, Ti}) where {Ti, Tv}
    iPtr  = JLVector{Ti}(undef, length(xs.nzind))
    nzVal = JLVector{Tv}(undef, length(xs.nzval))
    copyto!(iPtr, convert(Vector{Ti}, xs.nzind))
    copyto!(nzVal, convert(Vector{Tv}, xs.nzval))
    return JLSparseVector{Tv, Ti}(iPtr, nzVal, length(xs),)
end
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
JLSparseMatrixCSC(xs::SparseVector) = JLSparseMatrixCSC(SparseMatrixCSC(xs))
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
function JLSparseMatrixCSR(xs::JLSparseMatrixCSC{Tv, Ti}) where {Ti, Tv}
    return JLSparseMatrixCSR(SparseMatrixCSC(xs))
end
function JLSparseMatrixCSC(xs::JLSparseMatrixCSR{Tv, Ti}) where {Ti, Tv}
    return JLSparseMatrixCSC(SparseMatrixCSC(xs))
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


## adapt for the device

Adapt.adapt_structure(to::Adaptor, x::JLSparseMatrixCSC{Tv,Ti}) where {Tv,Ti} =
GPUSparseDeviceMatrixCSC{Tv,Ti,JLDeviceArray{Ti, 1}, JLDeviceArray{Tv, 1}, AS.Generic}(adapt(to, x.colPtr), adapt(to, x.rowVal), adapt(to, x.nzVal), x.dims, x.nnz)
Adapt.adapt_structure(to::Adaptor, x::JLSparseMatrixCSR{Tv,Ti}) where {Tv,Ti} =
GPUSparseDeviceMatrixCSR{Tv,Ti,JLDeviceArray{Ti, 1}, JLDeviceArray{Tv, 1}, AS.Generic}(adapt(to, x.rowPtr), adapt(to, x.colVal), adapt(to, x.nzVal), x.dims, x.nnz)
Adapt.adapt_structure(to::Adaptor, x::JLSparseVector{Tv,Ti}) where {Tv,Ti} =
GPUSparseDeviceVector{Tv,Ti,JLDeviceArray{Ti, 1}, JLDeviceArray{Tv, 1}, AS.Generic}(adapt(to, x.iPtr), adapt(to, x.nzVal), x.len, x.nnz)
