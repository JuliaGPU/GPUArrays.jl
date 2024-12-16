export JLSparseVector, JLSparseMatrixCSC

## Sparse Vector

struct JLSparseVector{Tv,Ti<:Integer} <: AbstractGPUSparseVector{Tv,Ti}
    n::Ti              # Length of the sparse vector
    nzind::JLVector{Ti}   # Indices of stored values
    nzval::JLVector{Tv}   # Stored values, typically nonzeros

    function JLSparseVector{Tv,Ti}(n::Integer, nzind::JLVector{Ti}, nzval::JLVector{Tv}) where {Tv,Ti<:Integer}
        n >= 0 || throw(ArgumentError("The number of elements must be non-negative."))
        length(nzind) == length(nzval) ||
            throw(ArgumentError("index and value vectors must be the same length"))
        new(convert(Ti, n), nzind, nzval)
    end
end

JLSparseVector(n::Integer, nzind::JLVector{Ti}, nzval::JLVector{Tv}) where {Tv,Ti} =
    JLSparseVector{Tv,Ti}(n, nzind, nzval)

JLSparseVector(V::SparseVector) = JLSparseVector(V.n, JLVector(V.nzind), JLVector(V.nzval))
SparseVector(V::JLSparseVector) = SparseVector(V.n, Vector(V.nzind), Vector(V.nzval))

Base.copy(V::JLSparseVector) = JLSparseVector(V.n, copy(V.nzind), copy(V.nzval))

Base.length(V::JLSparseVector) = V.n
Base.size(V::JLSparseVector) = (V.n,)

SparseArrays.nonzeros(V::JLSparseVector) = V.nzval
SparseArrays.nonzeroinds(V::JLSparseVector) = V.nzind

## SparseMatrixCSC

struct JLSparseMatrixCSC{Tv,Ti<:Integer} <: AbstractGPUSparseMatrixCSC{Tv,Ti}
    m::Int                  # Number of rows
    n::Int                  # Number of columns
    colptr::JLVector{Ti}      # Column i is in colptr[i]:(colptr[i+1]-1)
    rowval::JLVector{Ti}      # Row indices of stored values
    nzval::JLVector{Tv}       # Stored values, typically nonzeros

    function JLSparseMatrixCSC{Tv,Ti}(m::Integer, n::Integer, colptr::JLVector{Ti},
                            rowval::JLVector{Ti}, nzval::JLVector{Tv}) where {Tv,Ti<:Integer}
        SparseArrays.sparse_check_Ti(m, n, Ti)
        GPUArrays._goodbuffers_csc(m, n, colptr, rowval, nzval) ||
            throw(ArgumentError("Invalid buffers for JLSparseMatrixCSC construction n=$n, colptr=$(summary(colptr)), rowval=$(summary(rowval)), nzval=$(summary(nzval))"))
        new(Int(m), Int(n), colptr, rowval, nzval)
    end
end
function JLSparseMatrixCSC(m::Integer, n::Integer, colptr::JLVector, rowval::JLVector, nzval::JLVector)
    Tv = eltype(nzval)
    Ti = promote_type(eltype(colptr), eltype(rowval))
    SparseArrays.sparse_check_Ti(m, n, Ti)
    # SparseArrays.sparse_check(n, colptr, rowval, nzval) # TODO: this uses scalar indexing
    # silently shorten rowval and nzval to usable index positions.
    maxlen = abs(widemul(m, n))
    isbitstype(Ti) && (maxlen = min(maxlen, typemax(Ti) - 1))
    length(rowval) > maxlen && resize!(rowval, maxlen)
    length(nzval) > maxlen && resize!(nzval, maxlen)
    JLSparseMatrixCSC{Tv,Ti}(m, n, colptr, rowval, nzval)
end

JLSparseMatrixCSC(A::SparseMatrixCSC) = JLSparseMatrixCSC(A.m, A.n, JLVector(A.colptr), JLVector(A.rowval), JLVector(A.nzval))
SparseMatrixCSC(A::JLSparseMatrixCSC) = SparseMatrixCSC(A.m, A.n, Vector(A.colptr), Vector(A.rowval), Vector(A.nzval))

Base.copy(A::JLSparseMatrixCSC) = JLSparseMatrixCSC(A.m, A.n, copy(A.colptr), copy(A.rowval), copy(A.nzval))

Base.size(A::JLSparseMatrixCSC) = (A.m, A.n)
Base.length(A::JLSparseMatrixCSC) = A.m * A.n

SparseArrays.nonzeros(A::JLSparseMatrixCSC) = A.nzval
SparseArrays.getcolptr(A::JLSparseMatrixCSC) = A.colptr
SparseArrays.rowvals(A::JLSparseMatrixCSC) = A.rowval
