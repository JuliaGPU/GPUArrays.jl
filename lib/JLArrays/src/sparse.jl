export JLSparseMatrixCSC

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

Base.copy(A::JLSparseMatrixCSC) = JLSparseMatrixCSC(A.m, A.n, copy(A.colptr), copy(A.rowval), copy(A.nzval))
