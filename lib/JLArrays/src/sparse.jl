# sparse arrays with JLArray storage: GPUArrays implements the formats, JLArrays only names
# them and adds constructors

const JLSparseVector{Tv,Ti} = GPUSparseVector{Tv,Ti,<:JLVector{Ti},<:JLVector{Tv}}
const JLSparseMatrixCSR{Tv,Ti} = GPUSparseMatrixCSR{Tv,Ti,<:JLVector{Ti},<:JLVector{Tv}}
const JLSparseMatrixCSC{Tv,Ti} = GPUSparseMatrixCSC{Tv,Ti,<:JLVector{Ti},<:JLVector{Tv}}
const JLSparseMatrixCOO{Tv,Ti} = GPUSparseMatrixCOO{Tv,Ti,<:JLVector{Ti},<:JLVector{Tv}}
const JLSparseMatrix = Union{JLSparseMatrixCSR,JLSparseMatrixCSC,JLSparseMatrixCOO}

# move the input to JLArray storage (keeping its format), then convert on the device
for (alias, S) in ((:JLSparseVector, :GPUSparseVector), (:JLSparseMatrixCSR, :GPUSparseMatrixCSR),
                   (:JLSparseMatrixCSC, :GPUSparseMatrixCSC), (:JLSparseMatrixCOO, :GPUSparseMatrixCOO))
    @eval begin
        $alias(A::AbstractArray) = $S(adapt(JLArray, A))
        $alias{Tv}(A::AbstractArray) where {Tv} = $S{Tv}(adapt(JLArray, A))
        $alias{Tv,Ti}(A::AbstractArray) where {Tv,Ti} = $S{Tv,Ti}(adapt(JLArray, A))
    end
end

# densify on the device
JLArray{T,1}(x::JLSparseVector) where {T} = copyto!(similar(nonzeros(x), T, size(x)), x)
JLArray{T,2}(A::JLSparseMatrix) where {T} = copyto!(similar(nonzeros(A), T, size(A)), A)
