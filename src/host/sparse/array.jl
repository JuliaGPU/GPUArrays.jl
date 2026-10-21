# Array semantics that SparseArrays implements through host scalar loops or downloads.

function Base.:(==)(A::GPUSparseArray, B::GPUSparseArray)
    size(A) == size(B) || return false
    # Stored zeros need not have identical patterns.
    return !any(comparison_operand(A) .!= comparison_operand(B))
end

comparison_operand(A::Union{GPUSparseVector,GPUSparseMatrixCOO}) = A
function comparison_operand(A::GPUSparseMatrixCompressed)
    # A scalar comparison does not create a public sparse output. Its structural union
    # can exceed either input's pointer width, including when both store only zeros.
    ptr = compressed_ptr(A)
    eltype(ptr) === Int && return A
    sparse_format_type(A)(convert_index_buffer(Int, ptr), compressed_ind(A), A.nzVal, size(A))
end

function sparse_broadcast!(dest::GPUSparseArray, bc::Broadcasted)
    check_sparse_mutation(dest)
    bc = Broadcast.instantiate(Broadcast.broadcasted(bc.f, bc.args...))
    if isempty(axes(bc))
        value = bc[]
        iszero(value) || throw(ArgumentError("nonzero scalar fill of a GPU sparse destination is unsupported"))
        fill!(nonzeros(dest), value)
        return dest
    end
    axes(dest) == axes(bc) || throw(DimensionMismatch("sparse broadcast destination axes must match the result"))
    # Materializing before touching dest makes structural growth and overlapping arguments
    # safe. The temporary owns its buffers and can be copied after complete preflight.
    tmp = copy(bc)
    tmp isa GPUSparseArray ||
        throw(ArgumentError("dense or non-zero-preserving broadcast into a GPU sparse destination is unsupported"))
    if dest isa GPUSparseMatrix && tmp isa GPUSparseMatrix
        return copyto!(dest, convert(matrix_format(dest), tmp))
    elseif dest isa GPUSparseVector && tmp isa GPUSparseVector
        return copyto!(dest, tmp)
    end
    throw(ArgumentError("broadcast cannot change the sparse destination's dimensionality"))
end
Broadcast.materialize!(dest::GPUSparseArray, bc::Broadcasted) = sparse_broadcast!(dest, bc)
Base.copyto!(dest::GPUSparseArray, bc::Broadcasted{<:SPVM}) = sparse_broadcast!(dest, bc)
Base.copyto!(dest::GPUSparseArray, bc::Broadcasted{Nothing}) = sparse_broadcast!(dest, bc)
Base.copyto!(dest::GPUSparseArray, bc::Broadcasted{<:AbstractGPUArrayStyle}) = sparse_broadcast!(dest, bc)
Base.copyto!(dest::GPUSparseArray, bc::Broadcasted{Broadcast.DefaultArrayStyle{0}}) = sparse_broadcast!(dest, bc)

function Base.fill!(A::GPUSparseArray, value)
    iszero(value) || throw(ArgumentError("nonzero fill of a GPU sparse destination is unsupported"))
    fill!(nonzeros(A), value)
    return A
end

Base.getindex(A::GPUSparseMatrix, ::Colon) = vec(A)
Base.getindex(A::GPUSparseMatrix,I::AbstractRange{<:Integer}) = vec(A)[I]

const GPUSparseTransposed = Union{Transpose{<:Any,<:GPUSparseMatrix},Adjoint{<:Any,<:GPUSparseMatrix}}
LinearAlgebra.Hermitian{T}(A::GPUSparseMatrix,uplo::Symbol=:U) where {T} =
    Hermitian(with_values(A,T),uplo)
LinearAlgebra.Symmetric{T}(A::GPUSparseMatrix,uplo::Symbol=:U) where {T} =
    Symmetric(with_values(A,T),uplo)
for op in (:+,:-)
    @eval begin
        Base.$op(A::GPUSparseMatrix,B::GPUSparseTransposed) = sparse_map($op,A,copy(B))
        Base.$op(A::GPUSparseTransposed,B::GPUSparseMatrix) = sparse_map($op,copy(A),B)
        Base.$op(A::GPUSparseTransposed,B::GPUSparseTransposed) = sparse_map($op,copy(A),copy(B))
    end
end
SparseArrays.nnz(A::GPUSparseTransposed) = nnz(parent(A))
SparseArrays.nonzeros(A::Transpose{<:Any,<:GPUSparseMatrix}) = nonzeros(parent(A))
SparseArrays.nonzeros(A::Adjoint{<:Any,<:GPUSparseMatrix}) = conj.(nonzeros(parent(A)))
SparseArrays.SparseMatrixCSC(A::GPUSparseTransposed) = SparseMatrixCSC(copy(A))
Base.Array(A::GPUSparseTransposed) = Array(copy(A))
Base.collect(A::GPUSparseTransposed) = Array(A)
Base.mapreduce(f, op, A::GPUSparseTransposed; kwargs...) = mapreduce(f, op, copy(A); kwargs...)
Base.getindex(A::GPUSparseTransposed, I::SparseRangeIndex, J::SparseRangeIndex) = copy(A)[I, J]

function sparse_diagonal_like(A::GPUSparseMatrix,values::AbstractVector,dims::Dims{2}=size(A))
    n=min(dims...)
    length(values)==n || throw(DimensionMismatch("diagonal length must match matrix dimensions"))
    check_sparse_dims(indtype(A),dims)
    inds=sparse_similar(nonzeros(A),indtype(A),n)
    inds .= 1:n
    Tp=A isa GPUSparseMatrixCompressed ? eltype(compressed_ptr(A)) : promote_type(Int,indtype(A))
    assemble_matrix(inds,copy(inds),move_buffer(nonzeros(A),values),dims...,
                    default_combine(eltype(values)),matrix_format_symbol(A);pointer_type=Tp)
end
matrix_format_symbol(::GPUSparseMatrixCSR)=:csr
matrix_format_symbol(::GPUSparseMatrixCSC)=:csc
matrix_format_symbol(::GPUSparseMatrixCOO)=:coo
function sparse_identity_like(A::GPUSparseMatrix,value)
    n=LinearAlgebra.checksquare(A)
    sparse_diagonal_like(A,fill!(sparse_similar(nonzeros(A),typeof(value),n),value))
end
Base.one(A::GPUSparseMatrix)=sparse_identity_like(A,one(eltype(A)))
for op in (:+,:-)
    @eval begin
        function Base.$op(A::Union{GPUSparseMatrix,GPUSparseTransposed},J::UniformScaling)
            B=materialize(A)
            sparse_map($op,B,sparse_identity_like(B,J.λ))
        end
        function Base.$op(J::UniformScaling,A::Union{GPUSparseMatrix,GPUSparseTransposed})
            B=materialize(A)
            sparse_map($op,sparse_identity_like(B,J.λ),B)
        end
        function Base.$op(A::Union{GPUSparseMatrix,GPUSparseTransposed},D::Diagonal)
            B=materialize(A)
            size(B)==size(D) || throw(DimensionMismatch("matrix dimensions must match"))
            sparse_map($op,B,sparse_diagonal_like(B,D.diag))
        end
        function Base.$op(D::Diagonal,A::Union{GPUSparseMatrix,GPUSparseTransposed})
            B=materialize(A)
            size(B)==size(D) || throw(DimensionMismatch("matrix dimensions must match"))
            sparse_map($op,sparse_diagonal_like(B,D.diag),B)
        end
    end
end

# Explicit errors prevent SparseArrays' concatenation fallbacks from downloading through
# SparseMatrixCSC. These operations can be implemented centrally without backend changes.
const GPUSparseConcatOperand = Union{GPUSparseArray,GPUSparseTransposed}
const GPUSparseConcatInput = Union{GPUSparseConcatOperand,AnyGPUArray}
# SparseArrays uses this helper for every position in hcat/vcat/cat/block syntax.
# Reject before its conversion of the first GPU operand can download any buffers.
SparseArrays._makesparse(::GPUSparseVector) = throw(ArgumentError("GPU sparse concatenation is not implemented"))
SparseArrays._makesparse(::Union{GPUSparseMatrix,GPUSparseTransposed}) =
    throw(ArgumentError("GPU sparse concatenation is not implemented"))
SparseArrays._makesparse(::Union{AnyGPUVector,AnyGPUMatrix}) =
    throw(ArgumentError("concatenation involving GPU arrays and sparse operands is not implemented"))
for f in (:hcat, :vcat)
    @eval begin
        Base.$f(A::GPUSparseConcatOperand, Bs::GPUSparseConcatInput...) =
            throw(ArgumentError($(string(f, " of GPU sparse arrays is not implemented; concatenate coordinates and assemble explicitly"))))
        Base.$f(A::AnyGPUArray, B::GPUSparseConcatOperand, Cs::GPUSparseConcatInput...) =
            throw(ArgumentError($(string(f, " with GPU sparse arrays is not implemented"))))
    end
end
SparseArrays.blockdiag(A::GPUSparseMatrix, Bs::GPUSparseMatrix...) =
    throw(ArgumentError("blockdiag of GPU sparse arrays is not implemented"))
