# array semantics that SparseArrays implements with scalar loops on the host

## equality

# compare the values, not the stored structure, so that stored zeros do not matter
Base.:(==)(A::GPUSparseArray, B::GPUSparseArray) = size(A) == size(B) && !any(A .!= B)
Base.:(==)(A::GPUSparseArray, B::AnyGPUArray) = size(A) == size(B) && !any(A .!= B)
Base.:(==)(A::AnyGPUArray, B::GPUSparseArray) = B == A


## in-place broadcast

# A broadcast into a sparse destination is materialized first, which gives it the
# structure of the result (and copes with the destination being an argument), and then
# copied over. A result that is not sparse (dense arguments, or a function that does not
# preserve zeros) cannot be stored.
function sparse_broadcast!(dest::GPUSparseArray, bc::Broadcasted)
    bc = Broadcast.instantiate(Broadcast.broadcasted(bc.f, bc.args...))
    if isempty(axes(bc))
        # `A .= x`: only a zero can fill a sparse array
        fill!(dest, bc[])
        return dest
    end
    axes(dest) == axes(bc) ||
        throw(DimensionMismatch("destination has axes $(axes(dest)), result has axes $(axes(bc))"))
    result = copy(bc)
    result isa GPUSparseArray ||
        throw(ArgumentError("cannot broadcast a dense result into a sparse array"))
    if dest isa GPUSparseMatrix
        return copyto!(dest, convert(matrix_format(dest), result))
    else
        return copyto!(dest, result)
    end
end
Broadcast.materialize!(dest::GPUSparseArray, bc::Broadcasted) = sparse_broadcast!(dest, bc)
Base.copyto!(dest::GPUSparseArray, bc::Broadcasted) = sparse_broadcast!(dest, bc)
Base.copyto!(dest::GPUSparseArray, bc::Broadcasted{Nothing}) = sparse_broadcast!(dest, bc)
# (disambiguation with the dense GPU and the zero-dimensional methods)
for Style in (AbstractGPUArrayStyle, Broadcast.AbstractArrayStyle{0}, AbstractGPUArrayStyle{0})
    @eval Base.copyto!(dest::GPUSparseArray, bc::Broadcasted{<:$Style}) =
        sparse_broadcast!(dest, bc)
end

function Base.fill!(A::GPUSparseArray, x)
    iszero(x) || throw(ArgumentError("a sparse array can only be filled with zeros"))
    fill!(nonzeros(A), x)
    return A
end


## linear indexing

Base.getindex(A::GPUSparseMatrix, ::Colon) = vec(A)
Base.getindex(A::GPUSparseMatrix, I::AbstractRange{<:Integer}) = vec(A)[I]


## lazy wrappers

# Operations with transposed, Symmetric and Hermitian sparse matrices materialize the
# wrapper, as the sparse kernels only index the formats themselves. (Products are the
# exception; see `linalg.jl`.)
const GPUSparseTransposed = Union{Transpose{<:Any,<:GPUSparseMatrix},
                                  Adjoint{<:Any,<:GPUSparseMatrix}}
const GPUSparseStructured = Union{Symmetric{<:Any,<:GPUSparseMatrix},
                                  Hermitian{<:Any,<:GPUSparseMatrix}}
const GPUSparseWrapped = Union{GPUSparseTransposed,GPUSparseStructured}

for op in (:+, :-)
    @eval begin
        Base.$op(A::GPUSparseWrapped, B::GPUSparseWrapped) =
            sparse_map($op, materialize(A), materialize(B))
        Base.$op(A::GPUSparseMatrix, B::GPUSparseWrapped) = sparse_map($op, A, materialize(B))
        Base.$op(A::GPUSparseWrapped, B::GPUSparseMatrix) = sparse_map($op, materialize(A), B)

        # (disambiguation with SparseArrays)
        Base.$op(A::Symmetric{<:Real,<:GPUSparseMatrix}, B::Hermitian{<:Any,<:GPUSparseMatrix}) =
            sparse_map($op, materialize(A), materialize(B))
        Base.$op(A::Hermitian{<:Any,<:GPUSparseMatrix}, B::Symmetric{<:Real,<:GPUSparseMatrix}) =
            sparse_map($op, materialize(A), materialize(B))
    end
end

# (LinearAlgebra keeps Symmetric and Hermitian wrappers when scaling by a real number,
# which works through the parent, but not a Hermitian matrix by a complex one)
for (S, T) in ((:GPUSparseTransposed, :Number), (:(Hermitian{<:Any,<:GPUSparseMatrix}), :Complex))
    @eval begin
        Base.:(*)(A::$S, x::$T) = materialize(A) .* x
        Base.:(*)(x::$T, A::$S) = x .* materialize(A)
        Base.:(/)(A::$S, x::$T) = materialize(A) ./ x
        Base.:(\)(x::$T, A::$S) = x .\ materialize(A)
    end
end

# (Julia 1.10's `power_by_squaring` would convert the wrapper to the inferred, non-concrete
# type of its product)
Base.:(^)(A::GPUSparseTransposed, p::Integer) = materialize(A)^p

SparseArrays.nnz(A::GPUSparseTransposed) = nnz(parent(A))
SparseArrays.nonzeros(A::Transpose{<:Any,<:GPUSparseMatrix}) = nonzeros(parent(A))
SparseArrays.nonzeros(A::Adjoint{<:Any,<:GPUSparseMatrix}) = conj.(nonzeros(parent(A)))
Base.mapreduce(f, op, A::GPUSparseTransposed; kwargs...) = mapreduce(f, op, copy(A); kwargs...)
Base.getindex(A::GPUSparseTransposed, I::SparseRangeIndex, J::SparseRangeIndex) = copy(A)[I, J]

SparseArrays.sparse(A::GPUSparseWrapped) = materialize(A)
SparseArrays.SparseMatrixCSC(A::GPUSparseWrapped) = SparseMatrixCSC(materialize(A))
SparseArrays.SparseMatrixCSC{Tv}(A::GPUSparseWrapped) where {Tv} =
    SparseMatrixCSC{Tv}(SparseMatrixCSC(A))
SparseArrays.SparseMatrixCSC{Tv,Ti}(A::GPUSparseWrapped) where {Tv,Ti} =
    SparseMatrixCSC{Tv,Ti}(SparseMatrixCSC(A))
Base.Array(A::GPUSparseWrapped) = Array(materialize(A))
Base.collect(A::GPUSparseWrapped) = Array(A)

# (LinearAlgebra's constructors would convert the parent through scalar indexing. The
# element type is taken from the wrapper type because Julia 1.10 cannot bind it from
# `::Type{Symmetric{T}}`.)
(W::Type{Symmetric{T}} where {T})(A::GPUSparseMatrix, uplo::Symbol=:U) =
    Symmetric(with_eltypes(A, eltype(W), indtype(A)), uplo)
(W::Type{Hermitian{T}} where {T})(A::GPUSparseMatrix, uplo::Symbol=:U) =
    Hermitian(with_eltypes(A, eltype(W), indtype(A)), uplo)


## diagonal matrices and uniform scaling

# a diagonal matrix with the format and storage of `A`
function sparse_diagonal(A::GPUSparseMatrix, d::AbstractVector, dims::Dims{2}=size(A))
    length(d) == min(dims...) ||
        throw(DimensionMismatch("a $(dims[1])×$(dims[2]) matrix has $(min(dims...)) diagonal entries, got $(length(d))"))
    D = spdiagm(dims..., 0 => move_buffer(nonzeros(A), d))
    return convert(matrix_format(A), with_eltypes(D, eltype(D), indtype(A)))
end
sparse_identity(A::GPUSparseMatrix, λ) =
    sparse_diagonal(A, fill!(similar(nonzeros(A), typeof(λ), min(size(A)...)), λ))

Base.one(A::GPUSparseMatrix) = (LinearAlgebra.checksquare(A); sparse_identity(A, one(eltype(A))))
Base.oneunit(A::GPUSparseMatrix) = (LinearAlgebra.checksquare(A); sparse_identity(A, oneunit(eltype(A))))

const GPUSparseMatrixOrWrapped = Union{GPUSparseMatrix,GPUSparseWrapped}

# (unlike with `UniformScaling`, which SparseArrays also adds to rectangular matrices)
check_diagonal_dims(A, D::Diagonal) = size(A) == size(D) ||
    throw(DimensionMismatch("cannot combine a $(join(size(A), '×')) matrix with a $(join(size(D), '×')) diagonal matrix"))

# wrappers compare by value too
Base.:(==)(A::GPUSparseWrapped, B::GPUSparseMatrixOrWrapped) = materialize(A) == materialize(B)
Base.:(==)(A::GPUSparseMatrix, B::GPUSparseWrapped) = A == materialize(B)
Base.:(*)(A::GPUSparseMatrixOrWrapped, J::UniformScaling) = materialize(A) .* J.λ
Base.:(*)(J::UniformScaling, A::GPUSparseMatrixOrWrapped) = J.λ .* materialize(A)
for op in (:+, :-)
    @eval begin
        function Base.$op(A::GPUSparseMatrixOrWrapped, J::UniformScaling)
            B = materialize(A)
            sparse_map($op, B, sparse_identity(B, J.λ))
        end
        function Base.$op(J::UniformScaling, A::GPUSparseMatrixOrWrapped)
            B = materialize(A)
            sparse_map($op, sparse_identity(B, J.λ), B)
        end
        function Base.$op(A::GPUSparseMatrixOrWrapped, D::Diagonal)
            B = materialize(A)
            check_diagonal_dims(B, D)
            sparse_map($op, B, sparse_diagonal(B, D.diag))
        end
        function Base.$op(D::Diagonal, A::GPUSparseMatrixOrWrapped)
            B = materialize(A)
            check_diagonal_dims(B, D)
            sparse_map($op, sparse_diagonal(B, D.diag), B)
        end
    end
end
# (disambiguation with LinearAlgebra's Symmetric and Hermitian methods)
for op in (:+, :-), (Wrapper, Tλ, Td) in ((:Symmetric, :Number, :Number),
                                         (:Hermitian, :Complex, :Real))
    @eval begin
        Base.$op(A::$Wrapper{<:Any,<:GPUSparseMatrix}, J::UniformScaling{<:$Tλ}) =
            invoke($op, Tuple{GPUSparseMatrixOrWrapped,UniformScaling}, A, J)
        Base.$op(J::UniformScaling{<:$Tλ}, A::$Wrapper{<:Any,<:GPUSparseMatrix}) =
            invoke($op, Tuple{UniformScaling,GPUSparseMatrixOrWrapped}, J, A)
        Base.$op(A::$Wrapper{<:Any,<:GPUSparseMatrix}, D::Diagonal{<:$Td}) =
            invoke($op, Tuple{GPUSparseMatrixOrWrapped,Diagonal}, A, D)
        Base.$op(D::Diagonal{<:$Td}, A::$Wrapper{<:Any,<:GPUSparseMatrix}) =
            invoke($op, Tuple{Diagonal,GPUSparseMatrixOrWrapped}, D, A)
    end
end

# scaling the rows or columns of a sparse matrix keeps its structure
function scale_sparse(A::GPUSparseMatrixOrWrapped, D::Diagonal, side::Symbol)
    B = materialize(A)
    n = side === :right ? size(B, 2) : size(B, 1)
    size(D, 1) == n ||
        throw(DimensionMismatch("cannot multiply a $(join(size(B), '×')) matrix with a $(join(size(D), '×')) diagonal matrix on the $side"))
    d = move_buffer(nonzeros(B), D.diag)
    rows, cols = entry_coordinates(B)
    vals = side === :right ? nonzeros(B) .* d[cols] : d[rows] .* nonzeros(B)
    return sparse_format_type(B)(map(copy, Base.front(sparse_buffers(B)))..., vals, size(B))
end
Base.:(*)(A::GPUSparseMatrixOrWrapped, D::Diagonal) = scale_sparse(A, D, :right)
Base.:(*)(D::Diagonal, A::GPUSparseMatrixOrWrapped) = scale_sparse(A, D, :left)


## concatenation

# The coordinates of the entries are offset and concatenated, and the result assembled
# from them: there are no duplicates, and the entries of each array come out in order.
function sparse_cat(dim::Integer, As::GPUSparseMatrix...)
    A = first(As)
    other = 3 - dim
    all(B -> size(B, other) == size(A, other), As) ||
        throw(DimensionMismatch("mismatch in dimension $other of the concatenated matrices"))
    Ti = mapreduce(indtype, promote_type, As)
    Tv = mapreduce(eltype, promote_type, As)
    dims = ntuple(d -> d == dim ? sum(B -> size(B, dim), As) : size(A, d), 2)
    check_sparse_dims(Ti, dims)
    Is, Js, Vs = [], [], []
    offset = 0
    for B in As
        rows, cols = entry_coordinates(B)
        push!(Is, rows .+ Ti(dim == 1 ? offset : 0))
        push!(Js, cols .+ Ti(dim == 2 ? offset : 0))
        push!(Vs, convert_buffer(Tv, nonzeros(B)))
        offset += size(B, dim)
    end
    return assemble_matrix(reduce(vcat, Is), reduce(vcat, Js), reduce(vcat, Vs), dims..., +,
                           matrix_format_symbol(A))
end
matrix_format_symbol(::GPUSparseMatrixCSR) = :csr
matrix_format_symbol(::GPUSparseMatrixCSC) = :csc
matrix_format_symbol(::GPUSparseMatrixCOO) = :coo

Base.hcat(A::GPUSparseMatrixOrWrapped, As::GPUSparseMatrixOrWrapped...) =
    sparse_cat(2, materialize(A), map(materialize, As)...)
Base.vcat(A::GPUSparseMatrixOrWrapped, As::GPUSparseMatrixOrWrapped...) =
    sparse_cat(1, materialize(A), map(materialize, As)...)

# vectors concatenate to a vector, or side by side to a matrix, as in SparseArrays
function Base.vcat(x::GPUSparseVector, rest::GPUSparseVector...)
    xs = (x, rest...)
    Ti = mapreduce(indtype, promote_type, xs)
    Tv = mapreduce(eltype, promote_type, xs)
    len = sum(length, xs)
    check_sparse_dims(Ti, (len,))
    inds, vals = [], []
    offset = 0
    for x in xs
        push!(inds, x.nzInd .+ Ti(offset))
        push!(vals, convert_buffer(Tv, nonzeros(x)))
        offset += length(x)
    end
    return GPUSparseVector(reduce(vcat, inds), reduce(vcat, vals), len)
end
Base.hcat(x::GPUSparseVector, xs::GPUSparseVector...) =
    sparse_cat(2, GPUSparseMatrixCSC(x), map(GPUSparseMatrixCSC, xs)...)

SparseArrays.blockdiag(A::GPUSparseMatrix, Bs::GPUSparseMatrix...) =
    throw(ArgumentError("blockdiag of GPU sparse arrays is not supported"))
