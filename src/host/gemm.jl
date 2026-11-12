using LinearAlgebra: MulAddMul, wrapper_char, _unwrap

## Matmul kernel

# A[i, j] of a symmetric/hermitian matrix with only one triangle stored
@inline function symherm(A, i, j, ::Val{upper}, ::Val{conjugate}) where {upper, conjugate}
    stored = upper ? (i <= j) : (i >= j)
    val = stored ? (@inbounds A[i, j]) : (@inbounds A[j, i])
    if conjugate
        if i == j
            val = oftype(val, real(val))
        elseif !stored
            val = conj(val)
        end
    end
    return val
end

# op(X)[i, j], reading the stored (possibly transposed/wrapped) matrix. used for both
# operands: A's element is op(A)[outrow, contr], B's is op(B)[contr, outcol]
@inline function opelem(X, ::Val{TX}, i, j) where {TX}
    if TX === 'N'
        @inbounds X[i, j]
    elseif TX === 'C'
        @inbounds conj(X[j, i])
    elseif TX === 'T'
        @inbounds X[j, i]
    elseif TX === 'S' || TX === 's'
        symherm(X, i, j, Val(TX === 'S'), Val(false))
    elseif TX isa Char # 'H' / 'h'
        symherm(X, i, j, Val(TX === 'H'), Val(true))
    else
        triangular(X, i, j, Val(TX))
    end
end

# op(X)[i, j] of a triangular X, for a `(trans, uplo, diag)` tuple describing the stored X
@inline function triangular(X, i, j, ::Val{TX}) where {TX}
    trans, uplo, diag = TX
    si, sj = trans === 'N' ? (i, j) : (j, i)
    if diag === 'U' && si == sj
        oneunit(eltype(X))
    elseif uplo === 'U' ? si <= sj : si >= sj
        val = @inbounds X[si, sj]
        trans === 'C' ? conj(val) : val
    else
        zero(eltype(X))
    end
end

# whether op(X) is upper triangular
op_upper(TX) = (TX[2] === 'U') == (TX[1] === 'N')

# the tiles along the inner dimension that a triangular operand doesn't zero out
@inline function tile_range(::Val{TA}, ::Val{TB}, grow, gcol, NUM_TILES) where {TA, TB}
    lo, hi = 0, NUM_TILES - 1
    if TA isa Tuple
        op_upper(TA) ? (lo = max(lo, grow - 1)) : (hi = min(hi, grow - 1))
    end
    if TB isa Tuple
        op_upper(TB) ? (hi = min(hi, gcol - 1)) : (lo = max(lo, gcol - 1))
    end
    lo:hi
end

const TILE_DIM = 16
function coalesced_matmul_kernel!(
        C, A, B, add,
        M, N, K, ::Val{TA}, ::Val{TB},
        ::Val{TILE},
    ) where {TA, TB, TILE}
    TAT = eltype(A); TBT = eltype(B); R = eltype(C)

    li, lj, _ = KI.get_local_id()
    grow, gcol, _ = KI.get_group_id()
    gi = (grow - 1) * TILE + li
    gj = (gcol - 1) * TILE + lj

    # no padding: consecutive work-items (`li`) access consecutive elements of a column, or
    # all the same element of `tile2`, so there are no bank conflicts to avoid. unpadded
    # columns stay aligned, which lets each work-item read `tile2[:, lj]` with vector loads.
    tile1 = KI.localmemory(TAT, (TILE, TILE))
    tile2 = KI.localmemory(TBT, (TILE, TILE))

    # private variable for tile C
    Tacc = promote_type(R, typeof(zero(TAT) * zero(TBT) + zero(TAT) * zero(TBT)))
    acc = zero(Tacc)

    # number of tiles depends on inner dimension
    NUM_TILES = cld(K, TILE)

    # loop over all tiles needed for this calculation
    for t in tile_range(Val(TA), Val(TB), grow, gcol, NUM_TILES)
        # load inputs into tiles, with bounds checking for non-square matrices
        k0 = t * TILE
        ac = k0 + lj
        if gi <= M && ac <= K
            @inbounds tile1[li, lj] = opelem(A, Val(TA), gi, ac)
        else
            @inbounds tile1[li, lj] = zero(TAT)
        end
        ar = k0 + li
        if gj <= N && ar <= K
            @inbounds tile2[li, lj] = opelem(B, Val(TB), ar, gj)
        else
            @inbounds tile2[li, lj] = zero(TBT)
        end

        # wait for all tiles to be loaded
        KI.barrier()

        @inbounds for k in 1:TILE
            acc = muladd(tile1[li, k], tile2[k, lj], acc)
        end

        KI.barrier()
    end

    # save if inbounds
    if gi <= M && gj <= N
        @inbounds C[gi, gj] = add(acc, C[gi, gj])
    end
    return
end

op_char(op::Char) = op
op_char(op::Tuple) = first(op)

@inline function check_matmul_shapes(C, A, B, cA='N', cB='N')
    cA, cB = op_char(cA), op_char(cB)
    tA = cA === 'T' || cA === 'C'
    tB = cB === 'T' || cB === 'C'
    M = tA ? size(A, 2) : size(A, 1)
    K = tA ? size(A, 1) : size(A, 2)
    N = tB ? size(B, 1) : size(B, 2)
    KB = tB ? size(B, 2) : size(B, 1)
    if K != KB
        throw(DimensionMismatch("matrix A has dimensions $(size(A)), matrix B has dimensions $(size(B))"))
    end
    if size(C,1) != M || size(C,2) != N
        throw(DimensionMismatch("result C has dimensions $(size(C)), needs $((M,N))"))
    end

    return M, N, K
end

# legacy methods
generic_matmatmul!(C::AbstractArray, A::AbstractArray, B::AbstractArray, a::Number, b::Number) =
    generic_matmatmul!(C, wrapper_char(A), wrapper_char(B), _unwrap(A), _unwrap(B), MulAddMul(a, b))
generic_matmatmul!(C::AbstractArray, A::AbstractArray, B::AbstractArray, add::MulAddMul) =
    generic_matmatmul!(C, wrapper_char(A), wrapper_char(B), _unwrap(A), _unwrap(B), add)

# New method
function generic_matmatmul!(C::AbstractArray{R}, cA::AbstractChar, cB::AbstractChar, A::AbstractArray{T}, B::AbstractArray{S}, add::MulAddMul) where {T, S, R}
    # Char(::WrapperChar) encodes the uplo flag as upper/lowercase ('S'/'s', 'H'/'h')
    tiled_matmatmul!(C, Char(cA), Char(cB), A, B, add)
end

@static if isdefined(LinearAlgebra, Symbol("@stable_muladdmul")) # added in 1.12
    tiled_matmatmul!(C, opA, opB, A, B, α::Number, β::Number) =
        LinearAlgebra.@stable_muladdmul tiled_matmatmul!(C, opA, opB, A, B, MulAddMul(α, β))
else
    tiled_matmatmul!(C, opA, opB, A, B, α::Number, β::Number) =
        tiled_matmatmul!(C, opA, opB, A, B, MulAddMul(α, β))
end

# `opA` and `opB` are characters as above, or `(trans, uplo, diag)` for triangular operands
function tiled_matmatmul!(C::AbstractArray, opA, opB, A::AbstractArray, B::AbstractArray, add::MulAddMul)
    M, N, K = check_matmul_shapes(C, A, B, opA, opB)

    # an empty inner dimension still needs to apply β; the kernel handles that (no tiles)
    isempty(C) && return C

    workgroupsize=(TILE_DIM, TILE_DIM)
    numworkgroups=(cld(M, TILE_DIM), cld(N, TILE_DIM))

    KI.@launch KI.get_backend(C) workgroupsize=workgroupsize numgroups=numworkgroups coalesced_matmul_kernel!(
            C, A, B, add, Int(M), Int(N), Int(K), Val(opA), Val(opB), Val(TILE_DIM))
    C
end
