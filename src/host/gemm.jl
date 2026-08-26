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
    else # 'H' / 'h'
        symherm(X, i, j, Val(TX === 'H'), Val(true))
    end
end

const MAX_TILE_DIM = 16
@kernel unsafe_indices = true function coalesced_matmul_kernel!(
        C, A, B, add,
        M, N, K, ::Val{TA}, ::Val{TB},
        ::Val{TILE},
    ) where {TA, TB, TILE}
    @uniform TAT = eltype(A); @uniform TBT = eltype(B); @uniform R = eltype(C)

    # li, lj, _ = KI.get_local_id()
    li, lj = @index(Local, NTuple)

    # grow, gcol, _ = KI.get_group_id()
    grow, gcol = @index(Group, NTuple)
    # Uncomment once moved to KernelInterface
    # gi = (grow - 1) * TILE + li
    # gj = (gcol - 1) * TILE + lj

    # padded by one row to avoid bank conflicts on shared memory
    tile1 = @localmem TAT (TILE + 1, TILE)
    tile2 = @localmem TBT (TILE + 1, TILE)

    # private variable for tile C
    @uniform Tacc = promote_type(R, typeof(zero(TAT) * zero(TBT) + zero(TAT) * zero(TBT)))
    # acc = -zero(Tacc) #Use this when KernelInterface
    acc = @private Tacc 1
    @inbounds acc[1] = zero(Tacc)

    # number of tiles depends on inner dimension
    # NUM_TILES = div(K + TILE - 1, TILE)
    @uniform NUM_TILES = cld(K, TILE)

    # loop over all tiles needed for this calculation
    for t in 0:(NUM_TILES - 1)
        # Remove once moved to KernelInterface
        gi = (grow - 1) * TILE + li
        gj = (gcol - 1) * TILE + lj

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
         @synchronize()

        # Remove once moved to KernelInterface
         gi = (grow - 1) * TILE + li
         gj = (gcol - 1) * TILE + lj

        @inbounds for k in 1:TILE
            acc[1] = muladd(tile1[li, k], tile2[k, lj], acc[1])
        end

         @synchronize()
    end

    # Remove once moved to KernelInterface
    gi = (grow - 1) * TILE + li
    gj = (gcol - 1) * TILE + lj

    # save if inbounds
    if gi <= M && gj <= N
        @inbounds C[gi, gj] = add(acc[1], C[gi, gj])
    end
    # return
end

@inline function check_matmul_shapes(C, A, B, cA='N', cB='N')
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
    cA, cB = Char(cA), Char(cB)
    M, N, K = check_matmul_shapes(C, A, B, cA, cB)

    # an empty inner dimension still needs to apply β; the kernel handles that (no tiles)
    isempty(C) && return C

    workgroupsize=(MAX_TILE_DIM, MAX_TILE_DIM)
    numworkgroups=(cld(M, MAX_TILE_DIM), cld(N, MAX_TILE_DIM))

    # KI.@kernel KI.get_backend(C) workgroupsize=workgroupsize numworkgroups=numworkgroups coalesced_matmul_kernel!(
            # C, A, B, add, Int(M), Int(N), Int(K), Val(cA), Val(cB), Val(MAX_TILE_DIM))
    coalesced_matmul_kernel!(get_backend(C), workgroupsize)(
            C, A, B, add, Int(M), Int(N), Int(K), Val(cA), Val(cB), Val(MAX_TILE_DIM); ndrange=numworkgroups .* workgroupsize)
    C
end
