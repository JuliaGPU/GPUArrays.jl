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
    for t in 0:(NUM_TILES - 1)
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

# acc[r + (c-1)*TM] += a[r] * b[c]
@inline outer_muladd(a::NTuple{TM}, b::NTuple{TN}, acc::NTuple{L}) where {TM, TN, L} =
    ntuple(Val(L)) do q
        @inline
        r, c = (q - 1) % TM + 1, (q - 1) ÷ TM + 1
        @inbounds muladd(a[r], b[c], acc[q])
    end

# each work-item of a (TX, TY) workgroup computes a TM×TN block of a (TX*TM)×(TY*TN) tile of C
function blocked_matmul_kernel!(
        C, A, B, add,
        M, N, K, ::Val{TA}, ::Val{TB},
        ::Val{TX}, ::Val{TY}, ::Val{TM}, ::Val{TN}, ::Val{BK},
    ) where {TA, TB, TX, TY, TM, TN, BK}
    TAT = eltype(A); TBT = eltype(B); R = eltype(C)
    BM, BN, NT = TX * TM, TY * TN, TX * TY

    li, lj, _ = KI.get_local_id()
    grow, gcol, _ = KI.get_group_id()
    row0, col0 = (grow - 1) * BM, (gcol - 1) * BN
    tid = (li - 1) + (lj - 1) * TX

    # B is stored transposed, so that work-items with consecutive `lj` read consecutive elements
    tileA = KI.localmemory(TAT, (BM, BK))
    tileB = KI.localmemory(TBT, (BN, BK))

    Tacc = promote_type(R, typeof(zero(TAT) * zero(TBT) + zero(TAT) * zero(TBT)))
    acc = ntuple(_ -> zero(Tacc), Val(TM * TN))

    # load along the contiguous dimension of the stored matrix
    a_kfast = TA === 'T' || TA === 'C'
    b_kfast = !(TB === 'T' || TB === 'C')

    for k0 in 0:BK:(K - 1)
        for s in 0:(BM * BK ÷ NT - 1)
            idx = tid + s * NT
            i, k = a_kfast ? (idx ÷ BK, idx % BK) : (idx % BM, idx ÷ BM)
            gi, gk = row0 + i + 1, k0 + k + 1
            @inbounds tileA[i + 1, k + 1] =
                gi <= M && gk <= K ? opelem(A, Val(TA), gi, gk) : zero(TAT)
        end
        for s in 0:(BN * BK ÷ NT - 1)
            idx = tid + s * NT
            j, k = b_kfast ? (idx ÷ BK, idx % BK) : (idx % BN, idx ÷ BN)
            gj, gk = col0 + j + 1, k0 + k + 1
            @inbounds tileB[j + 1, k + 1] =
                gj <= N && gk <= K ? opelem(B, Val(TB), gk, gj) : zero(TBT)
        end

        KI.barrier()

        for k in 1:BK
            a = ntuple(r -> @inbounds(tileA[li + (r - 1) * TX, k]), Val(TM))
            b = ntuple(c -> @inbounds(tileB[lj + (c - 1) * TY, k]), Val(TN))
            acc = outer_muladd(a, b, acc)
        end

        KI.barrier()
    end

    for c in 1:TN, r in 1:TM
        gi, gj = row0 + li + (r - 1) * TX, col0 + lj + (c - 1) * TY
        if gi <= M && gj <= N
            @inbounds C[gi, gj] = add(acc[r + (c - 1) * TM], C[gi, gj])
        end
    end
    return
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

    backend = KI.get_backend(C)
    # the blocked kernel needs enough workgroups to fill the device
    if M * N >= 2^19 && isbitstype(T) && isbitstype(S) && max(sizeof(T), sizeof(S)) <= 16
        TX = TY = 16
        TM = TN = 4
        BK = 8
        KI.@launch backend workgroupsize=(TX, TY) numgroups=(cld(M, TX * TM), cld(N, TY * TN)) blocked_matmul_kernel!(
                C, A, B, add, Int(M), Int(N), Int(K), Val(cA), Val(cB),
                Val(TX), Val(TY), Val(TM), Val(TN), Val(BK))
    else
        KI.@launch backend workgroupsize=(TILE_DIM, TILE_DIM) numgroups=(cld(M, TILE_DIM), cld(N, TILE_DIM)) coalesced_matmul_kernel!(
                C, A, B, add, Int(M), Int(N), Int(K), Val(cA), Val(cB), Val(TILE_DIM))
    end
    C
end
