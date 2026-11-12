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
    N == 1 && return generic_matvec!(C, cA, cB, A, B, add, M, K)

    workgroupsize=(TILE_DIM, TILE_DIM)
    numworkgroups=(cld(M, TILE_DIM), cld(N, TILE_DIM))

    KI.@launch KI.get_backend(C) workgroupsize=workgroupsize numgroups=numworkgroups coalesced_matmul_kernel!(
            C, A, B, add, Int(M), Int(N), Int(K), Val(cA), Val(cB), Val(TILE_DIM))
    C
end

## Matvec kernels

# a matrix-vector product splits into a part that reads A along its rows, op(A)[i, k] =
# A[i, k], and one that reads it along its columns, op(A)[i, k] = A[k, i]; the symmetric and
# Hermitian cases have both. each kernel can split the inner dimension over several
# work-groups, in which case it writes partial sums to `P[s, i]`, which are then summed by
# the column kernel with `x === nothing`.

matvec_acctype(C, A, x) =
    promote_type(eltype(C), typeof(zero(eltype(A)) * zero(eltype(x)) + zero(eltype(A)) * zero(eltype(x))))

@inline matvec_x(x, ::Val{TX}, k) where {TX} = opelem(x, Val(TX), k, 1)
@inline matvec_x(::Nothing, ::Val, k) = true

# y[i] = Σ_k A[i, k] x[k], for k ≥ i (upper), k ≤ i (lower) or all k. a work-group covers
# TM rows, with TK work-items per row each taking every TK-th k
function matvec_rows_kernel!(C, P, ::Type{Tacc}, A, x, add, M, K, kchunk, off, ::Val{TX}, ::Val{region},
                             ::Val{herm}, ::Val{direct}, ::Val{TM}, ::Val{TK}) where {Tacc, TX, region, herm, direct, TM, TK}
    li, lk, _ = KI.get_local_id()
    g, s, _ = KI.get_group_id()
    i = (g - 1) * TM + li

    klo = (s - 1) * kchunk + 1
    khi = min(s * kchunk, K)
    region === :upper && (klo = max(klo, i))
    region === :lower && (khi = min(khi, i))

    acc = zero(Tacc)
    if i <= M
        for k in (klo + lk - 1):TK:khi
            a = @inbounds A[i, k]
            herm && k == i && (a = oftype(a, real(a)))
            acc = muladd(a, matvec_x(x, Val(TX), k), acc)
        end
    end

    scratch = KI.localmemory(Tacc, (TM, TK))
    @inbounds scratch[li, lk] = acc
    KI.barrier()
    if lk == 1 && i <= M
        @inbounds for l in 2:TK
            acc += scratch[li, l]
        end
        if direct
            @inbounds C[i, 1] = add(acc, C[i, 1])
        else
            @inbounds P[off + s, i] = acc
        end
    end
    return
end

# y[j] = Σ_k op(A[k, j]) x[k], for k < j (above), k > j (below) or all k. a work-group covers
# NT ÷ tk columns, with tk work-items per column whose sums are combined in local memory
function matvec_cols_kernel!(C, P, ::Type{Tacc}, A, x, add, M, K, kchunk, off, tk, ::Val{TX}, ::Val{region},
                             ::Val{conjugate}, ::Val{direct}, ::Val{NT}) where {Tacc, TX, region, conjugate, direct, NT}
    lk, lj, _ = KI.get_local_id()
    g, s, _ = KI.get_group_id()
    j = (g - 1) * (NT ÷ tk) + lj

    klo = (s - 1) * kchunk + 1
    khi = min(s * kchunk, K)
    region === :below && (klo = max(klo, j + 1))
    region === :above && (khi = min(khi, j - 1))

    acc = zero(Tacc)
    if j <= M
        for k in (klo + lk - 1):tk:khi
            a = @inbounds A[k, j]
            conjugate && (a = conj(a))
            acc = muladd(a, matvec_x(x, Val(TX), k), acc)
        end
    end

    if tk > 1
        scratch = KI.localmemory(Tacc, NT)
        idx = lk + (lj - 1) * tk
        @inbounds scratch[idx] = acc
        stride = tk ÷ 2
        while stride >= 1
            KI.barrier()
            if lk <= stride
                @inbounds scratch[idx] += scratch[idx + stride]
            end
            stride ÷= 2
        end
        acc = @inbounds scratch[idx]
    end
    if lk == 1 && j <= M
        if direct
            @inbounds C[j, 1] = add(acc, C[j, 1])
        else
            @inbounds P[off + s, j] = acc
        end
    end
    return
end

const MATVEC_ROWS = (64, 4)
const MATVEC_COLS = 256
const MATVEC_GROUPS = 1024

# split the inner dimension so that there are enough work-groups, but at least a few
# iterations per work-item
function matvec_splits(groups, K, items)
    K = max(K, 1)
    S = clamp(cld(MATVEC_GROUPS, groups), 1, cld(K, 32 * items))
    return cld(K, S), cld(K, cld(K, S))
end

matvec_rows_geometry(M, K) = matvec_splits(cld(M, MATVEC_ROWS[1]), K, MATVEC_ROWS[2])

function matvec_cols_geometry(M, K)
    tk = min(nextpow(2, cld(max(K, 1), 4)), MATVEC_COLS)
    return tk, matvec_splits(cld(M, MATVEC_COLS ÷ tk), K, tk)...
end

function launch_matvec_rows!(C, P, Tacc, A, x, add, M, K, off, cx, region, herm, direct)
    kchunk, S = matvec_rows_geometry(M, K)
    KI.@launch KI.get_backend(C) workgroupsize=MATVEC_ROWS numgroups=(cld(M, MATVEC_ROWS[1]), S) matvec_rows_kernel!(
        C, P, Tacc, A, x, add, Int(M), Int(K), Int(kchunk), Int(off), Val(cx), Val(region), Val(herm),
        Val(direct), Val(MATVEC_ROWS[1]), Val(MATVEC_ROWS[2]))
end

function launch_matvec_cols!(C, P, Tacc, A, x, add, M, K, off, cx, region, conjugate, direct)
    tk, kchunk, S = matvec_cols_geometry(M, K)
    # the final sum over partial results has to be done in one pass
    P === nothing && ((kchunk, S) = (max(K, 1), 1))
    KI.@launch KI.get_backend(C) workgroupsize=(tk, MATVEC_COLS ÷ tk) numgroups=(cld(M, MATVEC_COLS ÷ tk), S) matvec_cols_kernel!(
        C, P, Tacc, A, x, add, Int(M), Int(K), Int(kchunk), Int(off), Int(tk), Val(cx), Val(region),
        Val(conjugate), Val(direct), Val(MATVEC_COLS))
end

# C = add(op(A) * op(x), C) for a single column C
function generic_matvec!(C, cA, cx, A, x, add, M, K)
    A, x = Base.unalias(C, A), Base.unalias(C, x)
    Tacc = matvec_acctype(C, A, x)
    if cA == 'N' || cA == 'T' || cA == 'C'
        rows = cA == 'N'
        S = rows ? matvec_rows_geometry(M, K)[2] : matvec_cols_geometry(M, K)[3]
        if S == 1
            rows ? launch_matvec_rows!(C, nothing, Tacc, A, x, add, M, K, 0, cx, :all, false, true) :
                   launch_matvec_cols!(C, nothing, Tacc, A, x, add, M, K, 0, cx, :all, cA == 'C', true)
            return C
        end
        P = similar(C, Tacc, (S, M))
        rows ? launch_matvec_rows!(C, P, Tacc, A, x, add, M, K, 0, cx, :all, false, false) :
               launch_matvec_cols!(C, P, Tacc, A, x, add, M, K, 0, cx, :all, cA == 'C', false)
    else
        # the stored triangle is read along its rows, the other one along its columns
        upper = cA == 'S' || cA == 'H'
        herm = cA == 'H' || cA == 'h'
        S1 = matvec_rows_geometry(M, K)[2]
        P = similar(C, Tacc, (S1 + matvec_cols_geometry(M, K)[3], M))
        launch_matvec_rows!(C, P, Tacc, A, x, add, M, K, 0, cx, upper ? :upper : :lower, herm, false)
        launch_matvec_cols!(C, P, Tacc, A, x, add, M, K, S1, cx, upper ? :above : :below, herm, false)
    end
    launch_matvec_cols!(C, nothing, Tacc, P, nothing, add, M, size(P, 1), 0, 'N', :all, false, true)
    C
end
