# vendor BLAS hooks
#
# GPUArrays implements LinearAlgebra's matrix products and triangular operations for every
# GPU array: it checks dimensions, unwraps operands into BLAS characters, and decides which
# BLAS routine an operation corresponds to. It then calls one of the `blas_*` functions
# below, whose default methods use GPUArrays' own kernels, often by way of a BLAS 3 hook.
# Back-ends with a vendor BLAS add methods for the operand types and eltypes their library
# supports; the contract is documented in docs/src/interface.md.

# matrix operands of BLAS 2 and 3 routines
blas_layout(x::AbstractVector) = stride(x, 1) > 0
blas_layout(A::AbstractMatrix) = blas_matrix_layout(A)
blas_matrix_layout(A::AbstractVecOrMat) = stride(A, 1) == 1 && stride(A, 2) >= size(A, 1)

# like LinearAlgebra, promote scalars to the output eltype, unless they are of a wider type
blas_scalar(::Type, x::Bool) = x
function blas_scalar(::Type{T}, x::Number) where {T}
    y = first(promote(x, zero(T)))
    y isa T ? y : x
end
blas_scalars(::Type{T}, α, β) where {T} = α isa Union{Bool,T} && β isa Union{Bool,T}

trans_op(t::AbstractChar) = t == 'N' ? identity : t == 'T' ? transpose : adjoint
trans_char(f::Function) = f === identity ? 'N' : f === transpose ? 'T' : 'C'

symm_char(uplo::AbstractChar) = uplo == 'U' ? 'S' : 's'
hemm_char(uplo::AbstractChar) = uplo == 'U' ? 'H' : 'h'

@static if isdefined(LinearAlgebra, Symbol("@stable_muladdmul")) # added in 1.12
    generic_matmatmul!(C, tA::AbstractChar, tB::AbstractChar, A, B, α::Number, β::Number) =
        LinearAlgebra.@stable_muladdmul generic_matmatmul!(C, tA, tB, A, B, MulAddMul(α, β))
else
    generic_matmatmul!(C, tA::AbstractChar, tB::AbstractChar, A, B, α::Number, β::Number) =
        generic_matmatmul!(C, tA, tB, A, B, MulAddMul(α, β))
end


## BLAS 3

"""
    GPUArrays.blas_gemm!(tA, tB, α, A, B, β, C)

`C = α op(A) op(B) + β C`, like `LinearAlgebra.BLAS.gemm!`. The operands can be vectors, as
``n×1`` matrices, and are the only ones that can have different eltypes, while `α` and `β`
can be of yet another type if converting them to the eltype of `C` would lose precision.
"""
blas_gemm!(tA, tB, α, A, B, β, C) =
    generic_matmatmul!(C, tA, tB, A, B, α, β)

"""
    GPUArrays.blas_symm!(side, uplo, α, A, B, β, C)
    GPUArrays.blas_hemm!(side, uplo, α, A, B, β, C)

`C = α A B + β C` (`side == 'L'`) or `C = α B A + β C` (`side == 'R'`) with `A` symmetric
or Hermitian, stored in its `uplo` triangle, like `LinearAlgebra.BLAS.symm!`/`hemm!`. `B` and
`C` can be vectors.
"""
blas_symm!(side, uplo, α, A, B, β, C) =
    side == 'L' ? generic_matmatmul!(C, symm_char(uplo), 'N', A, B, α, β) :
                  generic_matmatmul!(C, 'N', symm_char(uplo), B, A, α, β)

@doc (@doc blas_symm!)
blas_hemm!(side, uplo, α, A, B, β, C) =
    side == 'L' ? generic_matmatmul!(C, hemm_char(uplo), 'N', A, B, α, β) :
                  generic_matmatmul!(C, 'N', hemm_char(uplo), B, A, α, β)

"""
    GPUArrays.blas_syrk!(uplo, trans, α, A, β, C)
    GPUArrays.blas_herk!(uplo, trans, α, A, β, C)

Update the `uplo` triangle of `C` with `α A Aᵀ + β C` (`trans == 'N'`) or `α Aᵀ A + β C`
(`trans == 'T'`), or the Hermitian `α A Aᴴ + β C` and `α Aᴴ A + β C` (`trans == 'C'`), like
`LinearAlgebra.BLAS.syrk!`/`herk!`. For `herk!`, `α` and `β` are real. The other triangle
of `C` is left unspecified.
"""
blas_syrk!(uplo, trans, α, A, β, C) =
    blas_gemm!(trans, trans == 'N' ? 'T' : 'N', α, A, A, β, C)

@doc (@doc blas_syrk!)
function blas_herk!(uplo, trans, α, A, β, C)
    T = eltype(C)
    # unlike gemm, herk ignores the imaginary part of the diagonal and produces a real one
    d = view(C, diagind(C))
    iszero(β) || (d .= real.(d))
    blas_gemm!(trans, trans == 'N' ? 'C' : 'N', blas_scalar(T, α), A, A, blas_scalar(T, β), C)
    d .= real.(d)
    return C
end

"""
    GPUArrays.blas_trmm!(side, uplo, trans, diag, α, A, B, C)

`C = α op(A) B` (`side == 'L'`) or `C = α B op(A)` (`side == 'R'`) with `A` triangular, stored
in its `uplo` triangle and with a unit diagonal if `diag == 'U'`. Unlike
`LinearAlgebra.BLAS.trmm!`, the result goes to `C`, which can be `B`. `B` and `C` can be
vectors.
"""
function blas_trmm!(side, uplo, trans,
                    diag, α, A, B, C)
    B′ = C === B ? copy(B) : B
    if side == 'L'
        generic_trimatmul!(C, uplo, diag, trans_op(trans), A, B′)
    else
        generic_mattrimul!(C, uplo, diag, trans_op(trans), B′, A)
    end
    isone(α) || generic_rmul!(C, α)
    return C
end

"""
    GPUArrays.blas_trsm!(side, uplo, trans, diag, α, A, B)

Overwrite `B` with `α op(A) \\ B` (`side == 'L'`) or `α B / op(A)` (`side == 'R'`) with `A`
triangular, like `LinearAlgebra.BLAS.trsm!`. `B` can be a vector. A singular `A` yields Infs
and NaNs, as with vendor libraries, rather than an exception.
"""
function blas_trsm!(side, uplo, trans,
                    diag, α, A, B)
    isone(α) || generic_rmul!(B, α)
    if side == 'L'
        generic_trimatdiv!(B, uplo, diag, trans_op(trans), A, B)
    else
        generic_mattridiv!(B, uplo, diag, trans_op(trans), B, A)
    end
end


## BLAS 2

"""
    GPUArrays.blas_gemv!(tA, α, A, x, β, y)

`y = α op(A) x + β y`, like `LinearAlgebra.BLAS.gemv!`, with the same freedom in eltypes as
[`GPUArrays.blas_gemm!`](@ref), which the default calls.
"""
blas_gemv!(tA, α, A, x, β, y) =
    if blas_matrix_layout(x) && blas_matrix_layout(y)
        blas_gemm!(tA, 'N', α, A, x, β, y)
    else
        generic_matmatmul!(y, tA, 'N', A, x, α, β)
    end

"""
    GPUArrays.blas_symv!(uplo, α, A, x, β, y)
    GPUArrays.blas_hemv!(uplo, α, A, x, β, y)

`y = α A x + β y` with `A` symmetric or Hermitian, like `LinearAlgebra.BLAS.symv!`/`hemv!`.
The defaults call [`GPUArrays.blas_symm!`](@ref) and [`GPUArrays.blas_hemm!`](@ref GPUArrays.blas_symm!).
"""
blas_symv!(uplo, α, A, x, β, y) =
    if blas_matrix_layout(x) && blas_matrix_layout(y)
        blas_symm!('L', uplo, α, A, x, β, y)
    else
        generic_matmatmul!(y, symm_char(uplo), 'N', A, x, α, β)
    end

@doc (@doc blas_symv!)
blas_hemv!(uplo, α, A, x, β, y) =
    if blas_matrix_layout(x) && blas_matrix_layout(y)
        blas_hemm!('L', uplo, α, A, x, β, y)
    else
        generic_matmatmul!(y, hemm_char(uplo), 'N', A, x, α, β)
    end

"""
    GPUArrays.blas_trmv!(uplo, trans, diag, A, x)
    GPUArrays.blas_trsv!(uplo, trans, diag, A, x)

Overwrite `x` with `op(A) x` or `op(A) \\ x` with `A` triangular, like
`LinearAlgebra.BLAS.trmv!`/`trsv!`. The defaults call [`GPUArrays.blas_trmm!`](@ref) and
[`GPUArrays.blas_trsm!`](@ref).
"""
blas_trmv!(uplo, trans, diag, A, x) =
    if blas_matrix_layout(x)
        blas_trmm!('L', uplo, trans, diag, true, A, x, x)
    else
        # the kernel reads other rows of its input than the one it writes
        generic_trimatmul!(x, uplo, diag, trans_op(trans), A, copy(x))
    end

@doc (@doc blas_trmv!)
blas_trsv!(uplo, trans, diag, A, x) =
    if blas_matrix_layout(x)
        blas_trsm!('L', uplo, trans, diag, true, A, x)
    else
        generic_trimatdiv!(x, uplo, diag, trans_op(trans), A, x)
    end
