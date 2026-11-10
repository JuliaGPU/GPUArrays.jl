# BLAS hooks
#
# GPUArrays implements LinearAlgebra's storage-level matrix products and triangular
# operations for all GPU arrays. It handles correctness checks then decides which BLAS
# routine an operation corresponds to. We define generic `blas_*!` functions which
# provide generic fallbacks for GPU arrays with no vendor BLAS to call into, but can
# be overriden to use a high-performance vendor implementation, as shown below:
#
#     GPUArrays.blas_gemm!(tA, tB, α::Union{Bool,T}, A::StridedROCVecOrMat{T},
#                          B::StridedROCVecOrMat{T}, β::Union{Bool,T},
#                          C::StridedROCVecOrMat{T}) where {T<:ROCBLASFloat} =
#         rocBLAS.gemm!(tA, tB, T(α), A, B, T(β), C)
#
# If the types in the call don't match any override, we fall back onto the generic kernel.
#
# The hooks are only called with operands whose layout BLAS can address (see `blas_layout`),
# and that do not alias *unless* the function works in place.

blas_layout(x::AbstractVector) = length(x) <= 1 || stride(x, 1) > 0
blas_layout(A::AbstractMatrix) = blas_matrix_layout(A)
function blas_matrix_layout(A::AbstractVecOrMat)
    m = size(A, 1)
    (stride(A, 1) == 1 || m <= 1) && (m == 0 || stride(A, 2) >= m)
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


## BLAS 2

blas_gemv!(tA::AbstractChar, α, A, x, β, y) =
    if blas_matrix_layout(x) && blas_matrix_layout(y)
        blas_gemm!(tA, 'N', α, A, x, β, y)
    else
        generic_matmatmul!(y, tA, 'N', A, x, α, β)
    end

blas_symv!(uplo::AbstractChar, α, A, x, β, y) =
    generic_matmatmul!(y, symm_char(uplo), 'N', A, x, α, β)

blas_hemv!(uplo::AbstractChar, α, A, x, β, y) =
    generic_matmatmul!(y, hemm_char(uplo), 'N', A, x, α, β)

blas_trmv!(uplo::AbstractChar, trans::AbstractChar, diag::AbstractChar, A, x) =
    # the kernel reads other rows of its input than the one it writes
    generic_trimatmul!(x, uplo, diag, trans_op(trans), A, copy(x))

blas_trsv!(uplo::AbstractChar, trans::AbstractChar, diag::AbstractChar, A, x) =
    generic_trimatdiv!(x, uplo, diag, trans_op(trans), A, x)


## BLAS 3

blas_gemm!(tA::AbstractChar, tB::AbstractChar, α, A, B, β, C) =
    generic_matmatmul!(C, tA, tB, A, B, α, β)

blas_symm!(side::AbstractChar, uplo::AbstractChar, α, A, B, β, C) =
    side == 'L' ? generic_matmatmul!(C, symm_char(uplo), 'N', A, B, α, β) :
                  generic_matmatmul!(C, 'N', symm_char(uplo), B, A, α, β)

blas_hemm!(side::AbstractChar, uplo::AbstractChar, α, A, B, β, C) =
    side == 'L' ? generic_matmatmul!(C, hemm_char(uplo), 'N', A, B, α, β) :
                  generic_matmatmul!(C, 'N', hemm_char(uplo), B, A, α, β)

function blas_trmm!(side::AbstractChar, uplo::AbstractChar, trans::AbstractChar,
                    diag::AbstractChar, α, A, B, C)
    B′ = C === B ? copy(B) : B
    if side == 'L'
        generic_trimatmul!(C, uplo, diag, trans_op(trans), A, B′)
    else
        generic_mattrimul!(C, uplo, diag, trans_op(trans), B′, A)
    end
    isone(α) || rmul!(C, α)
    return C
end

function blas_trsm!(side::AbstractChar, uplo::AbstractChar, trans::AbstractChar,
                    diag::AbstractChar, α, A, B)
    isone(α) || rmul!(B, α)
    if side == 'L'
        generic_trimatdiv!(B, uplo, diag, trans_op(trans), A, B)
    else
        generic_mattridiv!(B, uplo, diag, trans_op(trans), B, A)
    end
end


## BLAS 1

blas_scal!(α, x) = generic_rmul!(x, α)

# dot conjugates x, dotu doesn't
blas_dot(x, y) = mapreduce(dot, +, x, y)
blas_dotu(x, y) = mapreduce(*, +, x, y)

blas_nrm2(x) = generic_norm(x, 2)

blas_asum(x) = mapreduce(z -> abs(real(z)) + abs(imag(z)), +, x; init=zero(real(eltype(x))))

blas_axpy!(α, x, y) = y .+= x .* α
blas_axpby!(α, x, β, y) = y .= x .* α .+ y .* β

blas_rot!(x, y, c, s) = generic_rotate!(x, y, c, s)


## BLAS 3 extensions

blas_syrk!(uplo::AbstractChar, trans::AbstractChar, α, A, β, C) =
    generic_matmatmul!(C, trans, trans == 'N' ? 'T' : 'N', A, A, α, β)
blas_herk!(uplo::AbstractChar, trans::AbstractChar, α, A, β, C) =
    generic_matmatmul!(C, trans, trans == 'N' ? 'C' : 'N', A, A, α, β)

# C = α op(A) + β op(B)
function blas_geam!(tA::AbstractChar, tB::AbstractChar, α, A, β, B, C)
    opA, opB = trans_op(tA), trans_op(tB)
    if β === false
        α === true && opA !== identity ? transpose_f!(opA, C, A) : (C .= α .* opA(A))
    else
        C .= α .* opA(A) .+ β .* opB(B)
    end
    return C
end

# C = Diagonal(x) * A or A * Diagonal(x)
blas_dgmm!(side::AbstractChar, A, x, C) =
    side == 'L' ? (C .= x .* A) : (C .= A .* transpose(x))
