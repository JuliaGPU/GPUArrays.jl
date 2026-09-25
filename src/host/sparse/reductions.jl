# reductions

# `op` folded over `n ≥ 1` copies of `x`, as the implicit zeros of a sparse array contribute
# to a reduction: by repeated squaring for an associative `op` (e.g. `n * x` for `+`), and a
# single `x` for idempotent ones.
@inline function _fold_repeated(op, x, n::Integer)
    _isidempotent(op) && return x
    result = x
    power = x
    n -= one(n)
    while n > 0
        if isodd(n)
            result = op(result, power)
        end
        power = op(power, power)
        n >>= 1
    end
    return result
end
_isidempotent(op) = false
_isidempotent(::Union{typeof(max), typeof(min), typeof(|), typeof(&)}) = true

## COV_EXCL_START
# reduce the stored values `nzVal[first_ptr:last_ptr]` of a slice that also contains
# `nzeros` implicit zeros, and has at least one element. As in SparseArrays, the stored
# values are reduced first and the implicit zeros folded in afterwards, so `op` is assumed
# to be commutative as well as associative.
@inline function _reduce_slice(f, op, ::Type{T}, nzVal, first_ptr, last_ptr, nzeros) where {T}
    if first_ptr <= last_ptr
        val = convert(T, f(@inbounds nzVal[first_ptr]))
        for ptr in first_ptr+one(first_ptr):last_ptr
            val = convert(T, op(val, f(@inbounds nzVal[ptr])))
        end
        if nzeros > 0
            val = convert(T, op(val, _fold_repeated(op, convert(T, f(zero(eltype(nzVal)))), nzeros)))
        end
        val
    else
        _fold_repeated(op, convert(T, f(zero(eltype(nzVal)))), nzeros)
    end
end

# every thread reduces one slice (a row of a CSR matrix, a column of a CSC one), which has
# `n` elements
@kernel function compressed_reduce_kernel(f::F, op::OP, output::AbstractArray{T}, ptr, nzVal,
                                          n) where {F, OP, T}
    j = @index(Global, Linear)
    if j ≤ length(output)
        first_ptr = @inbounds ptr[j]
        last_ptr = @inbounds ptr[j+1] - one(first_ptr)
        nzeros = n - (last_ptr - first_ptr + 1)
        @inbounds output[j] = _reduce_slice(f, op, T, nzVal, first_ptr, last_ptr, nzeros)
    end
end
## COV_EXCL_STOP

# the element type of a reduction, relying on inference to reason through the map and
# reduce functions, or on the type of the initializer
function reduction_eltype(f, op, A, init)
    ET = Broadcast.combine_eltypes(f, (A,))
    ET = Base.promote_op(op, ET, ET)
    if init !== nothing
        ET = Base.promote_op(op, typeof(init), ET)
    end
    (ET === Union{} || ET === Any) &&
        error("mapreduce cannot figure the output element type, please pass an explicit init value")
    return ET
end

# every slice reduction `r` becomes `op(init, r)`; empty slices give `init`, or the
# reduction of an empty collection as in Base
empty_reduction(f, op, ::Type{Tv}, init) where {Tv} =
    init === nothing ? Base.mapreduce_empty(f, op, Tv) : init

# reduce the stored values, then fold in the implicit zeros
function reduce_all(f, op, A::GPUSparseArray, init)
    Tv = eltype(A)
    length(A) == 0 && return empty_reduction(f, op, Tv, init)
    ET = reduction_eltype(f, op, A, init)
    nzeros = length(A) - nnz(A)
    zeros = nzeros > 0 ? _fold_repeated(op, convert(ET, f(zero(Tv))), nzeros) : nothing
    if nnz(A) > 0
        stored = init === nothing ? mapreduce(f, op, nonzeros(A)) :
                                    mapreduce(f, op, nonzeros(A); init)
        return zeros === nothing ? stored : op(stored, zeros)
    else
        return init === nothing ? zeros : op(init, zeros)
    end
end

Base.mapreduce(f, op, x::GPUSparseVector; dims=:, init=nothing) =
    dims === Colon() ? reduce_all(f, op, x, init) :
                       error("only dims=: is supported for sparse vectors")

function Base.mapreduce(f, op, A::GPUSparseMatrix; dims=:, init=nothing)
    dims === Colon() && return reduce_all(f, op, A, init)
    dims in (1, 2) || error("only dims=:, dims=1 or dims=2 is supported")

    # reduce along the compressed dimension, regrouping the entries if necessary. an atomic
    # scatter could reduce along the other dimension directly, for `+` and friends.
    B = dims == 1 ? GPUSparseMatrixCSC(A) : GPUSparseMatrixCSR(A)
    ET = reduction_eltype(f, op, A, init)
    output = similar(nonzeros(B), ET, dims == 1 ? (1, size(B, 2)) : (size(B, 1), 1))
    isempty(output) && return output
    if size(B, dims) == 0
        return fill!(output, empty_reduction(f, op, eltype(A), init))
    end
    compressed_reduce_kernel(get_backend(B))(f, op, output, compressed_ptr(B), nonzeros(B),
                                             minor_dim(B); ndrange=length(output))
    if init !== nothing
        output .= op.(init, output)
    end
    return output
end

# `nnz(A) == 0` means nothing is stored, otherwise check the stored values
Base.iszero(A::GPUSparseArray) = nnz(A) == 0 || all(iszero, nonzeros(A))

# the norm of the stored values, as in SparseArrays (which makes the -Inf norm the smallest
# stored magnitude)
function LinearAlgebra.norm(A::GPUSparseArray, p::Real=2)
    if p == Inf
        return maximum(abs, nonzeros(A); init=zero(real(eltype(A))))
    elseif p == -Inf
        return nnz(A) == 0 ? zero(real(eltype(A))) : minimum(abs, nonzeros(A))
    elseif p == 0
        return float(real(eltype(A)))(count(!iszero, nonzeros(A)))
    else
        return sum(x -> abs(x)^p, nonzeros(A); init=zero(real(eltype(A))))^(1/p)
    end
end

function LinearAlgebra.opnorm(A::GPUSparseMatrix, p::Real=2)
    if p == Inf
        return maximum(sum(abs, A; dims=2))
    elseif p == 1
        return maximum(sum(abs, A; dims=1))
    else
        throw(ArgumentError("p=$p is not supported"))
    end
end
