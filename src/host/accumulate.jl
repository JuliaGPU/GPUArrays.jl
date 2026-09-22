# accumulate

# GPUArrays' accumulate / cumsum / cumprod forward to AcceleratedKernels, the same direct-delegation
# approach as sort and reverse. Hooking `Base._accumulate!` covers `accumulate!`, `cumsum!` and
# `cumprod!`; the `Base.accumulate` method below adds the out-of-place entry and the whole-array
# (no `dims`) case that `_accumulate!` does not reach. AK owns the scan implementation for every
# backend, so there is no per-backend kernel here.

Base._accumulate!(op, output::AnyGPUArray, input::AnyGPUVector, dims::Nothing, init::Nothing) =
    AK.accumulate!(op, output, input, get_backend(output); dims,
                   init=AK.neutral_element(op, eltype(output)))

Base._accumulate!(op, output::AnyGPUArray, input::AnyGPUArray, dims::Integer, init::Nothing) =
    AK.accumulate!(op, output, input, get_backend(output); dims,
                   init=AK.neutral_element(op, eltype(output)))

Base._accumulate!(op, output::AnyGPUArray, input::AnyGPUVector, dims::Nothing, init::Some) =
    AK.accumulate!(op, output, input, get_backend(output); dims, init=something(init))

Base._accumulate!(op, output::AnyGPUArray, input::AnyGPUArray, dims::Integer, init::Some) =
    AK.accumulate!(op, output, input, get_backend(output); dims, init=something(init))

Base.accumulate_pairwise!(op, result::AnyGPUVector, v::AnyGPUVector) = accumulate!(op, result, v)

# out-of-place; also handles the whole-array (no `dims`) case for N-D inputs
function Base.accumulate(op, A::AnyGPUArray; dims::Union{Nothing,Integer}=nothing, kw...)
    nt = values(kw)
    if dims === nothing && !(A isa AbstractVector)
        # linearize the array and scan it as one vector, then restore the shape
        init = :init in keys(kw) ? nt.init : AK.neutral_element(op, eltype(A))
        return reshape(AK.accumulate(op, A[:], get_backend(A); init), size(A))
    end
    if isempty(kw)
        out = similar(A, Base.promote_op(op, eltype(A), eltype(A)))
        init = AK.neutral_element(op, eltype(out))
    elseif keys(nt) === (:init,)
        out = similar(A, Base.promote_op(op, typeof(nt.init), eltype(A)))
        init = nt.init
    else
        throw(ArgumentError("accumulate does not support the keyword arguments $(setdiff(keys(nt), (:init,)))"))
    end
    AK.accumulate!(op, out, A, get_backend(A); dims, init)
end
