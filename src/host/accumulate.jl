## Base interface

Base._accumulate!(op, output::AnyGPUArray, input::AnyGPUVector, dims::Nothing, init::Nothing) =
    AK.accumulate!(op, output, input, get_backend(output); dims, init=AK.neutral_element(op, eltype(output)))

Base._accumulate!(op, output::AnyGPUArray, input::AnyGPUArray, dims::Integer, init::Nothing) =
    AK.accumulate!(op, output, input, get_backend(output); dims, init=AK.neutral_element(op, eltype(output)))

Base._accumulate!(op, output::AnyGPUArray, input::AnyGPUVector, dims::Nothing, init::Some) =
    AK.accumulate!(op, output, input, get_backend(output); dims, init=something(init))

Base._accumulate!(op, output::AnyGPUArray, input::AnyGPUArray, dims::Integer, init::Some) =
    AK.accumulate!(op, output, input, get_backend(output); dims, init=something(init))

Base.accumulate_pairwise!(op, result::AnyGPUVector, v::AnyGPUVector) = accumulate!(op, result, v)

# default behavior unless dims are specified by the user
function Base.accumulate(op, A::AnyGPUArray;
                         dims::Union{Nothing,Integer}=nothing, kw...)
    nt = values(kw)
    if dims === nothing && !(A isa AbstractVector)
        # This branch takes care of the cases not handled by `_accumulate!`.
        return reshape(AK.accumulate(op, A[:], get_backend(A); init = (:init in keys(kw) ? nt.init : AK.neutral_element(op, eltype(A)))), size(A))
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
