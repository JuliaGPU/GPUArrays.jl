# accumulate

# JLArrays are backed by a regular Array, so the accumulate primitives defer to Base on that storage.
# This keeps the reference backend independent of AcceleratedKernels' GPU kernels.

Base._accumulate!(op, output::AnyJLArray, input::AnyJLVector, dims::Nothing, init::Nothing) =
    accumulate!(op, typed_data(output), typed_data(input); dims=1)

Base._accumulate!(op, output::AnyJLArray, input::AnyJLArray, dims::Integer, init::Nothing) =
    accumulate!(op, typed_data(output), typed_data(input); dims)

Base._accumulate!(op, output::AnyJLArray, input::AnyJLVector, dims::Nothing, init::Some) =
    accumulate!(op, typed_data(output), typed_data(input); dims=1, init=something(init))

Base._accumulate!(op, output::AnyJLArray, input::AnyJLArray, dims::Integer, init::Some) =
    accumulate!(op, typed_data(output), typed_data(input); dims, init=something(init))

Base.accumulate_pairwise!(op, result::AnyJLVector, v::AnyJLVector) = accumulate!(op, result, v)

function Base.accumulate(op, A::AnyJLArray; dims::Union{Nothing,Integer}=nothing, kw...)
    nt = values(kw)
    if dims === nothing && !(A isa AbstractVector)
        return reshape(accumulate(op, typed_data(A)[:]; kw...), size(A))
    end
    isempty(kw) || keys(nt) === (:init,) ||
        throw(ArgumentError("accumulate does not support the keyword arguments $(setdiff(keys(nt), (:init,)))"))
    JLArray(accumulate(op, typed_data(A); dims = dims === nothing ? 1 : dims, kw...))
end
