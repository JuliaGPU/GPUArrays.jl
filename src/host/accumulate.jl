# scans, implemented by AcceleratedKernels
#
# Base's `accumulate!` and `accumulate` reach these four methods, split as Base's are. `init` is
# passed on only when given (`Some(x)`), so that an explicit `init=nothing` stays an initial
# value. Base's rules are kept here: without `dims`, only vectors are scanned, and a `dims` beyond
# the array's copies it, ignoring `init` (AcceleratedKernels would apply `init` to every element).
# `B`'s element type, which Base's front-end chose, sets the running values' type in
# AcceleratedKernels.

Base._accumulate!(op, B::AnyGPUArray, A::AnyGPUVector, dims::Nothing, init::Nothing) =
    AK.accumulate!(op, B, A)
Base._accumulate!(op, B::AnyGPUArray, A::AnyGPUVector, dims::Nothing, init::Some) =
    AK.accumulate!(op, B, A; init=something(init))
function Base._accumulate!(op, B::AnyGPUArray, A::AnyGPUArray, dims::Integer, init::Nothing)
    dims > 0 || throw(ArgumentError("dims must be a positive integer"))
    axes(B) == axes(A) || throw(DimensionMismatch("shape of B must match A"))
    dims > ndims(A) && return copyto!(B, A)
    AK.accumulate!(op, B, A; dims)
end
function Base._accumulate!(op, B::AnyGPUArray, A::AnyGPUArray, dims::Integer, init::Some)
    dims > 0 || throw(ArgumentError("dims must be a positive integer"))
    axes(B) == axes(A) || throw(DimensionMismatch("shape of B must match A"))
    dims > ndims(A) && return copyto!(B, A)
    AK.accumulate!(op, B, A; dims, init=something(init))
end

# (`cumsum!` of floating-point vectors)
Base.accumulate_pairwise!(op, B::AnyGPUVector, A::AnyGPUVector) = accumulate!(op, B, A)

# Without `dims`, other arrays are scanned in linear order, keeping their shape
function Base.accumulate(op, A::AnyGPUArray; dims::Union{Nothing,Integer}=nothing, kw...)
    if dims === nothing && !(A isa AbstractVector)
        return reshape(accumulate(op, vec(A); kw...), size(A))
    end
    return invoke(accumulate, Tuple{Any, Any}, op, A; dims, kw...)
end
