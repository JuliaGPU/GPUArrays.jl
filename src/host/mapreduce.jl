# map-reduce

# GPUArrays' mapreduce methods build on `Base.mapreducedim!`, but with an additional
# argument `init` value to avoid eager initialization of `R` (if set to something).
mapreducedim!(f, op, R::AbstractGPUArray, A::AbstractArray, init=nothing) = error("Not implemented") # COV_EXCL_LINE
Base.mapreducedim!(f, op, R::AbstractGPUArray, A::AbstractArray) = mapreducedim!(f, op, R, A)

neutral_element(op, T) =
    error("""GPUArrays.jl needs to know the neutral element for your operator `$op`.
             Please pass it as an explicit argument to (if possible), or register it
             globally your operator by defining `GPUArrays.neutral_element(::typeof($op), T)`.""")
neutral_element(::typeof(Base.:(|)), T) = zero(T)
neutral_element(::typeof(Base.:(+)), T) = zero(T)
neutral_element(::typeof(Base.add_sum), T) = zero(T)
neutral_element(::typeof(Base.:(&)), T) = one(T)
neutral_element(::typeof(Base.:(*)), T) = one(T)
neutral_element(::typeof(Base.mul_prod), T) = one(T)
neutral_element(::typeof(Base.min), T) = typemax(T)
neutral_element(::typeof(Base.max), T) = typemin(T)

function Base.mapreduce(f, op, A::AbstractGPUArray; dims=:, init=nothing)
    # figure out the destination container type by looking at the initializer element,
    # or by relying on inference to reason through the map and reduce functions.
    if init === nothing
        ET = Base.promote_op(f, eltype(A))
        ET = Base.promote_op(op, ET, ET)
        (ET === Union{} || ET === Any) &&
            error("mapreduce cannot figure the output element type, please pass an explicit init value")

        init = neutral_element(op, ET)
    else
        ET = typeof(init)
    end

    sz = size(A)
    red = ntuple(i->(dims==Colon() || i in dims) ? 1 : sz[i], ndims(A))
    R = similar(A, ET, red)
    mapreducedim!(f, op, R, A, init)

    if dims==Colon()
        @allowscalar R[]
    else
        R
    end
end

Base.any(A::AbstractGPUArray{Bool}) = mapreduce(identity, |, A)
Base.all(A::AbstractGPUArray{Bool}) = mapreduce(identity, &, A)

Base.any(f::Function, A::AbstractGPUArray) = mapreduce(f, |, A)
Base.all(f::Function, A::AbstractGPUArray) = mapreduce(f, &, A)
Base.count(pred::Function, A::AbstractGPUArray) = mapreduce(pred, +, A; init = 0)

Base.:(==)(A::AbstractGPUArray, B::AbstractGPUArray) = Bool(mapreduce(==, &, A, B))

# avoid calling into `initarray!``
Base.sum!(R::AbstractGPUArray, A::AbstractGPUArray) = Base.reducedim!(Base.add_sum, R, A)
Base.prod!(R::AbstractGPUArray, A::AbstractGPUArray) = Base.reducedim!(Base.mul_prod, R, A)
Base.maximum!(R::AbstractGPUArray, A::AbstractGPUArray) = Base.reducedim!(max, R, A)
Base.minimum!(R::AbstractGPUArray, A::AbstractGPUArray) = Base.reducedim!(min, R, A)

LinearAlgebra.ishermitian(A::AbstractGPUMatrix) = mapreduce(==, &, A, adjoint(A))
