# map-reduce

const AbstractArrayOrBroadcasted = Union{AbstractArray,Broadcast.Broadcasted}

# GPUArrays' mapreduce methods build on `Base.mapreducedim!`, but with an additional
# argument `init` value to avoid eager initialization of `R` (if set to something).
mapreducedim!(f, op, R::AbstractGPUArray, A::AbstractArrayOrBroadcasted;
              init=nothing) = error("Not implemented") # COV_EXCL_LINE
# resolve ambiguities
Base.mapreducedim!(f, op, R::AbstractGPUArray, A::AbstractArray) = mapreducedim!(f, op, R, A)
Base.mapreducedim!(f, op, R::AbstractGPUArray, A::Broadcast.Broadcasted) = mapreducedim!(f, op, R, A)

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

# resolve ambiguities
Base.mapreduce(f, op, A::AbstractGPUArray, As::AbstractArrayOrBroadcasted...;
               dims=:, init=nothing) = _mapreduce(f, op, A, As...; dims=dims, init=init)
Base.mapreduce(f, op, A::Broadcast.Broadcasted{<:AbstractGPUArrayStyle}, As::AbstractArrayOrBroadcasted...;
               dims=:, init=nothing) = _mapreduce(f, op, A, As...; dims=dims, init=init)

function _mapreduce(f, op, As...; dims, init)
    # mapreduce should apply `f` like `map` does, consuming elements like iterators.
    bc = if allequal(size.(As)...)
        Broadcast.instantiate(Broadcast.broadcasted(f, As...))
    else
        # TODO: can we avoid the reshape + view?
        indices = LinearIndices.(As)
        common_length = minimum(length.(indices))
        Bs = map(As) do A
            view(reshape(A, length(A)), 1:common_length)
        end
        Broadcast.instantiate(Broadcast.broadcasted(f, Bs...))
    end

    # figure out the destination container type by looking at the initializer element,
    # or by relying on inference to reason through the map and reduce functions.
    if init === nothing
        ET = Broadcast.combine_eltypes(bc.f, bc.args)
        ET = Base.promote_op(op, ET, ET)
        (ET === Union{} || ET === Any) &&
            error("mapreduce cannot figure the output element type, please pass an explicit init value")

        init = neutral_element(op, ET)
    else
        ET = typeof(init)
    end

    sz = size(bc)
    red = ntuple(i->(dims==Colon() || i in dims) ? 1 : sz[i], length(sz))
    R = similar(bc, ET, red)
    mapreducedim!(identity, op, R, bc; init=init)

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

# avoid calling into `initarray!`
Base.sum!(R::AbstractGPUArray, A::AbstractGPUArray) = Base.reducedim!(Base.add_sum, R, A)
Base.prod!(R::AbstractGPUArray, A::AbstractGPUArray) = Base.reducedim!(Base.mul_prod, R, A)
Base.maximum!(R::AbstractGPUArray, A::AbstractGPUArray) = Base.reducedim!(max, R, A)
Base.minimum!(R::AbstractGPUArray, A::AbstractGPUArray) = Base.reducedim!(min, R, A)

LinearAlgebra.ishermitian(A::AbstractGPUMatrix) = mapreduce(==, &, A, adjoint(A))
