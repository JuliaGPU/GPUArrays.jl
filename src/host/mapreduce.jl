# map-reduce

const AbstractArrayOrBroadcasted = Union{AbstractArray,Broadcast.Broadcasted}

# GPUArrays' mapreduce methods build on `Base.mapreducedim!`, but with an additional
# argument `init` value to avoid eager initialization of `R` (if set to something).
mapreducedim!(f, op, R::AnyGPUArray, A::AbstractArrayOrBroadcasted;
              init=nothing) = error("Not implemented") # COV_EXCL_LINE
# resolve ambiguities
Base.mapreducedim!(f, op, R::AnyGPUArray, A::AbstractArray) = mapreducedim!(f, op, R, A)
Base.mapreducedim!(f, op, R::AnyGPUArray, A::Broadcast.Broadcasted) = mapreducedim!(f, op, R, A)

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
if VERSION >= v"1.8.0-DEV.1465"
    neutral_element(::typeof(Base._extrema_rf), ::Type{<:NTuple{2,T}}) where {T} = typemax(T), typemin(T)
end

# resolve ambiguities
Base.mapreduce(f, op, A::AnyGPUArray, As::AbstractArrayOrBroadcasted...;
               dims=:, init=nothing) = _mapreduce(f, op, A, As...; dims=dims, init=init)
Base.mapreduce(f, op, A::Broadcast.Broadcasted{<:AbstractGPUArrayStyle}, As::AbstractArrayOrBroadcasted...;
               dims=:, init=nothing) = _mapreduce(f, op, A, As...; dims=dims, init=init)

function _mapreduce(f::F, op::OP, As::Vararg{Any,N}; dims::D, init) where {F,OP,N,D}
    # mapreduce should apply `f` like `map` does, consuming elements like iterators
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
    # or by relying on inference to reason through the map and reduce functions
    if init === nothing
        ET = Broadcast.combine_eltypes(f, As)
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

    if dims === Colon()
        @allowscalar R[]
    else
        R
    end
end

Base.any(A::AnyGPUArray{Bool}) = mapreduce(identity, |, A)
Base.all(A::AnyGPUArray{Bool}) = mapreduce(identity, &, A)

Base.any(f::Function, A::AnyGPUArray) = mapreduce(f, |, A)
Base.all(f::Function, A::AnyGPUArray) = mapreduce(f, &, A)

Base.count(pred::Function, A::AnyGPUArray; dims=:, init=0) =
    mapreduce(pred, Base.add_sum, A; init=init, dims=dims)

Base.:(==)(A::AnyGPUArray, B::AnyGPUArray) = Bool(mapreduce(==, &, A, B))

# avoid calling into `initarray!`
for (fname, op) in [(:sum, :(Base.add_sum)), (:prod, :(Base.mul_prod)),
                    (:maximum, :(Base.max)), (:minimum, :(Base.min)),
                    (:all, :&),              (:any, :|)]
    fname! = Symbol(fname, '!')
    @eval begin
        Base.$(fname!)(f::Function, r::AnyGPUArray, A::AnyGPUArray{T}) where T =
            GPUArrays.mapreducedim!(f, $(op), r, A; init=neutral_element($(op), T))
    end
end

LinearAlgebra.ishermitian(A::AbstractGPUMatrix) = mapreduce(==, &, A, adjoint(A))
