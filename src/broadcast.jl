using Base.Broadcast

import Base.Broadcast: BroadcastStyle, Broadcasted, ArrayStyle

# we define a generic `BroadcastStyle` here that should be sufficient for most cases.
# dependent packages like `CuArrays` can define their own `BroadcastStyle` allowing
# them to further change or optimize broadcasting.
#
# TODO: investigate if we should define out own `GPUArrayStyle{N} <: AbstractArrayStyle{N}`
#
# NOTE: this uses the specific `T` that was used e.g. `JLArray` or `CLArray` for ArrayStyle,
#       instead of using `ArrayStyle{GPUArray}`, due to the fact how `similar` works.
BroadcastStyle(::Type{T}) where {T<:GPUArray} = ArrayStyle{T}()

# These wrapper types otherwise forget that they are GPU compatible
#
# NOTE: Don't directly use ArrayStyle{GPUArray} here since that would mean that `CuArrays`
#       customization no longer take effect.
BroadcastStyle(::Type{<:LinearAlgebra.Transpose{<:Any,T}}) where {T<:GPUArray} = BroadcastStyle(T)
BroadcastStyle(::Type{<:LinearAlgebra.Adjoint{<:Any,T}}) where {T<:GPUArray} = BroadcastStyle(T)
BroadcastStyle(::Type{<:SubArray{<:Any,<:Any,T}}) where {T<:GPUArray} = BroadcastStyle(T)

# This Union is a hack. Ideally Base would have a Transpose <: WrappedArray <: AbstractArray
# and we could define our methods in terms of Union{GPUArray, WrappedArray{<:Any, <:GPUArray}}
const GPUDestArray = Union{GPUArray,
                           LinearAlgebra.Transpose{<:Any,<:GPUArray},
                           LinearAlgebra.Adjoint{<:Any,<:GPUArray},
                           SubArray{<:Any,<:Any,<:GPUArray}}

# This method is responsible for selection the output type of broadcast
function Base.similar(bc::Broadcasted{<:ArrayStyle{GPU}}, ::Type{ElType}) where
                     {GPU <: GPUArray, ElType}
    similar(GPU, ElType, axes(bc))
end

# We purposefully only specialize `copyto!`, dependent packages need to make sure that they
# can handle:
# - `bc::Broadcast.Broadcasted{Style}`
# - `ex::Broadcast.Extruded`
# - `LinearAlgebra.Transpose{,<:GPUArray}` and `LinearAlgebra.Adjoint{,<:GPUArray}`, etc
#    as arguments to a kernel and that they do the right conversion.
#
# This Broadcast can be further customize by:
# - `Broadcast.preprocess(dest::GPUArray, bc::Broadcasted{Nothing})` which allows for a
#   complete transformation based on the output type just at the end of the pipeline.
# - `Broadcast.broadcasted(::Style, f)` selection of an implementation of `f` compatible
#   with `Style`
#
# For more information see the Base documentation.
@inline function Base.copyto!(dest::GPUDestArray, bc::Broadcasted{Nothing})
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    bc′ = Broadcast.preprocess(dest, bc)
    gpu_call(dest, (dest, bc′)) do state, dest, bc′
        let I = CartesianIndex(@cartesianidx(dest))
            @inbounds dest[I] = bc′[I]
        end
    end

    return dest
end

# Base defines this method as a performance optimization, but we don't know how to do
# `fill!` in general for all `GPUDestArray` so we just go straight to the fallback
@inline Base.copyto!(dest::GPUDestArray, bc::Broadcasted{<:Broadcast.AbstractArrayStyle{0}}) =
    copyto!(dest, convert(Broadcasted{Nothing}, bc))

# TODO: is this still necessary?
function mapidx(f, A::GPUArray, args::NTuple{N, Any}) where N
    gpu_call(A, (f, A, args)) do state, f, A, args
        ilin = @linearidx(A, state)
        f(ilin, A, args...)
    end
end
