# broadcasting operations

export AbstractGPUArrayStyle

using Base.Broadcast

import Base.Broadcast: BroadcastStyle, Broadcasted, AbstractArrayStyle

"""
Abstract supertype for GPU array styles. The `N` parameter is the dimensionality.

Downstream implementations should provide a concrete array style type that inherits from
this supertype.
"""
abstract type AbstractGPUArrayStyle{N} <: AbstractArrayStyle{N} end

# Wrapper types otherwise forget that they are GPU compatible
# NOTE: don't directly use GPUArrayStyle here not to lose downstream customizations.
for (W, ctor) in Adapt.wrappers
  @eval begin
    BroadcastStyle(::Type{<:$W}) where {AT<:AbstractGPUArray} = BroadcastStyle(AT)
    backend(::Type{<:$W}) where {AT<:AbstractGPUArray} = backend(AT)
  end
end

# This Union is a hack. Ideally Base would have a Transpose <: WrappedArray <: AbstractArray
# and we could define our methods in terms of Union{AbstractGPUArray, WrappedArray{<:Any, <:AbstractGPUArray}}
@eval const GPUDestArray =
  Union{AbstractGPUArray,
        $((:($W where {AT <: AbstractGPUArray}) for (W, _) in Adapt.wrappers)...),
        Base.RefValue{<:AbstractGPUArray} }

# Ref is special: it's not a real wrapper, so not part of Adapt,
# but it is commonly used to bypass broadcasting of an argument
# so we need to preserve its dimensionless properties.
BroadcastStyle(::Type{Base.RefValue{AT}}) where {AT<:AbstractGPUArray} = typeof(BroadcastStyle(AT))(Val(0))
backend(::Type{Base.RefValue{AT}}) where {AT<:AbstractGPUArray} = backend(AT)
# but make sure we don't dispatch to the optimized copy method that directly indexes
function Broadcast.copy(bc::Broadcasted{<:AbstractGPUArrayStyle{0}})
    ElType = Broadcast.combine_eltypes(bc.f, bc.args)
    isbitstype(ElType) || error("Cannot broadcast function returning non-isbits $ElType.")
    dest = copyto!(similar(bc, ElType), bc)
    return @allowscalar dest[CartesianIndex()]  # 0D broadcast needs to unwrap results
end

# We purposefully only specialize `copyto!`, dependent packages need to make sure that they
# can handle:
# - `bc::Broadcast.Broadcasted{Style}`
# - `ex::Broadcast.Extruded`
# - `LinearAlgebra.Transpose{,<:AbstractGPUArray}` and `LinearAlgebra.Adjoint{,<:AbstractGPUArray}`, etc
#    as arguments to a kernel and that they do the right conversion.
#
# This Broadcast can be further customize by:
# - `Broadcast.preprocess(dest::AbstractGPUArray, bc::Broadcasted{Nothing})` which allows for a
#   complete transformation based on the output type just at the end of the pipeline.
# - `Broadcast.broadcasted(::Style, f)` selection of an implementation of `f` compatible
#   with `Style`
#
# For more information see the Base documentation.
@inline function Base.copyto!(dest::GPUDestArray, bc::Broadcasted{Nothing})
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    bc′ = Broadcast.preprocess(dest, bc)
    gpu_call(dest, bc′; name="broadcast") do ctx, dest, bc′
        let I = @cartesianidx(dest)
            @inbounds dest[I] = bc′[I]
        end
        return
    end

    return dest
end

# Base defines this method as a performance optimization, but we don't know how to do
# `fill!` in general for all `GPUDestArray` so we just go straight to the fallback
@inline Base.copyto!(dest::GPUDestArray, bc::Broadcasted{<:Broadcast.AbstractArrayStyle{0}}) =
    copyto!(dest, convert(Broadcasted{Nothing}, bc))


## map

allequal(x) = true
allequal(x, y, z...) = x == y && allequal(y, z...)

function Base.map!(f, dest::GPUDestArray, xs::AbstractArray...)
    indices = LinearIndices.((dest, xs...))
    common_length = minimum(length.(indices))

    # custom broadcast, ignoring the container size mismatches
    # (avoids the reshape + view that our mapreduce impl has to do)
    bc = Broadcast.instantiate(Broadcast.broadcasted(f, xs...))
    bc′ = Broadcast.preprocess(dest, bc)
    gpu_call(dest, bc′; name="map!", total_threads=common_length) do ctx, dest, bc′
        i = linear_index(ctx)
        if i <= common_length
            I = CartesianIndices(axes(bc′))[i]
            @inbounds dest[i] = bc′[I]
        end
        return
    end

    return dest
end

function Base.map(f, x::GPUDestArray, xs::AbstractArray...)
    # if argument sizes match, their shape needs to be preserved
    xs = (x, xs...)
    if allequal(size.(xs)...)
         return f.(xs...)
    end

    # if not, treat them as iterators
    indices = LinearIndices.(xs)
    common_length = minimum(length.(indices))

    # construct a broadcast to figure out the destination container
    bc = Broadcast.instantiate(Broadcast.broadcasted(f, xs...))
    ElType = Broadcast.combine_eltypes(bc.f, bc.args)
    isbitstype(ElType) || error("Cannot map function returning non-isbits $ElType.")
    dest = similar(bc, ElType, common_length)

    return map!(f, dest, xs...)
end
