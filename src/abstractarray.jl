@compat abstract type AbstractAccArray{T, N} <: DenseArray{T, N} end
# Sampler type that acts like a texture/image and allows interpolated access
@compat abstract type AbstractSampler{T, N} <: AbstractAccArray{T, N} end

@compat const AccVector{T} = AbstractAccArray{T, 1}
@compat const AccMatrix{T} = AbstractAccArray{T, 2}
@compat const AccVecOrMat{T} = Union{AbstractAccArray{T, 1}, AbstractAccArray{T, 2}}



type GPUArray{T, N, B, C} <: AbstractAccArray{T, N}
    buffer::B
    size::NTuple{N, Int}
    context::C
end


# interfaces

#=
Interface for accessing the lower level
=#

buffer(A::AbstractAccArray) = A.buffer
context(A::AbstractAccArray) = A.context
default_buffer_type(typ, context) = error("Found unsupported context: $context")

# GPU Local Memory
immutable LocalMemory{T} <: AbstractAccArray{T, 1}
    size::Int
end


"""
Optimal linear index in a GPU kernel
"""
function linear_index end



#=
AbstractArray interface
=#
Base.eltype{T}(::AbstractAccArray{T}) = T
Base.size(A::AbstractAccArray) = A.size

function Base.show(io::IO, mt::MIME"text/plain", A::AbstractAccArray)
    println(io, "GPUArray with ctx: $(context(A)): ")
    show(io, mt, Array(A))
end
function Base.showcompact(io::IO, mt::MIME"text/plain", A::AbstractAccArray)
    showcompact(io, mt, Array(A))
end

function Base.similar{T <: AbstractAccArray}(x::T)
    similar(x, eltype(x), size(x))
end
function Base.similar{T <: AbstractAccArray, ET}(x::T, ::Type{ET}; kw_args...)
    similar(x, ET, size(x); kw_args...)
end
function Base.similar{T <: AbstractAccArray, N}(x::T, dims::NTuple{N, Int}; kw_args...)
    similar(x, eltype(x), dims; kw_args...)
end
function Base.similar{N, ET}(x::AbstractAccArray, ::Type{ET}, sz::NTuple{N, Int}; kw_args...)
    similar(typeof(x), ET, sz, context = context(x); kw_args...)
end


using Compat.TypeUtils
function Base.similar{T <: GPUArray, ET, N}(
        ::Type{T}, ::Type{ET}, sz::NTuple{N, Int};
        context::Context = current_context(), kw_args...
    )
    bt = default_buffer_type(T, Tuple{ET, N}, context)
    GPUArray{ET, N, bt, typeof(context)}(sz; context = context)
end






#=
Host to Device data transfers
=#
function (::Type{A}){A <: AbstractAccArray}(x::AbstractArray)
    A(collect(x))
end
function (::Type{A}){A <: AbstractAccArray}(x::Array; kw_args...)
    out = similar(A, eltype(x), size(x); kw_args...)
    copy!(out, x)
    out
end
Base.convert{A <: AbstractAccArray}(::Type{A}, x::AbstractArray) = A(x)
Base.convert{A <: AbstractAccArray}(::Type{A}, x::A) = x

#=
Device to host data transfers
=#
function (::Type{Array}){T, N}(device_array::AbstractAccArray{T, N})
    Array{T, N}(device_array)
end
function (AT::Type{Array{T, N}}){T, N}(device_array::AbstractAccArray)
    convert(AT, Array(device_array))
end
function (AT::Type{Array{T, N}}){T, N}(device_array::AbstractAccArray{T, N})
    hostarray = similar(AT, size(device_array))
    copy!(hostarray, device_array)
    hostarray
end


# Function needed to be overloaded by backends
function mapidx end
# It is kinda hard to overwrite map/broadcast, which is why we lift it to our
# our own broadcast function.
# It has the signature:
# f::Function, Context, Main/Out::AccArray, args::NTuple{N}
# All arrays are already lifted and shape checked
function acc_broadcast! end
# same for mapreduce
function acc_mapreduce end


######################################
# Broadcast

# helper

#Broadcast
Base.@propagate_inbounds broadcast_index(::Val{false}, arg, shape, i) = arg
Base.@propagate_inbounds function broadcast_index{T, N}(
        ::Val{true}, arg::AbstractArray{T, N}, shape::NTuple{N, Integer}, i
    )
    @inbounds return arg[i]
end
@generated function broadcast_index{T, N}(::Val{true}, arg::AbstractArray{T, N}, shape, i)
    idx = []
    for i = 1:N
        push!(idx, :(ifelse(s[$i] < shape[$i], 1, idx[$i])))
    end
    expr = quote
        $(Expr(:meta, :inline, :propagate_inbounds))
        s = size(arg)
        idx = ind2sub(shape, i)
        @inbounds return arg[$(idx...)]
    end
end
Base.@propagate_inbounds broadcast_index(arg, shape, i) = arg
Base.@propagate_inbounds function broadcast_index{T, N}(
        arg::AbstractArray{T, N}, shape::NTuple{N, Integer}, i
    )
    @inbounds return arg[i]
end
@generated function broadcast_index{T, N}(arg::AbstractArray{T, N}, shape, i)
    idx = []
    for i = 1:N
        push!(idx, :(ifelse(s[$i] < shape[$i], 1, idx[$i])))
    end
    expr = quote
        $(Expr(:meta, :inline, :propagate_inbounds))
        s = size(arg)
        idx = ind2sub(shape, i)
        @inbounds return arg[$(idx...)]
    end
end

if !isdefined(Base.Broadcast, :_broadcast_eltype)
    eltypestuple(a) = (Base.@_pure_meta; Tuple{eltype(a)})
    eltypestuple(T::Type) = (Base.@_pure_meta; Tuple{Type{T}})
    eltypestuple(a, b...) = (Base.@_pure_meta; Tuple{eltypestuple(a).types..., eltypestuple(b...).types...})
    _broadcast_eltype(f, A, Bs...) = Sugar.return_type(f, eltypestuple(A, Bs...))
else
    import Base.Broadcast._broadcast_eltype
end

using ModernGL
function broadcast_similar(f, A, args)
    T = _broadcast_eltype(f, A, args...)
    similar(A, T, usage = GL_STATIC_DRAW)
end

include("broadcast.jl")


# we need to overload all the different broadcast functions, since x... is ambigious

# TODO check size
function Base.map!(f::Function, A::AbstractAccArray, args::AbstractAccArray...)
    acc_broadcast!(f, A, (args...))
end
function Base.map(f::Function, A::AbstractAccArray, args::AbstractAccArray...)
    out = broadcast_similar(f, A, args)
    acc_broadcast!(f, out, (A, args...))
    out
end


#############################
# reduce

# horrible hack to get around of fetching the first element of the GPUArray
# as a startvalue, which is a bit complicated with the current reduce implementation
function startvalue(f, T)
    error("Please supply a starting value for mapreduce. E.g: mapreduce($f, $op, 1, A)")
end
startvalue(::typeof(+), T) = zero(T)
startvalue(::typeof(*), T) = one(T)
startvalue(::typeof(Base.scalarmin), T) = typemax(T)
startvalue(::typeof(Base.scalarmax), T) = typemin(T)

# TODO widen and support Int64 and use Base.r_promote_type
gpu_promote_type{T}(op, ::Type{T}) = T
gpu_promote_type{T<:Base.WidenReduceResult}(op, ::Type{T}) = T
gpu_promote_type{T<:Base.WidenReduceResult}(::typeof(+), ::Type{T}) = T
gpu_promote_type{T<:Base.WidenReduceResult}(::typeof(*), ::Type{T}) = T
gpu_promote_type{T<:Number}(::typeof(+), ::Type{T}) = typeof(zero(T)+zero(T))
gpu_promote_type{T<:Number}(::typeof(*), ::Type{T}) = typeof(one(T)*one(T))
gpu_promote_type{T<:Base.WidenReduceResult}(::typeof(Base.scalarmax), ::Type{T}) = T
gpu_promote_type{T<:Base.WidenReduceResult}(::typeof(Base.scalarmin), ::Type{T}) = T
gpu_promote_type{T<:Base.WidenReduceResult}(::typeof(max), ::Type{T}) = T
gpu_promote_type{T<:Base.WidenReduceResult}(::typeof(min), ::Type{T}) = T

function Base.mapreduce{T, N}(f::Function, op::Function, A::AbstractAccArray{T, N})
    OT = gpu_promote_type(op, T)
    v0 = startvalue(op, OT) # TODO do this better
    mapreduce(f, op, v0, A)
end


function Base.mapreduce(f, op, v0, A::AbstractAccArray, B::AbstractAccArray, C::Number)
    acc_mapreduce(f, op, v0, A, (B, C))
end
function Base.mapreduce(f, op, v0, A::AbstractAccArray, B::AbstractAccArray)
    acc_mapreduce(f, op, v0, A, (B,))
end
function Base.mapreduce(f, op, v0, A::AbstractAccArray)
    acc_mapreduce(f, op, v0, A, ())
end

############################################
# Constructor

function Base.fill!{T, N}(A::AbstractAccArray{T, N}, val)
    A .= identity.(T(val))
    A
end
function Base.rand{T <: AbstractAccArray, ET}(::Type{T}, ::Type{ET}, size...)
    T(rand(ET, size...))
end


############################################
# serialization

const BaseSerializer = if isdefined(Base, :AbstractSerializer)
    Base.AbstractSerializer
elseif isdefined(Base, :SerializationState)
    Base.SerializationState
else
    error("No Serialization type found. Probably unsupported Julia version")
end

function Base.serialize{T <: GPUArray}(s::BaseSerializer, t::T)
    Base.serialize_type(s, T)
    serialize(s, Array(t))
end
function Base.deserialize{T <: GPUArray}(s::BaseSerializer, ::Type{T})
    A = deserialize(s)
    T(A)
end

import Base: copy!, getindex, setindex!

@inline unpack_buffer(x) = x
@inline unpack_buffer(x::AbstractAccArray) = buffer(x)

function to_cartesian(indices::Tuple)
    start = CartesianIndex(map(indices) do val
        isa(val, Integer) && return val
        isa(val, UnitRange) && return first(val)
        error("GPU indexing only defined for integers or unit ranges. Found: $val")
    end)
    stop = CartesianIndex(map(indices) do val
        isa(val, Integer) && return val
        isa(val, UnitRange) && return last(val)
        error("GPU indexing only defined for integers or unit ranges. Found: $val")
    end)
    CartesianRange(start, stop)
end

crange(start, stop) = CartesianRange(CartesianIndex(start), CartesianIndex(stop))

#Hmmm... why is this not part of the Array constructors???
#TODO Figure out or issue THEM JULIA CORE PEOPLE SO HARD ... or PR? Who'd know
function array_convert{T, N}(t::Type{Array{T, N}}, x::Array)
    convert(t, x)
end


array_convert{T, N}(t::Type{Array{T, N}}, x::T) = [x]

function array_convert{T, N, T2}(t::Type{Array{T, N}}, x::T2)
    arr = collect(x) # iterator
    dims = ntuple(Val{N}) do i
        ifelse(ndims(arr) >= i, size(arr, i), 1)
    end
    return reshape(map(T, arr), dims) # broadcast dims
end

for (D, S) in ((AbstractAccArray, AbstractArray), (AbstractArray, AbstractAccArray), (AbstractAccArray, AbstractAccArray))
    @eval begin
        function copy!(
                dest::$D, doffset::Integer,
                src::$S, soffset::Integer, amount::Integer
            )
            copy!(
                unpack_buffer(dest), doffset,
                unpack_buffer(src), soffset, amount
            )
        end
        function copy!{T, N}(
                dest::$D{T, N}, rdest::NTuple{N, UnitRange},
                src::$S{T, N}, ssrc::NTuple{N, UnitRange},
            )
            drange = crange(start.(rdest), last.(rdest))
            srange = crange(start.(ssrc), last.(ssrc))
            copy!(dest, drange, src, srange)
        end
        function copy!{T}(
                dest::$D{T, 1}, d_range::CartesianRange{CartesianIndex{1}},
                src::$S{T, 1}, s_range::CartesianRange{CartesianIndex{1}},
            )
            amount = length(d_range)
            if length(s_range) != amount
                throw(ArgumentError("Copy range needs same length. Found: dest: $amount, src: $(length(s_range))"))
            end
            amount == 0 && return dest
            d_offset = first(d_range)[1]
            s_offset = first(s_range)[1]
            copy!(dest, d_offset, src, s_offset, amount)
        end
        function copy!{T, N}(
                dest::$D{T, N}, rdest::CartesianRange{CartesianIndex{N}},
                src::$S{T, N}, ssrc::CartesianRange{CartesianIndex{N}},
            )
            copy!(unpack_buffer(dest), rdest, unpack_buffer(src), ssrc)
        end
        function copy!{T, N}(
                dest::$D{T, N}, src::$S{T, N}
            )
            len = length(src)
            len == 0 && return dest
            if length(dest) > len
                throw(BoundsError(dest, length(src)))
            end
            copy!(dest, 1, src, 1, len)
        end
    end
end

function Base.setindex!{T, N}(A::AbstractAccArray{T, N}, value, indexes...)
    # similarly, value should always be a julia array
    shape = map(length, indexes)
    if !isa(value, T) # TODO, shape check errors for x[1:3] = 1
        Base.setindex_shape_check(value, indexes...)
    end
    checkbounds(A, indexes...)
    v = array_convert(Array{T, N}, value)
    # since you shouldn't update GPUArrays with single indices, we simplify the interface
    # by always mapping to ranges
    ranges_dest = to_cartesian(indexes)
    ranges_src = CartesianRange(size(v))

    copy!(A, ranges_dest, v, ranges_src)
    return
end

function Base.getindex{T, N}(A::AbstractAccArray{T, N}, indexes...)
    cindexes = Base.to_indices(A, indexes)
    # similarly, value should always be a julia array
    # We shouldn't really bother about checkbounds performance, since setindex/getindex will always be relatively slow
    checkbounds(A, cindexes...)

    shape = map(length, cindexes)
    result = Array{T, N}(shape)
    ranges_src = to_cartesian(cindexes)
    ranges_dest = CartesianRange(shape)
    copy!(result, ranges_dest, A, ranges_src)
    if all(i-> isa(i, Integer), cindexes) # scalar
        return result[]
    else
        return result
    end
end
