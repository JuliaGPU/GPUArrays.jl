@compat abstract type AbstractAccArray{T, N} <: DenseArray{T, N} end

type GPUArray{T, N, B, C} <: AbstractAccArray{T, N}
    buffer::B
    size::NTuple{N, Int}
    context::C
end
@compat const AccVector{T} = AbstractAccArray{T, 1}
@compat const AccMatrix{T} = AbstractAccArray{T, 2}
@compat const AccVecOrMat{T} = Union{AbstractAccArray{T, 1}, AbstractAccArray{T, 2}}

# interfaces

#=
Interface for accessing the lower level
=#

buffer(A::AbstractAccArray) = A.buffer
context(A::AbstractAccArray) = A.context

# GPU Local Memory
immutable LocalMemory{T} <: AbstractAccArray{T, 1}
    size::Int
end



"""
linear index in a GPU kernel
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
function Base.similar{T <: AbstractAccArray, ET}(x::T, ::Type{ET})
    similar(x, ET, size(x))
end
function Base.similar{T <: AbstractAccArray, N}(x::T, dims::NTuple{N, Int})
    similar(x, eltype(x), dims)
end

function Base.similar{T <: GPUArray, ET, N}(
        ::Type{T}, ::Type{ET}, sz::NTuple{N, Int};
        context::Context = current_context(), kw_args...
    )
    b = create_buffer(context, ET, sz; kw_args...)
    GPUArray{ET, N, typeof(b), typeof(context)}(b, sz, context)
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


#=
Copying
=#

function Base.copy!{T, N}(dest::Array{T, N}, source::AbstractAccArray{T, N})
    copy!(dest, buffer(source))
end
function Base.copy!{T, N}(dest::AbstractAccArray{T, N}, source::Array{T, N})
    copy!(buffer(dest), source)
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

function broadcast_similar(f, A, args)
    T = _broadcast_eltype(f, A, args...)
    similar(A, T)
end


# we need to overload all the different broadcast functions, since x... is ambigious

function Base.broadcast(f::Function, A::AbstractAccArray)
    out = broadcast_similar(f, A, ())
    acc_broadcast!(f, out, (A,))
    out
end
function Base.broadcast(f::Function, A::AbstractAccArray, B::Number)
    out = broadcast_similar(f, A, B)
    acc_broadcast!(f, out, (A, B,))
    out
end
function Base.broadcast(f::Function, A::AbstractAccArray, args::AbstractAccArray...)
    out = broadcast_similar(f, A, args)
    acc_broadcast!(f, out, (A, args...))
    out
end
function Base.broadcast!(f::Function, A::AbstractAccArray, args::AbstractAccArray...)
    acc_broadcast!(f, A, (args...))
end
# identity is overloaded in Base, so there will be ambiguities without explicitely overloading it!
function Base.broadcast!(f::typeof(identity), A::AbstractAccArray, B::Number)
    acc_broadcast!(f, A, (B,))
end
function Base.broadcast!(f::typeof(identity), A::AbstractAccArray, B::AbstractAccArray)
    acc_broadcast!(f, A, (B,))
end
# Various combinations with scalars
function Base.broadcast!(f::Function, A::AbstractAccArray)
    acc_broadcast!(f, A, ())
end
function Base.broadcast!(f::Function, A::AbstractAccArray, B)
    acc_broadcast!(f, A, (B,))
end
function Base.broadcast!(f::Function, A::AbstractAccArray, B::AbstractAccArray, args...)
    acc_broadcast!(f, A, (B, args...))
end


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

function Base.serialize{T<:GPUArray}(s::BaseSerializer, t::T)
    Base.serialize_type(s, T)
    serialize(s, Array(t))
end
function Base.deserialize{T<:GPUArray}(s::BaseSerializer, ::Type{T})
    A = deserialize(s)
    T(A)
end
