abstract AbstractAccArray{T, N} <: DenseArray{T, N}

type GPUArray{T, N, B, C} <: AbstractAccArray{T, N}
    buffer::B
    size::NTuple{N, Int}
    context::C
end

function Base.similar{T <: GPUArray, ET, N}(
        ::Type{T}, ::Type{ET}, sz::NTuple{N, Int};
        context::Context = current_context(), kw_args...
    )
    b = create_buffer(context, ET, sz; kw_args...)
    GPUArray{ET, N, typeof(b), typeof(context)}(b, sz, context)
end
#=
context interface
=#

buffer(A::AbstractAccArray) = A.buffer
context(A::AbstractAccArray) = A.context

#=
AbstractArray interface
=#

Base.eltype{T}(::AbstractAccArray{T}) = T
Base.size(A::AbstractAccArray) = A.size
Base.size(A::AbstractAccArray, i::Int) = A.size[i]

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
function Base.similar{T <: AbstractAccArray, N, ET}(x::T, ::Type{ET}, dims::NTuple{N, Int})
    out = similar(buffer(x), ET, dims)
    T(out)
end

#=
Host to Device data transfers
=#
function (::Type{A}){A <: AbstractAccArray}(x::AbstractArray)
    A(collect(x))
end
function (::Type{A}){A <: AbstractAccArray}(x::Array)
    out = similar(A, eltype(x), size(x))
    unsafe_copy!(out, x)
    out
end

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
    unsafe_copy!(hostarray, device_array)
    hostarray
end


#=
Copying
=#

function Base.unsafe_copy!{T, N}(dest::Array{T, N}, source::AbstractAccArray{T, N})
    Base.unsafe_copy!(dest, buffer(source))
end
function Base.unsafe_copy!{T, N}(dest::AbstractAccArray{T, N}, source::Array{T, N})
    Base.unsafe_copy!(buffer(dest), source)
end



######################################
# Broadcast

# helper

#Broadcast
@generated function broadcast_index{T, N}(arg::AbstractArray{T, N}, shape, idx)
    idx = ntuple(i->:(ifelse(s[$i] < shape[$i], 1, idx[$i])), N)
    expr = quote
        s = size(arg)::NTuple{N, Int}
        @inbounds i = CartesianIndex{N}(($(idx...),)::NTuple{N, Int})
        @inbounds return arg[i]::T
    end
end
function broadcast_index{T}(arg::T, shape, idx)
    arg::T
end



# It is kinda hard to overwrite map/broadcast, which is why we lift it to our
# our own broadcast function.
# It has the signature:
# f::Function, Context, Main/Out::AccArray, args::NTuple{N}
# All arrays are already lifted and shape checked
function acc_broadcast! end

# seems to be needed for ambiguities
function Base.broadcast!(f::typeof(identity), A::AbstractAccArray, args::Number)
    acc_broadcast!(f, A, (args,))
end
function Base.broadcast!(f::Function, A::AbstractAccArray)
    acc_broadcast!(f, A, ())
end
# Base.Broadcast.check_broadcast_shape(size(A), As...)
function Base.broadcast!(f::Function, A::AbstractAccArray, B::AbstractAccArray)
    acc_broadcast!(f, A, (B,))
end
function Base.broadcast!(f::Function, A::AbstractAccArray, B::Number)
    acc_broadcast!(f, A, (B,))
end
# Base.Broadcast.check_broadcast_shape(size(A), As...)
function Base.broadcast!(f::Function, A::AbstractAccArray, B::AbstractAccArray, args::AbstractAccArray...)
    acc_broadcast!(f, A, (B, args...))
end
function Base.broadcast!(f::Function, A::AbstractAccArray, B::AbstractAccArray, args::Number)
    acc_broadcast!(f, A, (B, args...))
end

function Base.broadcast!(f::Function, A::AbstractAccArray, B::AbstractAccArray, C::AbstractAccArray, D::Number)
    acc_broadcast!(f, A, (B, C, D))
end

function Base.broadcast(f::Function, A::AbstractAccArray)
    out = similar(A)
    acc_broadcast!(f, out, (A,))
    out
end
function Base.broadcast(f::Function, A::AbstractAccArray, B::Number)
    out = similar(A)
    acc_broadcast!(f, out, (A, B,))
    out
end

function Base.broadcast(f::Function, A::AbstractAccArray, args::AbstractAccArray...)
    out = similar(A)
    acc_broadcast!(f, out, (A, args...))
    out
end

# TODO check size
function Base.map!(f::Function, A::AbstractAccArray, args::AbstractAccArray...)
    acc_broadcast!(f, A, (args...))
end
function Base.map(f::Function, A::AbstractAccArray, args::AbstractAccArray...)
    broadcast(f, A, (args...))
end



############################################
# Constructor

function Base.fill!{N, T}(A::AbstractAccArray{N, T}, val)
    A .= identity.(T(val))
end
#
# function Base.rand{T <: AbstractAccArray, ET}(::Type{T}, ET, size...)
#     T(rand(ET, size...))
# end
