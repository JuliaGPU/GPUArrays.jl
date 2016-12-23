module JLBackend

using ..GPUArrays
using OpenCL

import GPUArrays: buffer, create_buffer, Context, AbstractAccArray
import Base.Threads: @threads

immutable JLContext <: Context
    nthreads::Int
end
immutable JLArray{T, N} <: AbstractAccArray{T, N}
    buffer::Array{T, N}
    context::JLContext
end
Base.@propagate_inbounds Base.getindex(A::JLArray, i::Integer) = A.buffer[i]
Base.@propagate_inbounds Base.setindex!(A::JLArray, val, i::Integer) = (A.buffer[i] = val)
Base.linearindexing{T <: JLArray}(::Type{T}) = Base.LinearFast
Base.size(x::JLArray) = size(buffer(x))
Base.size(x::JLArray, i::Integer) = size(buffer(x), i)

global current_context, make_current, init
let contexts = JLContext[]
    all_contexts() = copy(contexts)::Vector{JLContext}
    current_context() = last(contexts)::JLContext
    function init()
        ctx = JLContext(Base.Threads.nthreads())
        push!(contexts, ctx)
        ctx
    end
end

Base.show(io::IO, ctx::JLContext) = print(io, "JLContext")


function JLArray{T, N}(A::AbstractArray{T, N})
    JLArray{T, N}(A, current_context())
end

function Base.similar{T, N}(A::Type{JLArray{T, N}}, sz::Tuple)
    JLArray{T, N}(similar(A, sz), current_context())
end
function (AT::Type{Array{T, N}}){T, N}(A::JLArray)
    convert(AT, buffer(A))
end

# Map idx!
#TODO implement for args..., but just using a tuple here is simpler for now
# (`args...`` quickly gets you into a type inf and boxing mess)
function map_idx!{F, T <: Tuple}(f::F, A::JLArray, args::T)
    n = length(A); dims = size(A)
    @threads for i = 1:n
        idx = Vec{3, Int}(ind2sub(dims, i))
        @inbounds A[i] = f(idx, A, args)
    end
    A
end
@inline function Base.map!{F}(f::F, A::JLArray, args...)
    map!(f, A, args)
end
@inline function Base.map{F}(f::F, A::JLArray, args...)
    out = similar(A)
    map!(f, out, (A, args...))
end
@generated function Base.map!{F, N}(f::F, A::JLArray, args::NTuple{N})
    arg_expr = [:(args[$i]) for i=1:N]
    quote
        n = length(A)
        @threads for i = 1:n
            @inbounds A[i] = f(A[i], $(arg_expr...))
        end
        A
    end
end

function Base.map!{F}(f::F, dest::JLArray, A::JLArray, B::JLArray)
    n = length(dest)
    @threads for i = 1:n
        @inbounds dest[i] = f(A[i], B[i])
    end
    dest
end

@inline function Base.broadcast!{F}(f::F, dest::JLArray, A::JLArray, B::JLArray)
    map!(f, dest, A, B)
end

include("liftbase.jl")

end #CLBackend
