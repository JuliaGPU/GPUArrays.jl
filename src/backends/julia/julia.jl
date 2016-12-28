module JLBackend

using ..GPUArrays

import GPUArrays: buffer, create_buffer, Context, AbstractAccArray
import GPUArrays: broadcast_index, acc_broadcast!

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
Base.linearindexing{T <: JLArray}(::Type{T}) = Base.LinearFast()
Base.size(x::JLArray) = size(buffer(x))

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


function (::Type{JLArray}){T, N}(A::Array{T, N})
    JLArray{T, N}(A, current_context())
end

function Base.similar{T, N}(A::Type{JLArray{T, N}}, ::Type{T}, sz::Tuple)
    JLArray{T, N}(Array(T, sz), current_context())
end
function (AT::Type{Array{T, N}}){T, N}(A::JLArray)
    convert(AT, buffer(A))
end
function (::Type{A}){A <: JLArray, T, N}(x::Array{T, N})
    JLArray{T, N}(x, current_context())
end

# lol @threads makes @generated say that we have an unpure @generated function body.
# Lies!
# Well, we know how to deal with that from the CUDA backend
for i=0:10
    fargs = ntuple(x-> :(broadcast_index(args[$x], sz, idx)), i)
    @eval begin
        function acc_broadcast!{F}(f::F, A::JLArray, args::NTuple{$i})
            n = length(A)
            sz = size(A)
            @threads for i = 1:n
                idx = CartesianIndex(ind2sub(sz, Int(i)))
                @inbounds A[idx] = f($(fargs...))
            end
            return
        end
    end
end


include("liftbase.jl")

end #CLBackend
