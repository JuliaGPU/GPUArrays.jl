module JLBackend

using ..GPUArrays
using Compat
using StaticArrays, Interpolations

import GPUArrays: buffer, create_buffer, Context, context, mapidx
import GPUArrays: AbstractAccArray, AbstractSampler, acc_mapreduce, acc_broadcast!
import GPUArrays: broadcast_index, hasblas, blas_module, blasbuffer

import Base.Threads: @threads

immutable JLContext <: Context
    nthreads::Int
end

global current_context, make_current, init
let contexts = JLContext[]
    all_contexts() = copy(contexts)::Vector{JLContext}
    current_context() = last(contexts)::JLContext
    function init()
        ctx = JLContext(Base.Threads.nthreads())
        GPUArrays.make_current(ctx)
        push!(contexts, ctx)
        ctx
    end
end


immutable Sampler{T, N, Buffer} <: AbstractSampler{T, N}
    buffer::Buffer
    size::SVector{N, Float32}
end

function Sampler{T, N}(A::AbstractArray{T, N}, interpolation = Linear(), edge = Flat())
    Ai = extrapolate(interpolate(A, BSpline(interpolation), OnCell()), edge)
    Sampler{T, N, typeof(Ai)}(Ai, SVector{N, Float32}(size(A)) - 1f0)
end

@generated function Base.getindex{T, B, N, IF <: AbstractFloat}(x::Sampler{T, N, B}, idx::StaticVector{N, IF})
    quote
        scaled = idx .* x.size + 1f0
        x.buffer[$(ntuple(i-> :(scaled[$i]), Val{N})...)] # why does splatting not work -.-
    end
end

@compat const JLArray{T, N} = GPUArray{T, N, Array{T, N}, JLContext}

create_buffer{T, N}(ctx::JLContext, A::AbstractArray{T, N}) = A
function create_buffer{T, N}(
        ctx::JLContext, ::Type{T}, sz::NTuple{N, Int}
    )
    Array{T, N}(sz)
end

function Base.similar{T, N, ET}(x::JLArray{T, N}, ::Type{ET}, sz::NTuple{N, Int}; kw_args...)
    ctx = context(x)
    b = similar(buffer(x), ET, sz)
    GPUArray{ET, N, typeof(b), typeof(ctx)}(b, sz, ctx)
end
####################################
# constructors

function (::Type{JLArray}){T, N}(A::Array{T, N})
    JLArray{T, N}(A, size(A), current_context())
end

function (AT::Type{Array{T, N}}){T, N}(A::JLArray{T, N})
    buffer(A)
end
function (::Type{A}){A <: JLArray, T, N}(x::Array{T, N})
    JLArray{T, N}(x, current_context())
end

nthreads{T, N}(a::JLArray{T, N}) = context(a).nthreads

Base.@propagate_inbounds Base.getindex{T, N}(A::JLArray{T, N}, i::Integer) = A.buffer[i]
Base.@propagate_inbounds Base.setindex!{T, N}(A::JLArray{T, N}, val, i::Integer) = (A.buffer[i] = val)
@compat Base.IndexStyle{T, N}(::Type{JLArray{T, N}}) = IndexLinear()

function Base.show(io::IO, ctx::JLContext)
    cpu = Sys.cpu_info()
    print(io, "JLContext $(cpu[1].model) with $(ctx.nthreads) threads")
end
##############################################
# Implement BLAS interface

blasbuffer(ctx::JLContext, A) = Array(A)
blas_module(::JLContext) = Base.BLAS


# lol @threads makes @generated say that we have an unpure @generated function body.
# Lies!
# Well, we know how to deal with that from the CUDA backend
for i = 0:7
    fargs = ntuple(x-> :(broadcast_index(args[$x], sz, i)), i)
    fidxargs = ntuple(x-> :(args[$x]), i)
    @eval begin
        function acc_broadcast!{F, T, N}(f::F, A::JLArray{T, N}, args::NTuple{$i, Any})
            n = length(A)
            sz = size(A)
            @threads for i = 1:n
                @inbounds A[i] = f($(fargs...))
            end
            return A
        end

        function mapidx{F, T, N}(f::F, data::JLArray{T, N}, args::NTuple{$i, Any})
            @threads for i in eachindex(data)
                f(i, data, $(fidxargs...))
            end
        end
        function acc_mapreduce{T, N}(f, op, v0, A::JLArray{T, N}, args::NTuple{$i, Any})
            n = Base.Threads.nthreads()
            arr = Vector{typeof(op(v0, v0))}(n)
            slice = ceil(Int, length(A) / n)
            sz = size(A)
            @threads for threadid in 1:n
                low = ((threadid - 1) * slice) + 1
                high = min(length(A), threadid * slice)
                r = v0
                for i in low:high
                    r = op(r, f(A[i], $(fargs...)))
                end
                arr[threadid] = r
            end
            reduce(op, arr)
        end
    end
end

include("fft.jl")
hasblas(::JLContext) = true

end #JLBackend

using .JLBackend
export JLBackend
