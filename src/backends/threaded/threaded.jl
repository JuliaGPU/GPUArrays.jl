module JLBackend

using ..GPUArrays
using StaticArrays, Interpolations

import GPUArrays: buffer, create_buffer, Context, context, mapidx, unpack_buffer
import GPUArrays: AbstractAccArray, AbstractSampler, acc_mapreduce, gpu_call
import GPUArrays: hasblas, blas_module, blasbuffer, default_buffer_type
import GPUArrays: unsafe_reinterpret, broadcast_index, linear_index
import GPUArrays: is_cpu, name, threads, blocks, global_memory
import GPUArrays: new_context, init, free_global_memory

import Base.Threads: @threads

immutable JLContext <: Context
    nthreads::Int
end
# TODO,one could have multiple CPUs ?
immutable JLDevice <: Context
    index::Int
end


global all_contexts, current_context, current_device
let contexts = Dict{JLDevice, JLContext}(), active_device = JLDevice[]
    all_contexts() = values(contexts)
    function current_device()
        if isempty(active_device)
            push!(active_device, JLDevice(0))
        end
        active_device[]
    end
    current_context() = contexts[current_device()]
    function GPUArrays.init(dev::JLDevice)
        GPUArrays.setbackend!(JLBackend)
        if isempty(active_device)
            push!(active_device, dev)
        else
            active_device[] = dev
        end
        ctx = get!(()-> new_context(dev), contexts, dev)
        ctx
    end
end

new_context(dev::JLDevice) = JLContext(Threads.nthreads())
threads(x::JLDevice) = Base.Threads.nthreads()
global_memory(x::JLDevice) = Sys.total_memory()
free_global_memory(x::JLDevice) = Sys.free_memory()
name(x::JLDevice) = Sys.cpu_info()[1].model
is_cpu(::JLDevice) = true

devices() = (JLDevice(0),)


immutable Sampler{T, N, Buffer} <: AbstractSampler{T, N}
    buffer::Buffer
    size::SVector{N, Float32}
end
Base.IndexStyle{T, N}(::Type{Sampler{T, N}}) = IndexLinear()
(AT::Type{Array}){T, N, B}(s::Sampler{T, N, B}) = parent(parent(buffer(s)))

function Sampler{T, N}(A::Array{T, N}, interpolation = Linear(), edge = Flat())
    Ai = extrapolate(interpolate(A, BSpline(interpolation), OnCell()), edge)
    Sampler{T, N, typeof(Ai)}(Ai, SVector{N, Float32}(size(A)) - 1f0)
end

@generated function Base.getindex{T, B, N, IF <: AbstractFloat}(A::Sampler{T, N, B}, idx::StaticVector{N, IF})
    quote
        scaled = idx .* A.size + 1f0
        A.buffer[$(ntuple(i-> :(scaled[$i]), Val{N})...)] # why does splatting not work -.-
    end
end
Base.@propagate_inbounds function Base.getindex(A::Sampler, indexes...)
    Array(A)[indexes...]
end
Base.@propagate_inbounds function Base.setindex!(A::Sampler, val, indexes...)
    Array(A)[indexes...] = val
end

const JLArray{T, N} = GPUArray{T, N, Array{T, N}, JLContext}

default_buffer_type{T, N}(::Type, ::Type{Tuple{T, N}}, ::JLContext) = Array{T, N}

function (AT::Type{JLArray{T, N}}){T, N}(
        size::NTuple{N, Int};
        context = current_context(),
        kw_args...
    )
    # cuda doesn't allow a size of 0, but since the length of the underlying buffer
    # doesn't matter, with can just initilize it to 0
    AT(Array{T, N}(size), size, context)
end

function Base.similar{T, N, ET}(x::JLArray{T, N}, ::Type{ET}, sz::NTuple{N, Int}; kw_args...)
    ctx = context(x)
    b = similar(buffer(x), ET, sz)
    GPUArray{ET, N, typeof(b), typeof(ctx)}(b, sz, ctx)
end
####################################
# constructors
function unsafe_reinterpret(::Type{T}, A::JLArray{ET}, dims::NTuple{N, Integer}) where {T, ET, N}
    buff = buffer(A)
    newbuff = reinterpret(T, buff, dims)
    ctx = context(A)
    GPUArray{T, length(dims), typeof(newbuff), typeof(ctx)}(newbuff, dims, ctx)
end

function (::Type{JLArray}){T, N}(A::Array{T, N})
    JLArray{T, N}(A, size(A), current_context())
end

function (AT::Type{Array{T, N}}){T, N}(A::JLArray{T, N})
    buffer(A)
end
function (::Type{A}){A <: JLArray, T, N}(x::Array{T, N})
    JLArray{T, N}(x, size(x), current_context())
end

nthreads{T, N}(a::JLArray{T, N}) = context(a).nthreads

Base.@propagate_inbounds Base.getindex{T, N}(A::JLArray{T, N}, i::Integer) = A.buffer[i]
Base.@propagate_inbounds Base.setindex!{T, N}(A::JLArray{T, N}, val, i::Integer) = (A.buffer[i] = val)
Base.IndexStyle{T, N}(::Type{JLArray{T, N}}) = IndexLinear()

function Base.show(io::IO, ctx::JLContext)
    println("Threaded Julia Context with:")
    GPUArrays.device_summary(io, JLDevice(0))
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
        function mapidx{F, T, N}(f::F, data::JLArray{T, N}, args::NTuple{$i, Any})
            @threads for i in eachindex(data)
                f(i, data, $(fidxargs...))
            end
        end
        @noinline function inner_mapreduce(threadid, slice, v0, op, f, A, args::NTuple{$i, Any}, arr, sz)
            low = ((threadid - 1) * slice) + 1
            high = min(length(A), threadid * slice)
            r = v0
            for i in low:high
                @inbounds r = op(r, f(A[i], $(fargs...)))
            end
            @inbounds arr[threadid] = r
        end
        function acc_mapreduce{T, N}(f, op, v0, A::JLArray{T, N}, args::NTuple{$i, Any})
            n = Base.Threads.nthreads()
            arr = Vector{typeof(op(v0, v0))}(n)
            slice = ceil(Int, length(A) / n)
            sz = size(A)
            @threads for threadid in 1:n
                inner_mapreduce(threadid, slice, v0, op, f, A, args, arr, sz)
            end
            reduce(op, arr)
        end
    end
end

linear_index(A::AbstractArray, state) = state


@generated function parallel_kernel(id, width, f, args::NTuple{N, T where T}) where N
    args_unrolled = ntuple(Val{N}) do i
        :(args[$i])
    end
    quote
        @simd for idx = (id + 1):(id + width)
            f(idx, $(args_unrolled...))
        end
        return
    end
end
function gpu_call(f, A::JLArray, args::Tuple, globalsize = length(A), local_size = 0)
    unpacked_args = unpack_buffer.(args)
    n = nthreads(A)
    len = prod(globalsize)
    width = floor(Int, len / n)
    if width <= 10 # arbitrary number. TODO figure out good value for when it's worth launching threads
        parallel_kernel(0, len, f, unpacked_args)
        return
    end
    @threads for id = 1:n
        parallel_kernel((id - 1) * width, width, f, unpacked_args)
    end
    len_floored = width * n
    rest = len - len_floored
    if rest > 0
        parallel_kernel(len_floored, rest, f, unpacked_args)
    end
    return
end

include("fft.jl")
hasblas(::JLContext) = true

end #JLBackend

using .JLBackend
export JLBackend
