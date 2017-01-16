
module CUBackend

using ..GPUArrays, CUDAnative

import CUDAdrv, CUDArt #, CUFFT

import GPUArrays: buffer, create_buffer, acc_broadcast!
import GPUArrays: Context, GPUArray, context, broadcast_index

immutable GraphicsResource{T}
    glbuffer::T
    resource::Ref{CUDArt.rt.cudaGraphicsResource_t}
    ismapped::Bool
end

immutable CUContext <: Context
    ctx::CUDAdrv.CuContext
    device::CUDAdrv.CuDevice
end

Base.show(io::IO, ctx::CUContext) = print(io, "CUContext")

function any_context()
    dev = CUDAdrv.CuDevice(0)
    ctx = CUDAdrv.CuContext(dev)
    CUContext(ctx, dev)
end

#typealias GLArrayImg{T, N} GPUArray{T, N, gl.Texture{T, N}, GLContext}
typealias CUArray{T, N, B} GPUArray{T, N, B, CUContext} #, GLArrayImg{T, N}}
typealias CUArrayBuff{T, N} CUArray{T, N, CUDAdrv.CuArray{T, N}}


global init, all_contexts, current_context
let contexts = CUContext[]
    all_contexts() = copy(contexts)::Vector{CUContext}
    current_context() = last(contexts)::CUContext
    function init(;ctx = any_context())
        GPUArrays.make_current(ctx)
        push!(contexts, ctx)
        ctx
    end
end

function create_buffer{T, N}(ctx::CUContext, ::Type{T}, sz::NTuple{N, Int}; kw_args...)
    CUDAdrv.CuArray{T}(sz)
end
function Base.unsafe_copy!{T,N}(dest::Array{T,N}, source::CUArray{T,N})
    copy!(dest, buffer(source))
end
function Base.unsafe_copy!{T,N}(dest::CUArray{T,N}, source::Array{T,N})
    copy!(buffer(dest), source)
end

function thread_blocks_heuristic(A::AbstractArray)
    thread_blocks_heuristic(size(A))
end

function thread_blocks_heuristic{N}(s::NTuple{N, Int})
    len = prod(s)
    threads = min(len, 1024)
    blocks = ceil(Int, len/threads)
    blocks, threads
end

@inline function linear_index()
    (blockIdx().x-1) * blockDim().x + threadIdx().x
end
@inline function map_eachindex_kernel{N}(A, f, args::Vararg{N})
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if length(A) >= i
        f(CartesianIndex(ind2sub(size(A), Int(i))), A, args...)
    end
    nothing
end



unpack_cu_array(x) = x
unpack_cu_array{T,N}(x::CUArray{T,N}) = buffer(x)

@inline function call_cuda(kernel, A::CUArray, rest...)
    blocks, thread = thread_blocks_heuristic(A)
    args = map(unpack_cu_array, rest)
    @cuda (blocks, thread) kernel(buffer(A), args...)
end


#####################################
# The problem is, that I can't pass Tuple{CuArray} as a type, so I can't
# write a @generated function to unrole the arguments.
# And without unroling of the arguments, GPU codegen will cry!

for i=0:10
    args = ntuple(x-> Symbol("arg_", x), i)
    fargs = ntuple(x-> :(broadcast_index($(args[x]), sz, idx)), i)
    @eval begin
        function broadcast_kernel(A, f, $(args...))
            i = Int((blockIdx().x-1) * blockDim().x + threadIdx().x)
            @inbounds if i <= length(A)
                sz = size(A)
                idx = CartesianIndex(ind2sub(sz, Int(i)))
                A[idx] = f($(fargs...))
            end
            nothing
        end
    end
end

function acc_broadcast!{F <: Function, N}(f::F, A::CUArray, args::NTuple{N})
    call_cuda(broadcast_kernel, A, f, args...)
end

#################################
# Reduction

function reduce_warp{T,F<:Function}(val::T, op::F)
    offset = CUDAnative.warpsize() ÷ 2
    while offset > 0
        val = op(val, shfl_down(val, offset))
        offset ÷= 2
    end
    return val
end

function reduce_block{T, F <: Function}(val::T, op::F, v0::T)
    shared = @cuStaticSharedMem(T, 32)
    wid, lane = fldmod1(threadIdx().x, CUDAnative.warpsize())
    val = reduce_warp(val, op)
    if lane == 1
        @inbounds shared[wid] = val
    end
    sync_threads()
    # read from shared memory only if that warp existed
    @inbounds begin
        val = (threadIdx().x <= fld(blockDim().x, CUDAnative.warpsize())) ? shared[lane] : v0
    end
    if wid == 1
        # final reduce within first warp
        val = reduce_warp(val, op)
    end
    return val
end
function reduce_kernel{F <: Function, OP <: Function,T1, T2, N}(
        A::AbstractArray{T1, N}, out::AbstractArray{T2,1}, f::F, op::OP, v0::T2
    )
    #reduce multiple elements per thread

    i = Int((blockIdx().x-Int32(1)) * blockDim().x + threadIdx().x)
    step = blockDim().x * gridDim().x
    result = v0
    while i <= length(A)
        @inbounds result = op(result, f(A[i]))
        i += step
    end
    result = reduce_block(result, op, v0)
    if (threadIdx().x == 1)
        @inbounds out[blockIdx().x] = result;
    end
    return
end

# horrible hack to get around of fetching the first element of the GPUArray
# as a startvalue, which is a bit complicated with the current reduce implementation
function startvalue(op, T)
    error("Please supply a starting value for mapreduce. E.g: mapreduce($f, $op, 1, A)")
end
startvalue(::typeof(+), T) = zero(T)
startvalue(::typeof(*), T) = one(T)
startvalue(::typeof(Base.scalarmin), T) = typemax(T)
startvalue(::typeof(Base.scalarmax), T) = typemin(T)

function Base.mapreduce{T,N}(f::Function, op::Function, A::CUArray{T, N})
    OT = Base.r_promote_type(op, T)
    v0 = startvalue(op, OT) # TODO do this better
    mapreduce(f, op, v0, A)
end
function Base.mapreduce{T, OT, N}(f::Function, op::Function, v0::OT, A::CUArray{T,N})
    dev = context(A).device
    @assert(CUDAdrv.capability(dev) >= v"3.0", "Current CUDA reduce implementation requires a newer GPU")
    threads = 512
    blocks = min((length(A) + threads - 1) ÷ threads, 1024)
    out = similar(buffer(A), OT, (blocks,))
    # TODO MAKE THIS WORK FOR ALL FUNCTIONS .... v0 is really unfit for parallel reduction
    # since every thread basically needs its own v0
    @cuda (blocks, threads) reduce_kernel(buffer(A), out, f, op, v0)
    # for this size it doesn't seem beneficial to run on gpu?!
    # TODO actually benchmark this theory
    mapreduce(f, op, Array(out))
end


#  TODO figure out how interact with CUDArt and CUDAdr
#GFFT = GPUArray(Complex64, div(size(G,1),2)+1, size(G,2))
# function Base.fft!(A::CUArray)
#     G, GFFT = CUFFT.RCpair(A)
#     fft!(G, GFFT)
# end
# function Base.fft!(out::CUArray, A::CUArray)
#     plan(out, A)(out, A, true)
# end
#
# function Base.ifft!(A::CUArray)
#     G, GFFT = CUFFT.RCpair(A)
#     ifft!(G, GFFT)
# end
# function Base.ifft!(out::CUArray, A::CUArray)
#     plan(out, A)(out, A, false)
# end


########################################
# CUBLAS
using CUBLAS
import CUDArt

function blasbuffer(ctx::CUContext, A)
    buff = buffer(A)
    devptr = pointer(buff)
    device = CUDAdrv.device(devptr.ctx).handle
    CUDArt.CudaArray(CUDArt.CudaPtr(devptr.ptr), size(A), Int(device))
end

function convert{T <: CUArray}(t::T, A::CUDArt.CudaArray)
    ctx = context(t)
    ptr = DevicePtr(context(t))
    device = CUDAdrv.device(devptr.ctx).handle
    CUDArt.CudaArray(CUDArt.CudaPtr(devptr.ptr), size(A), Int(device))
    CuArray(size(A))
end




end
