module CUBackend

using ..GPUArrays
import GPUArrays: buffer, create_buffer, Context, GPUArray, context
using CUDArt, CUDAnative
import CUDAdrv
const cu = CUDArt.rt

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
    CUDAdrv.CuArray(T, sz; kw_args...)
end
function Base.unsafe_copy!{T,N}(dest::Array{T,N}, source::CUArray{T,N})
    copy!(dest, buffer(source))
end
function Base.unsafe_copy!{T,N}(dest::CUArray{T,N}, source::Array{T,N})
    copy!(buffer(dest), source)
end
function Base.similar{T,N}(A::CUDAdrv.CuArray{T,N})
    CUDAdrv.CuArray(T, size(A))
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

@inline @target ptx function linear_index()
    (blockIdx().x-1) * blockDim().x + threadIdx().x
end
@inline @target ptx function map_eachindex_kernel{N}(A, f, args::Vararg{N})
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

function map_eachindex(f, A::CUArray, rest...)
    call_cuda(map_eachindex_kernel, A, f, rest...)
end

@target ptx function map_kernel(a, f, b)
    i = linear_index()
    if length(a) >= i
        @inbounds a[i] = f(b[i])
    end
    nothing
end
@target ptx function map_kernel(a, f, b, c)
    i = linear_index()
    if length(a) >= i
        @inbounds a[i] = f(b[i], c[i])
    end
    nothing
end
@target ptx function map_kernel(a, f, b, c, d)
    i = linear_index()
    if length(a) >= i
        @inbounds a[i] = f(b[i], c[i], d[i])
    end
    nothing
end

function Base.map!(f::Function, A::CUArray, rest::CUArray...)
    call_cuda(map_kernel, A, f, rest...)
    A
end

function Base.map(f, A::CUArray, rest::CUArray...)
    B = similar(A)
    map!(f, A, B, rest...)
    B
end


#Broadcast

function Base.broadcast(f::Function, A::CUArray, rest::Union{CUArray, Number}...)
    result = similar(A)
    broadcast!(f, result, A, rest...)
    result
end


@target ptx @generated function broadcast_index{T, N}(arg::AbstractArray{T,N}, shape, idx)
    idx = ntuple(i->:(ifelse(s[$i] < shape[$i], 1, idx[$i])), N)
    expr = quote
        s = size(arg)::NTuple{N, Int}
        @inbounds i = CartesianIndex{N}(($(idx...),)::NTuple{N, Int})
        @inbounds return arg[i]::T
    end
end
@target ptx function broadcast_index{T}(arg::T, shape, idx)
    arg::T
end

# Crude workaround for splat not working all the time right now
for i=0:6
    args = ntuple(x-> Symbol("arg_", x), i)
    fargs = ntuple(x-> :(broadcast_index($(args[x]), sz, idx)), i)
    expr = quote
        @target ptx function broadcast_kernel(A, f, $(args...))
            i = Int((blockIdx().x-1) * blockDim().x + threadIdx().x)
            @inbounds if i <= length(A)
                sz = size(A)
                idx = CartesianIndex(ind2sub(sz, Int(i)))
                A[idx] = f($(fargs...))
            end
            nothing
        end
    end
    eval(expr)
end


function Base.broadcast!(f::Function, A::CUArray)
    cu_broadcast!(f, A)
end
function Base.broadcast!(f::Function, A::CUArray, B::Number)
    cu_broadcast!(f, A, B)
end
function Base.broadcast!(f::Function, A::CUArray, B::AbstractArray)
    cu_broadcast!(f, A, B)
end
function Base.broadcast!(f::Function, A::CUArray, B::CUArray, C::CUArray)
    cu_broadcast!(f, A, B, C)
end
function Base.broadcast!(f::Function, A::CUArray, B::CUArray, C::Number)
    cu_broadcast!(f, A, B, C)
end
function Base.broadcast!(f::Function, A::CUArray, B::AbstractArray, C::AbstractArray, D::AbstractArray, E::Number, F::Number)
    cu_broadcast!(f, A, B, C, D, E, F)
end
function Base.broadcast!(f::Function, A::CUArray, B::AbstractArray, C::AbstractArray, D::AbstractArray)
    cu_broadcast!(f, A, B, C, D)
end
function cu_broadcast!{N}(f::Function, A::CUArray, As::Vararg{Any, N})
    N > 6 && error("Does only work for maximum for args atm.")
    shape = indices(A)
    Base.Broadcast.check_broadcast_shape(shape, As...)
    call_cuda(broadcast_kernel, A, f, As...)
end







# reduction
@target ptx function reduce_warp{T,F<:Function}(val::T, op::F)
    offset = Int(warpsize) ÷ 2
    while offset > 0
        val = op(val, shfl_down(val, offset))::T
        offset ÷= 2
    end
    return val::T
end

@target ptx function reduce_block{T,F<:Function}(val::T, op::F)
    shared = @cuStaticSharedMem(T, 32)

    wid, lane = fldmod1(threadIdx().x, warpsize)

    val = reduce_warp(val, op)::T

    if lane == 1
        @inbounds shared[Int(wid)] = val
    end

    sync_threads()

    # read from shared memory only if that warp existed
    @inbounds val = ((threadIdx().x <= fld(blockDim().x, warpsize)) ? shared[Int(lane)] : zero(T))::T

    if wid == 1
        # final reduce within first warp
        val = reduce_warp(val, op)::T
    end

    return val::T
end
@target ptx function reduce_kernel{F<:Function,OP<:Function,T1,T2,N}(
        A::AbstractArray{T1,N}, out::AbstractArray{T2,1}, f::F, op::OP
    )
    result = zero(T2)
    #reduce multiple elements per thread
    i = Int((blockIdx().x-Int32(1)) * blockDim().x + threadIdx().x)
    while i <= length(A)
       @inbounds result = op(result, f(A[i]))
       i += blockDim().x * gridDim().x
    end
    result = reduce_block(result, op)
    if (threadIdx().x == 1)
        @inbounds out[Int(blockIdx().x)] = result;
    end
    nothing
end


function Base.mapreduce{T,N}(f::Function, op::Function, A::CUArray{T,N})
    threads = 512
    blocks = min((length(A) + threads - 1) ÷ threads, 1024)
    out = CUDAdrv.CuArray(Base.r_promote_type(op, T), (blocks,))
    @cuda (blocks, threads) reduce_kernel(buffer(A), out, f, op)
    mapreduce(f, op, Array(out)) # for this size it doesn't seem beneficial to run on gpu?!
end

# @target ptx funtion reduce_kernel{F<:Function, T, N, BlockSize}(
#         op::F, g_idata::AbstractArray{T,N}, g_odata::AbstractArray{T,N},
#         n, Val{BlockSize}
#     )
#     #extern __shared__ int sdata[];
#     sdata = @cuStaticSharedMem(T, BlockSize)
#     tid = threadIdx().x;
#     i = blockIdx().x*(blockSize*2) + tid;
#     gridSize = blockSize*2*gridDim().x;
#     sdata[tid] = 0;
#     while (i < n)
#         sdata[tid] = op(sdata[tid], op(g_idata[i], g_idata[i+blockSize]))
#         i += gridSize
#     end
#     sync_threads()
#     if (blockSize >= 512
#         (tid < 256) && (sdata[tid] = op(sdata[tid], sdata[tid + 256]))
#         sync_threads()
#     end
#     if (blockSize >= 256
#         (tid < 128) && (sdata[tid] = op(sdata[tid], sdata[tid + 128]))
#         sync_threads()
#     end
#     if (blockSize >= 128
#         (tid < 64) && (sdata[tid] = op(sdata[tid], sdata[tid + 64]))
#         sync_threads()
#     end
#     if tid < 32
#         (blockSize >= 64) && (sdata[tid] = op(sdata[tid], sdata[tid + 32]))
#         (blockSize >= 32) && (sdata[tid] = op(sdata[tid], sdata[tid + 16]))
#         (blockSize >= 16) && (sdata[tid] = op(sdata[tid], sdata[tid + 8]))
#         (blockSize >= 8) && (sdata[tid] = op(sdata[tid], sdata[tid + 4]))
#         (blockSize >= 4) && (sdata[tid] = op(sdata[tid], sdata[tid + 2]))
#         (blockSize >= 2) && (sdata[tid] = op(sdata[tid], sdata[tid + 1]))
#     end
#     (tid == 0) && (g_odata[blockIdx.x] = sdata[0])
#     nothing
# end
#


end
