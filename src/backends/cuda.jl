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
    function init(;ctx=any_context())
        GPUArrays.make_current(ctx)
        push!(contexts, ctx)
        ctx
    end
end

function create_buffer(ctx::CUContext, A::AbstractArray; kw_args...)
    CUDAdrv.CuArray(A; kw_args...)
end
function Base.unsafe_copy!{T,N}(dest::Array{T,N}, source::CUArray{T,N})
    copy!(dest, buffer(source))
end

function thread_blocks_heuristic(A::AbstractArray)
    thread_blocks_heuristic(size(A))
end
function thread_blocks_heuristic{N}(s::NTuple{N, Int})
    len = prod(s)
    threads = min(len, 1024)
    blocks = floor(Int, len/threads)
    blocks, threads
end

# Base.ind2sub seems to be troublesome
@inline @target ptx function simple_ind2sub(dim::NTuple{1, Int}, i::Int)
    return i
end
@inline @target ptx function simple_ind2sub(dim::NTuple{2, Int}, i::Int)
    return (i % dim[1], div(i, dim[1]))
end
@inline @target ptx function simple_ind2sub(dim::NTuple{3, Int}, i::Int)
    z = div(i, (dim[1] * dim[2]))
    i -= z * dim[1] * dim[2]
    return (i % dim[1], div(i, dim[1]), z)
end

@inline @target ptx function linear_index()
    (blockIdx().x-1) * blockDim().x + threadIdx().x
end
@inline @target ptx function map_eachindex_kernel(A, f, x,y,z)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if length(A) >= i
        f(simple_ind2sub(size(A), Int(i)), A, x,y,z)
    end
    nothing
end
unpack_cu_array(x) = x
unpack_cu_array{T,N}(x::CUArray{T,N}) = buffer(x)

@inline function call_cuda(kernel, A::CUArray, rest...)
    thread, blocks = thread_blocks_heuristic(A)
    args = map(unpack_cu_array, rest)
    @cuda (thread, blocks) kernel(buffer(A), args...)
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

function Base.broadcast(f, A::CUArray, rest::CUArray...)
    B = similar(A)
    map!(f, A, B, rest...)
    B
end

end
