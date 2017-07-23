module CUBackend

using ..GPUArrays, CUDAnative, StaticArrays, Compat

import CUDAdrv, CUDArt #, CUFFT

import GPUArrays: buffer, create_buffer, acc_mapreduce
import GPUArrays: Context, GPUArray, context, linear_index, gpu_call
import GPUArrays: blas_module, blasbuffer, is_blas_supported, hasblas
import GPUArrays: default_buffer_type, broadcast_index


using CUDAdrv: CuDefaultStream

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

#@compat const GLArrayImg{T, N} = GPUArray{T, N, gl.Texture{T, N}, GLContext}
const CUArray{T, N, B} = GPUArray{T, N, B, CUContext} #, GLArrayImg{T, N}}
const CUArrayBuff{T, N} = CUArray{T, N, CUDAdrv.CuArray{T, N}}


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
# synchronize
function GPUArrays.synchronize{T, N}(x::CUArray{T, N})
    CUDAdrv.synchronize(context(x).ctx) # TODO figure out the diverse ways of synchronization
end

function GPUArrays.free{T, N}(x::CUArray{T, N})
    GPUArrays.synchronize(x)
    Base.finalize(buffer(x))
    nothing
end


default_buffer_type{T, N}(::Type, ::Type{Tuple{T, N}}, ::CUContext) = CUDAdrv.CuArray{T, N}

function (AT::Type{CUArray{T, N, Buffer}}){T, N, Buffer <: CUDAdrv.CuArray}(
        size::NTuple{N, Int};
        context = current_context(),
        kw_args...
    )
    # cuda doesn't allow a size of 0, but since the length of the underlying buffer
    # doesn't matter, with can just initilize it to 0
    buff = prod(size) == 0 ? CUDAdrv.CuArray{T}((1,)) : CUDAdrv.CuArray{T}(size)
    AT(buff, size, context)
end


function Base.copy!{T}(
        dest::Array{T}, d_offset::Integer,
        source::CUDAdrv.CuArray{T}, s_offset::Integer, amount::Integer
    )
    amount == 0 && return dest
    d_offset = d_offset
    s_offset = s_offset - 1
    device_ptr = source.devptr
    sptr = CUDAdrv.DevicePtr{T}(device_ptr.ptr + (sizeof(T) * s_offset), device_ptr.ctx)
    CUDAdrv.Mem.download(Ref(dest, d_offset), sptr, sizeof(T) * (amount))
    dest
end
function Base.copy!{T}(
        dest::CUDAdrv.CuArray{T}, d_offset::Integer,
        source::Array{T}, s_offset::Integer, amount::Integer
    )
    amount == 0 && return dest
    d_offset = d_offset - 1
    s_offset = s_offset
    device_ptr = dest.devptr
    sptr = CUDAdrv.DevicePtr{T}(device_ptr.ptr + (sizeof(T) * d_offset), device_ptr.ctx)
    CUDAdrv.Mem.upload(sptr, Ref(source, s_offset), sizeof(T) * (amount))
    dest
end


function Base.copy!{T}(
        dest::CUDAdrv.CuArray{T}, d_offset::Integer,
        source::CUDAdrv.CuArray{T}, s_offset::Integer, amount::Integer
    )
    d_offset = d_offset - 1
    s_offset = s_offset - 1
    d_ptr = dest.devptr
    s_ptr = source.devptr
    dptr = CUDAdrv.DevicePtr{T}(d_ptr.ptr + (sizeof(T) * d_offset), d_ptr.ctx)
    sptr = CUDAdrv.DevicePtr{T}(s_ptr.ptr + (sizeof(T) * s_offset), s_ptr.ctx)
    CUDAdrv.Mem.transfer(sptr, dptr, sizeof(T) * (amount))
    dest
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

@inline function linear_index(::CUDAnative.CuDeviceArray, state)
    Cuint((blockIdx().x - Cuint(1)) * blockDim().x + threadIdx().x)
end


unpack_cu_array(x) = x
unpack_cu_array(x::Scalar) = unpack_cu_array(getfield(x, 1))
unpack_cu_array{T,N}(x::CUArray{T,N}) = buffer(x)
unpack_cu_array(x::Ref{<:GPUArrays.AbstractAccArray}) = unpack_cu_array(x[])

@inline function call_cuda(A::CUArray, kernel, rest...)
    blocks, thread = thread_blocks_heuristic(A)
    args = map(unpack_cu_array, rest)
    #cu_kernel, rewritten = CUDAnative.rewrite_for_cudanative(kernel, map(typeof, args))
    #println(CUDAnative.@code_typed kernel(args...))
    @cuda (blocks, thread) kernel(args...)
end

# TODO hook up propperly with CUDAdrv... This is a dirty adhoc solution
# to be consistent with the OpenCL backend
immutable CUFunction{T}
    kernel::T
end

if success(`nvcc --version`)
    include("compilation.jl")
    hasnvcc() = true
else
    hasnvcc() = false
    warn("Couldn't find nvcc, please add it to your path.
        This will disable the ability to compile a CUDA kernel from a string"
    )
end

function CUFunction{T, N}(A::CUArray{T, N}, f::Function, args...)
    CUFunction(f) # this is mainly for consistency with OpenCL
end
function CUFunction{T, N}(A::CUArray{T, N}, f::Tuple{String, Symbol}, args...)
    source, name = f
    kernel_name = string(name)
    ctx = context(A)
    kernel = _compile(ctx.device, kernel_name, source, "from string")
    CUFunction(kernel) # this is mainly for consistency with OpenCL
end
function (f::CUFunction{F}){F <: Function, T, N}(A::CUArray{T, N}, args...)
    dims = thread_blocks_heuristic(A)
    return CUDAnative.generated_cuda(
        dims, 0, CuDefaultStream(),
        f.kernel, map(unpack_cu_array, args)...
    )
end
function cu_convert{T, N}(x::CUArray{T, N})
    pointer(buffer(x))
end
cu_convert(x) = x

function (f::CUFunction{F}){F <: CUDAdrv.CuFunction, T, N}(A::CUArray{T, N}, args)
    griddim, blockdim = thread_blocks_heuristic(A)
    CUDAdrv.launch(
        f.kernel, CUDAdrv.CuDim3(griddim...), CUDAdrv.CuDim3(blockdim...), 0, CuDefaultStream(),
        map(cu_convert, args)
    )
end

function gpu_call{T, N}(f::Function, A::CUArray{T, N}, args, globalsize = size(A), localsize = nothing)
    call_cuda(A, f, 0f0, args...)
end
function gpu_call{T, N}(f::Tuple{String, Symbol}, A::CUArray{T, N}, args, globalsize = size(A), localsize = nothing)
    func = CUFunction(A, f, args...)
    # TODO cache
    func(A, args) # TODO pass through local/global size
end

#####################################
# The problem is, that I can't pass Tuple{CuArray} as a type, so I can't
# write a @generated function to unrole the arguments.
# And without unroling of the arguments, GPU codegen will cry!

for i = 0:10
    args = ntuple(x-> Symbol("arg_", x), i)
    fargs2 = ntuple(x-> :(broadcast_index($(args[x]), sz, i)), i)
    @eval begin

        function reduce_kernel{F <: Function, OP <: Function,T1, T2, N}(
                out::AbstractArray{T2,1}, f::F, op::OP, v0::T2,
                A::AbstractArray{T1, N}, $(args...)
            )
            #reduce multiple elements per thread

            i = (CUDAnative.blockIdx().x - UInt32(1)) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
            step = CUDAnative.blockDim().x * CUDAnative.gridDim().x
            sz = Cuint.(size(A))
            result = v0
            while i <= length(A)
                @inbounds result = op(result, f(A[i], $(fargs2...)))
                i += step
            end
            result = reduce_block(result, op, v0)
            if CUDAnative.threadIdx().x == UInt32(1)
                @inbounds out[CUDAnative.blockIdx().x] = result
            end
            return
        end
    end
end


#################################
# Reduction

# TODO do this correctly in CUDAnative/Base
using ColorTypes

function CUDAnative.shfl_down(
        val::Tuple{RGB{Float32}, UInt32}, srclane::Integer, width::Integer = Int32(32)
    )
    (
        RGB{Float32}(
            CUDAnative.shfl_down(val[1].r, srclane, width),
            CUDAnative.shfl_down(val[1].g, srclane, width),
            CUDAnative.shfl_down(val[1].b, srclane, width),
        ),
        CUDAnative.shfl_down(val[2], srclane, width)
    )
end

function reduce_warp{T, F<:Function}(val::T, op::F)
    offset = CUDAnative.warpsize() ÷ UInt32(2)
    while offset > UInt32(0)
        val = op(val, CUDAnative.shfl_down(val, offset))
        offset ÷= UInt32(2)
    end
    return val
end

@inline function reduce_block{T, F <: Function}(val::T, op::F, v0::T)::T
    shared = CUDAnative.@cuStaticSharedMem(T, 32)
    wid  = div(CUDAnative.threadIdx().x - UInt32(1), CUDAnative.warpsize()) + UInt32(1)
    lane = rem(CUDAnative.threadIdx().x - UInt32(1), CUDAnative.warpsize()) + UInt32(1)

     # each warp performs partial reduction
    val = reduce_warp(val, op)

    # write reduced value to shared memory
    if lane == 1
        @inbounds shared[wid] = val
    end
    # wait for all partial reductions
    CUDAnative.sync_threads()
    # read from shared memory only if that warp existed
    @inbounds begin
        val = (threadIdx().x <= fld(CUDAnative.blockDim().x, CUDAnative.warpsize())) ? shared[lane] : v0
    end
    if wid == 1
        # final reduce within first warp
        val = reduce_warp(val, op)
    end
    return val
end



function acc_mapreduce{T, OT, N}(
        f, op, v0::OT, A::CUArray{T, N}, rest::Tuple
    )
    dev = context(A).device
    @assert(CUDAdrv.capability(dev) >= v"3.0", "Current CUDA reduce implementation requires a newer GPU")
    threads = 512
    blocks = min((length(A) + threads - 1) ÷ threads, 1024)
    out = similar(buffer(A), OT, (blocks,))
    args = map(unpack_cu_array, rest)
    # TODO MAKE THIS WORK FOR ALL FUNCTIONS .... v0 is really unfit for parallel reduction
    # since every thread basically needs its own v0
    @cuda (blocks, threads) reduce_kernel(out, f, op, v0, buffer(A), args...)
    # for this size it doesn't seem beneficial to run on gpu?!
    # TODO actually benchmark this theory
    reduce(op, Array(out))
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
if is_blas_supported(:CUBLAS)
    using CUBLAS
    import CUDArt
    #
    # # implement blas interface
    hasblas(::CUContext) = true
    blas_module(::CUContext) = CUBLAS
    function blasbuffer(ctx::CUContext, A)
        buff = buffer(A)
        devptr = pointer(buff)
        device = CUDAdrv.device(devptr.ctx).handle
        CUDArt.CudaArray(CUDArt.CudaPtr(devptr.ptr, ctx.ctx), size(A), Int(device))
    end
end
#
# function convert{T <: CUArray}(t::T, A::CUDArt.CudaArray)
#     ctx = context(t)
#     ptr = DevicePtr(context(t))
#     device = CUDAdrv.device(devptr.ctx).handle
#     CUDArt.CudaArray(CUDArt.CudaPtr(devptr.ptr), size(A), Int(device))
#     CuArray(size(A))
# end
#
#

export CUFunction

end

using .CUBackend
export CUBackend
