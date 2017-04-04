
module CUBackend

using ..GPUArrays, CUDAnative, StaticArrays, Compat

import CUDAdrv, CUDArt #, CUFFT

import GPUArrays: buffer, create_buffer, acc_broadcast!, acc_mapreduce, mapidx
import GPUArrays: Context, GPUArray, context, broadcast_index, linear_index, gpu_call
import GPUArrays: synchronize

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
@compat const CUArray{T, N, B} = GPUArray{T, N, B, CUContext} #, GLArrayImg{T, N}}
@compat const CUArrayBuff{T, N} = CUArray{T, N, CUDAdrv.CuArray{T, N}}


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
function synchronize{T, N}(x::CUArray{T, N})
    CUDAdrv.synchronize(context(x).ctx) # TODO figure out the diverse ways of synchronization
end

function create_buffer{T, N}(ctx::CUContext, ::Type{T}, sz::NTuple{N, Int}; kw_args...)
    CUDAdrv.CuArray{T}(sz)
end
function Base.copy!{T,N}(dest::Array{T,N}, source::CUArray{T,N})
    copy!(dest, buffer(source))
end
function Base.copy!{T,N}(dest::CUArray{T,N}, source::Array{T,N})
    copy!(buffer(dest), source)
end
function Base.copy!{T,N}(dest::CUArray{T,N}, source::CUArray{T,N})
    copy!(buffer(dest), buffer(source))
end

function Base.similar{T, N, ET}(x::CUArray{T, N}, ::Type{ET}, sz::NTuple{N, Int}; kw_args...)
    ctx = context(x)
    b = create_buffer(ctx, ET, sz; kw_args...)
    GPUArray{ET, N, typeof(b), typeof(ctx)}(b, sz, ctx)
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

@inline function linear_index(::CUDAnative.CuDeviceArray)
    (blockIdx().x - UInt32(1)) * blockDim().x + threadIdx().x
end


unpack_cu_array(x) = x
unpack_cu_array(x::Scalar) = unpack_cu_array(getfield(x, 1))
unpack_cu_array{T,N}(x::CUArray{T,N}) = buffer(x)

@inline function call_cuda(A::CUArray, kernel, rest...)
    blocks, thread = thread_blocks_heuristic(A)
    args = map(unpack_cu_array, rest)
    @cuda (blocks, thread) kernel(args...)
end

# TODO hook up propperly with CUDAdrv... This is a dirty adhoc solution
# to be consistent with the OpenCL backend
immutable CUFunction{T}
    kernel::T
end
# TODO find a future for the kernel string compilation part
compile_lib = Pkg.dir("CUDAdrv", "examples", "compilation", "library.jl")
has_nvcc = try
    success(`nvcc --version`)
catch
    false
end
if isfile(compile_lib) && has_nvcc
    include(compile_lib)
    hasnvcc() = true
else
    hasnvcc() = false
    if !has_nvcc
        warn("Couldn't find nvcc, please add it to your path.
            This will disable the ability to compile a CUDA kernel from a string"
        )
    end
    if !isfile(compile_lib)
        warn("Couldn't find cuda compilation lib in default location.
        This will disable the ability to compile a CUDA kernel from a string
        To fix, install CUDAdrv in default location."
        )
    end
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

function gpu_call{T, N}(A::CUArray{T, N}, f::Function, args, globalsize = size(A), localsize = nothing)
    call_cuda(A, f, args...)
end
function gpu_call{T, N}(A::CUArray{T, N}, f::Tuple{String, Symbol}, args, globalsize = size(A), localsize = nothing)
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
    fargs = ntuple(x-> :(broadcast_index(which[$x], $(args[x]), sz, i)), i)
    fargs2 = ntuple(x-> :(broadcast_index($(args[x]), sz, i)), i)
    @eval begin
        function broadcast_kernel(A, f, sz, which, $(args...))
            i = linear_index(A)
            @inbounds if i <= length(A)
                A[i] = f($(fargs...))
            end
            nothing
        end
        function mapidx_kernel{F}(A, f::F, $(args...))
            i = linear_index(A)
            if i <= length(A)
                f(i, A, $(args...))
            end
            nothing
        end
        function reduce_kernel{F <: Function, OP <: Function,T1, T2, N}(
                out::AbstractArray{T2,1}, f::F, op::OP, v0::T2,
                A::AbstractArray{T1, N}, $(args...)
            )
            #reduce multiple elements per thread

            i = (blockIdx().x - UInt32(1)) * blockDim().x + threadIdx().x
            step = blockDim().x * gridDim().x
            sz = size(A)
            result = v0
            while i <= length(A)
                @inbounds result = op(result, f(A[i], $(fargs2...)))
                i += step
            end
            result = reduce_block(result, op, v0)
            if threadIdx().x == UInt32(1)
                @inbounds out[blockIdx().x] = result
            end
            return
        end
    end
end

function acc_broadcast!{F <: Function, N}(f::F, A::CUArray, args::NTuple{N, Any})
    which = map(args) do arg
        if isa(arg, AbstractArray) && !isa(arg, Scalar)
            return Val{true}()
        end
        Val{false}()
    end
    call_cuda(A, broadcast_kernel, A, f, map(UInt32, size(A)), which, args...)
end
function mapidx{F <: Function, N, T, N2}(f::F, A::CUArray{T, N2}, args::NTuple{N, Any})
    call_cuda(A, mapidx_kernel, A, f, args...)
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
            shfl_down(val[1].r, srclane, width),
            shfl_down(val[1].g, srclane, width),
            shfl_down(val[1].b, srclane, width),
        ),
        shfl_down(val[2], srclane, width)
    )
end

function reduce_warp{T, F<:Function}(val::T, op::F)
    offset = CUDAnative.warpsize() ÷ UInt32(2)
    while offset > UInt32(0)
        val = op(val, shfl_down(val, offset))
        offset ÷= UInt32(2)
    end
    return val
end

@inline function reduce_block{T, F <: Function}(val::T, op::F, v0::T)::T
    shared = @cuStaticSharedMem(T, 32)
    wid  = div(threadIdx().x - UInt32(1), CUDAnative.warpsize()) + UInt32(1)
    lane = rem(threadIdx().x - UInt32(1), CUDAnative.warpsize()) + UInt32(1)

     # each warp performs partial reduction
    val = reduce_warp(val, op)

    # write reduced value to shared memory
    if lane == 1
        @inbounds shared[wid] = val
    end
    # wait for all partial reductions
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
# using CUBLAS
# import CUDArt
#
# # implement blas interface
# blas_module(::CUContext) = CUBLAS
# function blasbuffer(ctx::CUContext, A)
#     buff = buffer(A)
#     devptr = pointer(buff)
#     device = CUDAdrv.device(devptr.ctx).handle
#     CUDArt.CudaArray(CUDArt.CudaPtr(devptr.ptr), size(A), Int(device))
# end
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
