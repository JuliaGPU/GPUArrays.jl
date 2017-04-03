module CLBackend

using Compat
using ..GPUArrays
using OpenCL
using OpenCL: cl

using ..GPUArrays, StaticArrays
#import CLBLAS, CLFFT

import GPUArrays: buffer, create_buffer, acc_broadcast!, acc_mapreduce, mapidx
import GPUArrays: Context, GPUArray, context, broadcast_index, linear_index
import GPUArrays: blasbuffer, blas_module, is_blas_supported, is_fft_supported
import GPUArrays: synchronize, hasblas, LocalMemory

using Transpiler
using Transpiler: CLTranspiler
import Transpiler.CLTranspiler.cli
import Transpiler.CLTranspiler.CLFunction
import Transpiler.CLTranspiler.cli.get_global_id

immutable CLContext <: Context
    device::cl.Device
    context::cl.Context
    queue::cl.CmdQueue
    function CLContext(device_type = nothing)
        device = if device_type == nothing
            devlist = cl.devices(:gpu)
            dev = if isempty(devlist)
                devlist = cl.devices(:cpu)
                if isempty(devlist)
                    error("no device found to be supporting opencl")
                else
                    first(devlist)
                end
            else
                first(devlist)
            end
            dev
        else
            # if device type supplied by user, assume it's actually existant!
            devlist = cl.devices(device_type)
            if isempty(devlist)
                error("Can't find OpenCL device for $device_type")
            end
            first(devlist)
        end
        ctx = cl.Context(device)
        queue = cl.CmdQueue(ctx)
        new(device, ctx, queue)
    end
end
function Base.show(io::IO, ctx::CLContext)
    name = replace(ctx.device[:name], r"\s+", " ")
    print(io, "CLContext: $name")
end

global init, all_contexts, current_context
let contexts = CLContext[]
    all_contexts() = copy(contexts)::Vector{CLContext}
    current_context() = last(contexts)::CLContext
    function init(;device_type = nothing, ctx = nothing)
        context = if ctx == nothing
            if isempty(contexts)
                CLContext(device_type)
            else
                current_context()
            end
        else
            ctx
        end
        GPUArrays.make_current(context)
        push!(contexts, context)
        context
    end
end

@compat const CLArray{T, N} = GPUArray{T, N, cl.Buffer{T}, CLContext}

#synchronize

function synchronize{T, N}(x::CLArray{T, N})
    cl.finish(context(x).queue) # TODO figure out the diverse ways of synchronization
end
# Constructor
function Base.copy!{T, N}(dest::Array{T, N}, source::CLArray{T, N})
    q = context(source).queue
    cl.finish(q)
    copy!(q, dest, buffer(source))
end

function Base.copy!{T, N}(dest::CLArray{T, N}, source::Array{T, N})
    q = context(dest).queue
    cl.finish(q)
    copy!(q, buffer(dest), source)
end

# copy the contents of a buffer into another buffer
function Base.copy!{T, N}(dst::CLArray{T, N}, src::CLArray{T, N})
    q = context(dst).queue
    cl.finish(q)
    copy!(q, buffer(dst), buffer(src))
end


function create_buffer{T, N}(ctx::CLContext, A::AbstractArray{T, N}, flag = :rw)
    cl.Buffer(T, ctx.context, (flag, :copy), hostbuf = A)
end
function create_buffer{T, N}(
        ctx::CLContext, ::Type{T}, sz::NTuple{N, Int};
        flag = :rw, kw_args...
    )
    cl.Buffer(T, ctx.context, flag, prod(sz))
end

function Base.similar{T, N, N2, ET}(x::CLArray{T, N}, ::Type{ET}, sz::Tuple{Vararg{Int, N2}}; kw_args...)
    ctx = context(x)
    b = create_buffer(ctx, ET, sz; kw_args...)
    GPUArray{ET, N2, typeof(b), typeof(ctx)}(b, sz, ctx)
end
# The implementation of prod in base doesn't play very well with current
# transpiler. TODO figure out what Core._apply maps to!
_prod{T}(x::NTuple{1, T}) = x[1]
_prod{T}(x::NTuple{2, T}) = x[1] * x[2]
_prod{T}(x::NTuple{3, T}) = x[1] * x[2] * x[3]

linear_index(::cli.CLArray) = get_global_id(0) + 1

###########################################################
# Broadcast
for i = 0:10
    args = ntuple(x-> Symbol("arg_", x), i)
    fargs = ntuple(x-> :(broadcast_index($(args[x]), sz, i)), i)
    @eval begin
        function broadcast_kernel(A, f, sz, $(args...))
            i = get_global_id(0) + 1
            A[i] = f($(fargs...))
            return
        end
        function mapidx_kernel{F}(A, f::F, $(args...))
            i = get_global_id(0) + 1
            f(i, A, $(args...))
            return
        end
    end
end

# extend the private interface for the compilation types
CLTranspiler._to_cl_types{T, N}(arg::CLArray{T, N}) = cli.CLArray{T, N}
CLTranspiler._to_cl_types{T}(x::LocalMemory{T}) = cli.LocalMemory{T}

CLTranspiler.cl_convert{T, N}(x::CLArray{T, N}) = buffer(x)
CLTranspiler.cl_convert{T}(x::LocalMemory{T}) = cl.LocalMem(T, x.size)

function acc_broadcast!{F <: Function, T, N}(f::F, A::CLArray{T, N}, args::Tuple)
    ctx = context(A)
    sz = map(Int32, size(A))
    q = ctx.queue
    cl.finish(q)
    clfunc = CLFunction(broadcast_kernel, (A, f, sz, args...), q)
    clfunc((A, f, sz, args...), length(A))
    A
end


function mapidx{F <: Function, N, T, N2}(f::F, A::CLArray{T, N2}, args::NTuple{N, Any})
    ctx = context(A)
    q = ctx.queue
    cl.finish(q)
    cl_args = (A, f, args...)
    clfunc = CLFunction(mapidx_kernel, cl_args, q)
    clfunc(cl_args, length(A))
end

function CLFunction{T, N}(A::CLArray{T, N}, f, args...)
    ctx = context(A)
    CLFunction(f, args, ctx.queue)
end
function (clfunc::CLFunction{T}){T, T2, N}(A::CLArray{T2, N}, args...)
    # TODO use better heuristic
    clfunc(args, length(A))
end
###################
# Blase interface
if is_blas_supported(:CLBLAS)
    import CLBLAS
    blas_module(::CLContext) = CLBLAS
    hasblas(::CLContext) = true
    function blasbuffer(::CLContext, A)
        buff = buffer(A)
        # LOL! TODO don't have CLArray in OpenCL/CLBLAS
        cl.CLArray(buff, context(A).queue, size(A))
    end
end

###################
# FFT interface
if is_fft_supported(:CLFFT)
    include("fft.jl")
end

export CLFunction, cli

end #CLBackend

using .CLBackend
export CLBackend
