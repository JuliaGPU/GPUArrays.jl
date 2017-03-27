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

using Transpiler
using Transpiler: CLTranspiler
import Transpiler.CLTranspiler.cli
import Transpiler.CLTranspiler.CLFunction
import Transpiler.CLTranspiler.cli.get_global_id

immutable CLContext <: Context
    device::cl.Device
    context::cl.Context
    queue::cl.CmdQueue
    function CLContext(device_type = :gpu)
        device = first(cl.devices(device_type))
        ctx    = cl.Context(device)
        queue  = cl.CmdQueue(ctx)
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
    function init(;typ = :gpu, ctx = CLContext(typ))
        GPUArrays.make_current(ctx)
        push!(contexts, ctx)
        ctx
    end
end

@compat const CLArray{T, N} = GPUArray{T, N, cl.Buffer{T}, CLContext}

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
function create_buffer{T, N}(ctx::CLContext, A::AbstractArray{T, N}, flag = :rw)
    cl.Buffer(T, ctx.context, (flag, :copy), hostbuf = A)
end
function create_buffer{T, N}(
        ctx::CLContext, ::Type{T}, sz::NTuple{N, Int};
        flag = :rw, kw_args...
    )
    cl.Buffer(T, ctx.context, flag, prod(sz))
end

function Base.similar{T, N, ET}(x::CLArray{T, N}, ::Type{ET}, sz::NTuple{N, Int}; kw_args...)
    ctx = context(x)
    b = create_buffer(ctx, ET, sz; kw_args...)
    GPUArray{ET, N, typeof(b), typeof(ctx)}(b, sz, ctx)
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
function CLTranspiler._to_cl_types{T, N}(arg::CLArray{T, N})
    return cli.CLArray{T, N}
end
CLTranspiler.cl_convert{T, N}(x::CLArray{T, N}) = buffer(x)

function acc_broadcast!{F <: Function, T, N}(f::F, A::CLArray{T, N}, args::Tuple)
    ctx = context(A)
    sz = map(Int32, size(A))
    q = ctx.queue
    cl.finish(q)
    clfunc = CLFunction(broadcast_kernel, (A, f, sz, args...), q)
    clfunc((A, f, sz, args...), length(A))
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
