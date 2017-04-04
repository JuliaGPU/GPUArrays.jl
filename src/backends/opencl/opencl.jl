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
import GPUArrays: synchronize, hasblas, LocalMemory, AccMatrix, AccVector, gpu_call

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
            # TODO I'm very sure, that we can tile this better
            # +1, Julia is 1 based indexed, and we (currently awkwardly) try to preserve this semantic
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

function gpu_call{T, N}(A::CLArray{T, N}, f, args, globalsize = length(A), localsize = nothing)
    ctx = GPUArrays.context(A)
    clfunc = CLFunction(f, args, ctx.queue)
    clfunc(args, globalsize, localsize)
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

using Transpiler.CLTranspiler: cli
using Transpiler.CLTranspiler.cli: get_local_id, get_global_id, barrier
using Transpiler.CLTranspiler.cli: CLK_LOCAL_MEM_FENCE, get_group_id
using Transpiler.CLTranspiler.cli: get_local_size, get_global_size

using OpenCL

# TODO generalize to CUDAnative
function transpose_kernel!{BLOCK}(
        At, A, width, height, A_local, ::Val{BLOCK}
    )
    base_idx_a = get_group_id(0) * BLOCK + get_group_id(1) * (BLOCK * width)
    base_idx_a_t = get_group_id(1) * BLOCK + get_group_id(0) * (BLOCK * height)

    glob_idx_a = base_idx_a + get_local_id(0) + width * get_local_id(1)
    glob_idx_a_t = base_idx_a_t + get_local_id(0) + height * get_local_id(1)

    A_local[get_local_id(1) * BLOCK + get_local_id(0) + 1] = A[glob_idx_a + 1]

    barrier(CLK_LOCAL_MEM_FENCE)

    At[glob_idx_a_t + 1] = A_local[get_local_id(0) * BLOCK + get_local_id(1) + 1]
    return
end

function Base.transpose!{T}(At::CLArray{T, 2}, A::CLArray{T, 2})
    ctx = context(A)
    block_size = cl.max_block_size(ctx.queue, size(A, 1), size(A, 2))
    outsize = map(Int32, size(At))
    lmem = GPUArrays.LocalMemory{Float32}(block_size * (block_size + 1))
    args = (At, A, outsize..., lmem, Val{block_size}())
    func = GPUArrays.CLFunction(At, transpose_kernel!, args...)
    func(args, outsize, (block_size, block_size))
    At
end



for i = 0:10
    args = ntuple(x-> Symbol("arg_", x), i)
    fargs = ntuple(x-> :(broadcast_index($(args[x]), length, global_index)), i)
    @eval begin
        function reduce_kernel(f, op, v0, A, tmp_local, length, result, $(args...))
            global_index = get_global_id(0) + 1
            local_v0 = v0
            # Loop sequentially over chunks of input vector
            while (global_index <= length)
                element = f(A[global_index], $(fargs...))
                local_v0 = op(local_v0, element)
                global_index += get_global_size(0)
            end

            # Perform parallel reduction
            local_index = get_local_id(0)
            tmp_local[local_index + 1] = local_v0
            barrier(CLK_LOCAL_MEM_FENCE)
            offset = div(get_local_size(0), 2)
            while offset > 0
                if (local_index < offset)
                    other = tmp_local[local_index + offset + 1]
                    mine = tmp_local[local_index + 1];
                    tmp_local[local_index + 1] = op(mine, other)
                end
                barrier(CLK_LOCAL_MEM_FENCE)
                offset = div(offset, 2)
            end
            if local_index == 0
                result[get_group_id(0) + 1] = tmp_local[1]
            end
            return
        end
    end
end

function acc_mapreduce{T, OT, N}(
        f, op, v0::OT, A::CLArray{T, N}, rest::Tuple
    )
    dev = context(A).device
    block_size = 16
    group_size = ceil(Int, length(A) / block_size)
    out = similar(A, OT, (group_size,))
    fill!(out, v0)
    lmem = GPUArrays.LocalMemory{OT}(block_size)
    args = (f, op, v0, A, lmem, Int32(length(A)), out, rest...)
    func = GPUArrays.CLFunction(A, reduce_kernel, args...)
    func(args, group_size * block_size, (block_size,))
    reduce(op, Array(out))
end


export CLFunction, cli


end #CLBackend

using .CLBackend
export CLBackend
