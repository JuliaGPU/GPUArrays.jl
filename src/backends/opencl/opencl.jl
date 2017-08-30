module CLBackend

using ..GPUArrays
using OpenCL
using OpenCL: cl

using ..GPUArrays, StaticArrays

import GPUArrays: buffer, create_buffer, acc_mapreduce, mapidx
import GPUArrays: Context, GPUArray, context, linear_index, free
import GPUArrays: blasbuffer, blas_module, is_blas_supported, is_fft_supported
import GPUArrays: synchronize, hasblas, LocalMemory, AccMatrix, AccVector, gpu_call
import GPUArrays: default_buffer_type, broadcast_index, unsafe_reinterpret

using Transpiler
import Transpiler: cli, cli.get_global_id

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

const CLArray{T, N} = GPUArray{T, N, B, CLContext} where B <: cl.Buffer

include("compilation.jl")


#synchronize
function synchronize{T, N}(x::CLArray{T, N})
    cl.finish(context(x).queue) # TODO figure out the diverse ways of synchronization
end

function free{T, N}(x::CLArray{T, N})
    synchronize(x)
    mem = buffer(x)
    finalize(mem)
    nothing
end

function linear_index(::cli.CLArray, state)
    (get_local_size(0)*get_group_id(0) + get_local_id(0)) + Cuint(1)
end

function cl_readbuffer(q, buf, dev_offset, hostref, nbytes)
    n_evts  = UInt(0)
    evt_ids = C_NULL
    ret_evt = Ref{cl.CL_event}()
    cl.@check cl.api.clEnqueueReadBuffer(
        q.id, buf.id, cl.cl_bool(true),
        dev_offset, nbytes, hostref,
        n_evts, evt_ids, ret_evt
    )
end
function cl_writebuffer(q, buf, dev_offset, hostref, nbytes)
    n_evts  = UInt(0)
    evt_ids = C_NULL
    ret_evt = Ref{cl.CL_event}()
    cl.@check cl.api.clEnqueueWriteBuffer(
        q.id, buf.id, cl.cl_bool(true),
        dev_offset, nbytes, hostref,
        n_evts, evt_ids, ret_evt
    )
end

function Base.copy!{T}(
        dest::Array{T}, d_offset::Integer,
        source::CLArray{T}, s_offset::Integer, amount::Integer
    )
    amount == 0 && return dest
    s_offset = (s_offset - 1) * sizeof(T)
    q = context(source).queue
    cl.finish(q)
    cl_readbuffer(q, buffer(source), unsigned(s_offset), Ref(dest, d_offset), amount * sizeof(T))
    dest
end

function Base.copy!{T}(
        dest::CLArray{T}, d_offset::Integer,
        source::Array{T}, s_offset::Integer, amount::Integer
    )
    amount == 0 && return dest
    q = context(dest).queue
    cl.finish(q)
    d_offset = (d_offset - 1) * sizeof(T)
    buff = buffer(dest)
    clT = eltype(buff)
    # element type has different padding from cl type in julia
    # for fixedsize arrays we use vload/vstore, so we can use it packed
    if sizeof(clT) != sizeof(T) && !Transpiler.is_fixedsize_array(T)
        # TODO only convert the range in the offset, or maybe convert elements and directly upload?
        # depends a bit on the overhead ov cl_writebuffer
        source = map(cl.packed_convert, source)
    end
    cl_writebuffer(q, buff, unsigned(d_offset), Ref(source, s_offset), amount * sizeof(clT))
    dest
end


function Base.copy!{T}(
        dest::CLArray{T}, d_offset::Integer,
        src::CLArray{T}, s_offset::Integer, amount::Integer
    )
    amount == 0 && return dest
    q = context(dest).queue
    cl.finish(q)
    d_offset = (d_offset - 1) * sizeof(T)
    s_offset = (s_offset - 1) * sizeof(T)
    cl.enqueue_copy_buffer(
        q, buffer(src), buffer(dest),
        Csize_t(amount * sizeof(T)), Csize_t(s_offset), Csize_t(d_offset),
        nothing
    )
    dest
end


default_buffer_type{T, N}(::Type, ::Type{Tuple{T, N}}, ::CLContext) = cl.Buffer{T}



function (AT::Type{<: CLArray{T, N}})(
        size::NTuple{N, Int};
        context = current_context(),
        flag = :rw, kw_args...
    ) where {T, N}
    ctx = context.context
    # element type has different padding from cl type in julia
    # for fixedsize arrays we use vload/vstore, so we can use it packed
    clT = if !Transpiler.is_fixedsize_array(T)
        cl.packed_convert(T)
    else
        T
    end
    buffsize = prod(size)
    buff = buffsize == 0 ? cl.Buffer(clT, ctx, flag, 1) : cl.Buffer(clT, ctx, flag, buffsize)
    GPUArray{T, N, typeof(buff), typeof(context)}(buff, size, context)
end

function unsafe_reinterpret(::Type{T}, A::CLArray{ET}, dims::Tuple) where {T, ET}
    buff = buffer(A)
    newbuff = cl.Buffer{T}(buff.id, true, prod(dims))
    ctx = context(A)
    GPUArray{T, length(dims), typeof(newbuff), typeof(ctx)}(newbuff, dims, ctx)
end




function CLFunction{T, N}(A::CLArray{T, N}, f, args...)
    ctx = context(A)
    CLFunction(f, args, ctx.queue)
end
function (clfunc::CLFunction{T}){T, T2, N}(A::CLArray{T2, N}, args...)
    # TODO use better heuristic
    clfunc(args, length(A))
end

function thread_blocks_heuristic(len::Integer)
    threads = min(len, 256)
    blocks = ceil(Int, len/threads)
    blocks = blocks * threads
    blocks, threads
end


function gpu_call{T, N}(f, A::CLArray{T, N}, args, globalsize = length(A), localsize = nothing)
    ctx = GPUArrays.context(A)
    _args = if !isa(f, Tuple{String, Symbol})
        (0f0, args...)# include "state"
    else
        args
    end
    clfunc = CLFunction(f, _args, ctx.queue)
    blocks, thread = thread_blocks_heuristic(globalsize)
    clfunc(_args, blocks, thread)
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

using Transpiler.cli: get_local_id, get_global_id, barrier,  CLK_LOCAL_MEM_FENCE
using Transpiler.cli: get_local_size, get_global_size, get_group_id

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
            global_index = get_global_id(0) + Cuint(1)
            local_v0 = v0
            # Loop sequentially over chunks of input vector
            while (global_index <= length)
                element = f(A[global_index], $(fargs...))
                local_v0 = op(local_v0, element)
                global_index += get_global_size(0)
            end

            # Perform parallel reduction
            local_index = get_local_id(0)
            tmp_local[local_index + Cuint(1)] = local_v0
            barrier(CLK_LOCAL_MEM_FENCE)
            offset = div(get_local_size(0), Cuint(2))
            while offset > 0
                if (local_index < offset)
                    other = tmp_local[local_index + offset + Cuint(1)]
                    mine = tmp_local[local_index + Cuint(1)];
                    tmp_local[local_index + Cuint(1)] = op(mine, other)
                end
                barrier(CLK_LOCAL_MEM_FENCE)
                offset = div(offset, Cuint(2))
            end
            if local_index == Cuint(0)
                result[get_group_id(0) + Cuint(1)] = tmp_local[1]
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
    args = (f, op, v0, A, lmem, Cuint(length(A)), out, rest...)

    func = GPUArrays.CLFunction(A, reduce_kernel, args...)
    func(args, group_size * block_size, (block_size,))
    x = reduce(op, Array(out))
    x
end


export CLFunction, cli


end #CLBackend

using .CLBackend
export CLBackend
