module CLBackend

using Compat
using ..GPUArrays
using OpenCL
using OpenCL: cl

using ..GPUArrays, StaticArrays
#import CLBLAS, CLFFT

import GPUArrays: buffer, create_buffer, acc_mapreduce, mapidx
import GPUArrays: Context, GPUArray, context, linear_index, free
import GPUArrays: blasbuffer, blas_module, is_blas_supported, is_fft_supported
import GPUArrays: synchronize, hasblas, LocalMemory, AccMatrix, AccVector, gpu_call
import GPUArrays: default_buffer_type, broadcast_index

using Transpiler
import Transpiler: cli, CLFunction, cli.get_global_id

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

linear_index(::cli.CLArray, state) = get_global_id(0) + Cuint(1)


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
    if clT != T # element type has different padding from cl type in julia
        # TODO only convert the range in the offset, or maybe convert elements and directly upload?
        # depends a bit on the overhead ov cl_writebuffer
        source = map(source) do x
            clx, off = aligned_convert(x)
            clx
        end
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

is_cl_vector(x::T) where T = _is_cl_vector(T)
is_cl_vector(x::Type{T}) where T = _is_cl_vector(T)
_is_cl_vector(x) = false
_is_cl_vector(x::Type{NTuple{N, T}}) where {N, T} = is_cl_number(T) && N in (2, 3, 4, 8, 16)
is_cl_number(x::Type{T}) where T = _is_cl_number(T)
is_cl_number(x::T) where T = _is_cl_number(T)
_is_cl_number(x) = false
function _is_cl_number(::Type{<: Union{
        Int64, Int32, Int16, Int8,
        UInt64, UInt32, UInt16, UInt8,
        Float64, Float32, Float16
    }})
    true
end
is_cl_inbuild{T}(x::T) = is_cl_vector(x) || is_cl_number(x)


struct Pad{N}
    val::NTuple{N, Int8}
    Pad{N}() where N = new{N}(ntuple(i-> Int8(0), Val{N}))
end
Base.isempty(::Type{Pad{N}}) where N = (N == 0)
Base.isempty(::Pad{N}) where N = N == 0
function walk(f, T, args...)
    if nfields(T) > 0
        for field in fieldnames(T)
            x = isa(T, DataType) ? fieldtype(T, field) : getfield(T, field)
            f(x, args...)
        end
    else
        f(T, args...)
    end
    return
end


inbuild_alignement(::Type{T}) where T = T <: NTuple && length(T.parameters) == 3 && sizeof(T) == 12 ? 16 : sizeof(T)
inbuild_alignement(x::T) where T = inbuild_alignement(T)

function cl_alignement(x)
    is_cl_inbuild(x) ? inbuild_alignement(x) : cl_sizeof(x)
end

function advance_aligned(offset, alignment)
    (offset == 0 || alignment == 0) && return 0
    if offset % alignment != 0
        npad = ((div(offset, alignment) + 1) * alignment) - offset
        offset += npad
    end
    offset
end

function cl_sizeof(x, offset = 0)
    align, size = if is_cl_inbuild(x) || nfields(x) == 0
        inbuild_alignement(x), sizeof(x)
    else
        nextoffset = offset
        for field in fieldnames(x)
            xelem = isa(x, DataType) ? fieldtype(x, field) : getfield(x, field)
            nextoffset = cl_sizeof(xelem, nextoffset)
        end
        size = nextoffset - offset
        size, size
    end
    offset = advance_aligned(offset, align)
    offset += size
    offset
end


function aligned_convert(x, offset = Ref(0), types = [])
    _aligned_convert(x, offset, types)
    ret = if length(types) == 1
        types[1]
    else
        isa(x, DataType) ? Tuple{types...} : (types...,)
    end
    ret, offset[]
end
function _aligned_convert(x, offset = Ref(0), types = [])
    alignment = cl_alignement(x)
    if alignment != 0 && offset[] % alignment != 0
        npad = ((div(offset[], alignment) + 1) * alignment) - offset[]
        pad = CLBackend.Pad{npad}()
        offset[] += npad
        if isa(x, DataType)
            push!(types, CLBackend.Pad{npad})
        else
            push!(types, CLBackend.Pad{npad}())
        end
    end
    if !CLBackend.is_cl_inbuild(x) && nfields(x) > 0
        walk(_aligned_convert, x, offset, types)
    else
        push!(types, x)
        offset[] += sizeof(x)
    end
    return
end



function (AT::Type{<: CLArray{T, N}})(
        size::NTuple{N, Int};
        context = current_context(),
        flag = :rw, kw_args...
    ) where {T, N}
    ctx = context.context
    clT, offset = aligned_convert(T)
    clT = sizeof(clT) == sizeof(T) ? T : clT
    buffsize = prod(size)
    buff = buffsize == 0 ? cl.Buffer(clT, ctx, flag, 1) : cl.Buffer(clT, ctx, flag, buffsize)
    GPUArray{T, N, typeof(buff), typeof(context)}(buff, size, context)
end

# extend the private interface for the compilation types
Transpiler._to_cl_types{T, N}(arg::CLArray{T, N}) = cli.CLArray{T, N}
Transpiler._to_cl_types{T}(x::LocalMemory{T}) = cli.LocalMemory{T}

Transpiler._to_cl_types(x::Ref{<: CLArray}) = Transpiler._to_cl_types(x[])
Transpiler.cl_convert(x::Ref{<: CLArray}) = Transpiler.cl_convert(x[])
Transpiler.cl_convert(x::CLArray) = buffer(x)
Transpiler.cl_convert{T}(x::LocalMemory{T}) = cl.LocalMem(T, x.size)

function CLFunction{T, N}(A::CLArray{T, N}, f, args...)
    ctx = context(A)
    CLFunction(f, args, ctx.queue)
end
function (clfunc::CLFunction{T}){T, T2, N}(A::CLArray{T2, N}, args...)
    # TODO use better heuristic
    clfunc(args, length(A))
end

function gpu_call{T, N}(f, A::CLArray{T, N}, args, globalsize = length(A), localsize = nothing)
    ctx = GPUArrays.context(A)
    _args = if !isa(f, Tuple{String, Symbol})
        (0f0, args...)# include "state"
    else
        args
    end
    clfunc = CLFunction(f, _args, ctx.queue)
    clfunc(_args, globalsize, localsize)
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
    reduce(op, Array(out))
end


export CLFunction, cli


end #CLBackend

using .CLBackend
export CLBackend
