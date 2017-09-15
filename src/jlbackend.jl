# Very simple Julia backend which is just for testing the implementation
# and can be used as a reference implementation
import Base: pointer, similar, size, copy!, convert
using Base: RefValue
struct JLArray{T, N} <: GPUArray{T, N}
    data::Array{T, N}
    size::NTuple{N, Int}
end

size(x::JLArray) = x.size
pointer(x::JLArray) = pointer(x.data)
to_device(x::JLArray) = x.data
to_device(x::Tuple) = to_device.(x)
to_device(x::RefValue{<: JLArray}) = RefValue(to_device(x[]))
to_device(x) = x

function (::Type{JLArray{T, N}})(size::NTuple{N, Integer}) where {T, N}
    JLArray{T, N}(Array{T, N}(size), size)
end

similar(::Type{<: JLArray}, ::Type{T}, size::Base.Dims{N}) where {T, N} = JLArray{T, N}(size)

function unsafe_reinterpret(::Type{T}, A::JLArray{ET}, size::NTuple{N, Integer}) where {T, ET, N}
    JLArray(collect(reshape(reinterpret(T, A.data), size)))
end

function copy!{T}(
        dest::Array{T}, d_offset::Integer,
        source::JLArray{T}, s_offset::Integer, amount::Integer
    )
    copy!(dest, d_offset, source.data, s_offset, amount)
end
function copy!{T}(
        dest::JLArray{T}, d_offset::Integer,
        source::Array{T}, s_offset::Integer, amount::Integer
    )
    copy!(dest.data, d_offset, source, s_offset, amount)
    dest
end
function copy!{T}(
        dest::JLArray{T}, d_offset::Integer,
        source::JLArray{T}, s_offset::Integer, amount::Integer
    )
    copy!(dest.data, d_offset, source.data, s_offset, amount)
    dest
end

mutable struct JLState{N}
    blockdim::NTuple{N, Int}
    threads::NTuple{N, Int}

    blockidx::NTuple{N, Int}
    threadidx::NTuple{N, Int}
end


function gpu_call(f, A::JLArray, args::Tuple, blocks = nothing, threads = C_NULL)
    if blocks == nothing
        blocks, threads = thread_blocks_heuristic(length(A))
    elseif isa(blocks, Integer)
        blocks = (blocks,)
        if threads == C_NULL
            threads = (1,)
        end
    end
    idx = ntuple(i-> 1, length(blocks))
    blockdim = ceil.(Int, blocks ./ threads)
    state = JLState(threads, threads, idx, idx)
    device_args = to_device.(args)
    for blockidx in CartesianRange(blockdim)
        state.blockidx = blockidx.I
        for threadidx in CartesianRange(threads)
            state.threadidx = threadidx.I
            f(state, device_args...)
        end
    end
    return
end

# "intrinsics"
struct JLDevice end
device(x::JLArray) = JLDevice()
threads(dev::JLDevice) = 256


@inline synchronize_threads(::JLState) = nothing

for f in (:blockidx, :blockdim, :threadidx), (i, sym) in enumerate((:x, :y, :z))
    fname = Symbol(string(f, '_', sym))
    @eval $fname(state::JLState) = Cuint(state.$f[$i])
end

blas_module(::JLArray) = Base.LinAlg.BLAS
blasbuffer(A::JLArray) = A.data

import Base: *, plan_ifft!, plan_fft!, plan_fft, plan_ifft, size, plan_bfft, plan_bfft!
# defining our own plan type is the easiest way to pass around the plans in Base interface without ambiguities
immutable FFTPlan{T}
    p::T
end
function plan_fft(A::JLArray; kw_args...)
    FFTPlan(plan_fft(A.data; kw_args...))
end
function plan_fft!(A::JLArray; kw_args...)
    FFTPlan(plan_fft!(A.data; kw_args...))
end
function plan_bfft!(A::JLArray; kw_args...)
    FFTPlan(plan_bfft!(A.data; kw_args...))
end
function plan_bfft(A::JLArray; kw_args...)
    FFTPlan(plan_bfft(A.data; kw_args...))
end
function plan_ifft!(A::JLArray; kw_args...)
    FFTPlan(plan_ifft!(A.data; kw_args...))
end
function plan_ifft(A::JLArray; kw_args...)
    FFTPlan(plan_ifft(A.data; kw_args...))
end


function *(plan::FFTPlan, A::JLArray)
    x = plan.p * A.data
    JLArray(x)
end
