# reference implementation of a CPU-based array type

module JLArrays

using GPUArrays

export JLArray


#
# Host array
#

# the definition of a host array type, implementing different Base interfaces
# to make it function properly and behave like the Base Array type.

struct JLArray{T, N} <: AbstractGPUArray{T, N}
    data::Array{T, N}
    dims::Dims{N}

    function JLArray{T,N}(data::Array{T, N}, dims::Dims{N}) where {T,N}
        new(data, dims)
    end
end

## constructors

# type and dimensionality specified, accepting dims as tuples of Ints
JLArray{T,N}(::UndefInitializer, dims::Dims{N}) where {T,N} =
  JLArray{T,N}(Array{T, N}(undef, dims), dims)

# type and dimensionality specified, accepting dims as series of Ints
JLArray{T,N}(::UndefInitializer, dims::Integer...) where {T,N} = JLArray{T,N}(undef, dims)

# type but not dimensionality specified
JLArray{T}(::UndefInitializer, dims::Dims{N}) where {T,N} = JLArray{T,N}(undef, dims)
JLArray{T}(::UndefInitializer, dims::Integer...) where {T} =
  JLArray{T}(undef, convert(Tuple{Vararg{Int}}, dims))

# empty vector constructor
JLArray{T,1}() where {T} = JLArray{T,1}(undef, 0)

Base.similar(a::JLArray{T,N}) where {T,N} = JLArray{T,N}(undef, size(a))
Base.similar(a::JLArray{T}, dims::Base.Dims{N}) where {T,N} = JLArray{T,N}(undef, dims)
Base.similar(a::JLArray, ::Type{T}, dims::Base.Dims{N}) where {T,N} = JLArray{T,N}(undef, dims)


## array interface

Base.elsize(::Type{<:JLArray{T}}) where {T} = sizeof(T)

Base.size(x::JLArray) = x.dims
Base.sizeof(x::JLArray) = Base.elsize(x) * length(x)


## interop with other arrays

JLArray{T,N}(x::AbstractArray{S,N}) where {T,N,S} =
    JLArray{T,N}(convert(Array{T}, x), size(x))

# underspecified constructors
JLArray{T}(xs::AbstractArray{S,N}) where {T,N,S} = JLArray{T,N}(xs)
(::Type{JLArray{T,N} where T})(x::AbstractArray{S,N}) where {S,N} = JLArray{S,N}(x)
JLArray(A::AbstractArray{T,N}) where {T,N} = JLArray{T,N}(A)

# idempotency
JLArray{T,N}(xs::JLArray{T,N}) where {T,N} = xs


## conversions

Base.convert(::Type{T}, x::T) where T <: JLArray = x


## broadcast

using Base.Broadcast: BroadcastStyle, Broadcasted, ArrayStyle

BroadcastStyle(::Type{<:JLArray}) = ArrayStyle{JLArray}()

function Base.similar(bc::Broadcasted{ArrayStyle{JLArray}}, ::Type{T}) where T
    similar(JLArray{T}, axes(bc))
end

Base.similar(bc::Broadcasted{ArrayStyle{JLArray}}, ::Type{T}, dims...) where {T} = JLArray{T}(undef, dims...)


## memory operations

function Base.copyto!(dest::Array{T}, d_offset::Integer,
                      source::JLArray{T}, s_offset::Integer,
                      amount::Integer) where T
    @boundscheck checkbounds(dest, d_offset+amount-1)
    @boundscheck checkbounds(source, s_offset+amount-1)
    copyto!(dest, d_offset, source.data, s_offset, amount)
end

function Base.copyto!(dest::JLArray{T}, d_offset::Integer,
                      source::Array{T}, s_offset::Integer,
                      amount::Integer) where T
    @boundscheck checkbounds(dest, d_offset+amount-1)
    @boundscheck checkbounds(source, s_offset+amount-1)
    copyto!(dest.data, d_offset, source, s_offset, amount)
    dest
end

function Base.copyto!(dest::JLArray{T}, d_offset::Integer,
                      source::JLArray{T}, s_offset::Integer,
                      amount::Integer) where T
    @boundscheck checkbounds(dest, d_offset+amount-1)
    @boundscheck checkbounds(source, s_offset+amount-1)
    copyto!(dest.data, d_offset, source.data, s_offset, amount)
    dest
end

## fft

using AbstractFFTs

# defining our own plan type is the easiest way to pass around the plans in FFTW interface
# without ambiguities

struct FFTPlan{T}
    p::T
end

AbstractFFTs.plan_fft(A::JLArray; kw_args...) = FFTPlan(plan_fft(A.data; kw_args...))
AbstractFFTs.plan_fft!(A::JLArray; kw_args...) = FFTPlan(plan_fft!(A.data; kw_args...))
AbstractFFTs.plan_bfft!(A::JLArray; kw_args...) = FFTPlan(plan_bfft!(A.data; kw_args...))
AbstractFFTs.plan_bfft(A::JLArray; kw_args...) = FFTPlan(plan_bfft(A.data; kw_args...))
AbstractFFTs.plan_ifft!(A::JLArray; kw_args...) = FFTPlan(plan_ifft!(A.data; kw_args...))
AbstractFFTs.plan_ifft(A::JLArray; kw_args...) = FFTPlan(plan_ifft(A.data; kw_args...))

function Base.:(*)(plan::FFTPlan, A::JLArray)
    x = plan.p * A.data
    JLArray(x)
end



#
# AbstractGPUArray interface
#

# implementation of GPUArrays-specific interfaces

GPUArrays.unsafe_reinterpret(::Type{T}, A::JLArray, size::Tuple) where T =
    reshape(reinterpret(T, A.data), size)


## execution

struct JLBackend <: AbstractGPUBackend end

GPUArrays.backend(::Type{<:JLArray}) = JLBackend()

mutable struct JLState
    blockdim::Int
    griddim::Int

    blockidx::Int
    threadidx::Int
    localmem_counter::Int
    localmems::Vector{Vector{Array}}
end

function JLState(threads::Int, blockdim::Int)
    blockcount = prod(blockdim)
    lmems = [Vector{Array}() for i in 1:blockcount]
    JLState(threads, blockdim, 1, 1, 0, lmems)
end

function JLState(state::JLState, threadidx::Int)
    JLState(
        state.blockdim,
        state.griddim,
        state.blockidx,
        threadidx,
        0,
        state.localmems
    )
end

to_device(state, x::JLArray{T,N}) where {T,N} = JLDeviceArray{T,N}(x.data, x.dims)
to_device(state, x::Tuple) = to_device.(Ref(state), x)
to_device(state, x::Base.RefValue{<: JLArray}) = Base.RefValue(to_device(state, x[]))
to_device(state, x) = x

function GPUArrays._gpu_call(::JLBackend, f, A, args::Tuple, blocks_threads::Tuple{Int, Int})
    blocks, threads = blocks_threads
    state = JLState(threads, blocks)
    device_args = to_device.(Ref(state), args)
    tasks = Array{Task}(undef, threads)
    for blockidx in 1:blocks
        state.blockidx = blockidx
        for threadidx in 1:threads
            thread_state = JLState(state, threadidx)
            tasks[threadidx] = @async @allowscalar f(thread_state, device_args...)
            # TODO: require 1.3 and use Base.Threads.@spawn for actual multithreading
            #       (this would require a different synchronization mechanism)
        end
        for t in tasks
            fetch(t)
        end
    end
    return
end


## device properties

struct JLDevice end

GPUArrays.device(x::JLArray) = JLDevice()

GPUArrays.threads(dev::JLDevice) = 256


## linear algebra

using LinearAlgebra

GPUArrays.blas_module(::JLArray) = LinearAlgebra.BLAS
GPUArrays.blasbuffer(A::JLArray) = A.data



#
# Device array
#

# definition of a minimal device array type that supports the subset of operations
# that are used in GPUArrays kernels

struct JLDeviceArray{T, N} <: AbstractDeviceArray{T, N}
    data::Array{T, N}
    dims::Dims{N}

    function JLDeviceArray{T,N}(data::Array{T, N}, dims::Dims{N}) where {T,N}
        new(data, dims)
    end
end

function GPUArrays.LocalMemory(state::JLState, ::Type{T}, ::Val{dims}, ::Val{id}) where {T, dims, id}
    state.localmem_counter += 1
    lmems = state.localmems[blockidx(state)]

    # first invocation in block
    data = if length(lmems) < state.localmem_counter
        lmem = fill(zero(T), dims)
        push!(lmems, lmem)
        lmem
    else
        lmems[state.localmem_counter]
    end

    N = length(dims)
    JLDeviceArray{T,N}(data, tuple(dims...))
end


## array interface

Base.size(x::JLDeviceArray) = x.dims


## indexing

@inline Base.getindex(A::JLDeviceArray, index::Integer) = getindex(A.data, index)
@inline Base.setindex!(A::JLDeviceArray, x, index::Integer) = setindex!(A.data, x, index)

for f in (:blockidx, :blockdim, :threadidx, :griddim)
    @eval GPUArrays.$f(state::JLState) = state.$f
end


## synchronization

@inline function GPUArrays.synchronize_threads(::JLState)
    # All threads are getting started asynchronously, so a yield will yield to the next
    # execution of the same function, which should call yield at the exact same point in the
    # program, leading to a chain of yields effectively syncing the tasks (threads).
    yield()
    return
end

end
