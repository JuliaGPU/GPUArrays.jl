# reference implementation on the CPU

# note that most of the code in this file serves to define a functional array type,
# the actual implementation of GPUArrays-interfaces is much more limited.

module JLArrays

export JLArray, jl

using GPUArrays

using Adapt


#
# Device functionality
#

## device properties

struct JLDevice <: AbstractGPUDevice end

GPUArrays.threads(dev::JLDevice) = 256


## execution

struct JLBackend <: AbstractGPUBackend end

mutable struct JLKernelContext <: AbstractKernelContext
    blockdim::Int
    griddim::Int
    blockidx::Int
    threadidx::Int

    localmem_counter::Int
    localmems::Vector{Vector{Array}}
end

function JLKernelContext(threads::Int, blockdim::Int)
    blockcount = prod(blockdim)
    lmems = [Vector{Array}() for i in 1:blockcount]
    JLKernelContext(threads, blockdim, 1, 1, 0, lmems)
end

function JLKernelContext(ctx::JLKernelContext, threadidx::Int)
    JLKernelContext(
        ctx.blockdim,
        ctx.griddim,
        ctx.blockidx,
        threadidx,
        0,
        ctx.localmems
    )
end

struct Adaptor end
jlconvert(arg) = adapt(Adaptor(), arg)

# FIXME: add Ref to Adapt.jl (but make sure it doesn't cause ambiguities with CUDAnative's)
struct JlRefValue{T} <: Ref{T}
  x::T
end
Base.getindex(r::JlRefValue) = r.x
Adapt.adapt_structure(to::Adaptor, r::Base.RefValue) = JlRefValue(adapt(to, r[]))

function GPUArrays.gpu_call(::JLBackend, f, args, threads::Int, blocks::Int;
                            name::Union{String,Nothing})
    ctx = JLKernelContext(threads, blocks)
    device_args = jlconvert.(args)
    tasks = Array{Task}(undef, threads)
    @disallowscalar for blockidx in 1:blocks
        ctx.blockidx = blockidx
        for threadidx in 1:threads
            thread_ctx = JLKernelContext(ctx, threadidx)
            tasks[threadidx] = @async f(thread_ctx, device_args...)
            # TODO: require 1.3 and use Base.Threads.@spawn for actual multithreading
            #       (this would require a different synchronization mechanism)
        end
        for t in tasks
            fetch(t)
        end
    end
    return
end


## executed on-device

# array type

struct JLDeviceArray{T, N} <: AbstractDeviceArray{T, N}
    data::Array{T, N}
    dims::Dims{N}

    function JLDeviceArray{T,N}(data::Array{T, N}, dims::Dims{N}) where {T,N}
        new(data, dims)
    end
end

Base.size(x::JLDeviceArray) = x.dims

@inline Base.getindex(A::JLDeviceArray, index::Integer) = getindex(A.data, index)
@inline Base.setindex!(A::JLDeviceArray, x, index::Integer) = setindex!(A.data, x, index)

# indexing

for f in (:blockidx, :blockdim, :threadidx, :griddim)
    @eval GPUArrays.$f(ctx::JLKernelContext) = ctx.$f
end

# memory

function GPUArrays.LocalMemory(ctx::JLKernelContext, ::Type{T}, ::Val{dims}, ::Val{id}) where {T, dims, id}
    ctx.localmem_counter += 1
    lmems = ctx.localmems[blockidx(ctx)]

    # first invocation in block
    data = if length(lmems) < ctx.localmem_counter
        lmem = fill(zero(T), dims)
        push!(lmems, lmem)
        lmem
    else
        lmems[ctx.localmem_counter]
    end

    N = length(dims)
    JLDeviceArray{T,N}(data, tuple(dims...))
end

# synchronization

@inline function GPUArrays.synchronize_threads(::JLKernelContext)
    # All threads are getting started asynchronously, so a yield will yield to the next
    # execution of the same function, which should call yield at the exact same point in the
    # program, leading to a chain of yields effectively syncing the tasks (threads).
    yield()
    return
end


#
# Host abstractions
#

struct JLArray{T, N} <: AbstractGPUArray{T, N}
    data::Array{T, N}
    dims::Dims{N}

    function JLArray{T,N}(data::Array{T, N}, dims::Dims{N}) where {T,N}
        @assert isbitstype(T) "JLArray only supports bits types"
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

Base.copy(a::JLArray{T,N}) where {T,N} = JLArray{T,N}(copy(a.data), size(a))


## array interface

Base.elsize(::Type{<:JLArray{T}}) where {T} = sizeof(T)

Base.size(x::JLArray) = x.dims
Base.sizeof(x::JLArray) = Base.elsize(x) * length(x)


## interop with Julia arrays

JLArray{T,N}(x::AbstractArray{<:Any,N}) where {T,N} =
    JLArray{T,N}(convert(Array{T}, x), size(x))

# underspecified constructors
JLArray{T}(xs::AbstractArray{S,N}) where {T,N,S} = JLArray{T,N}(xs)
(::Type{JLArray{T,N} where T})(x::AbstractArray{S,N}) where {S,N} = JLArray{S,N}(x)
JLArray(A::AbstractArray{T,N}) where {T,N} = JLArray{T,N}(A)

# idempotency
JLArray{T,N}(xs::JLArray{T,N}) where {T,N} = xs

jl(xs) = adapt(JLArray, xs)
Adapt.adapt_storage(::Type{JLArray}, xs::AbstractArray) = convert(JLArray, xs)


## conversions

Base.convert(::Type{T}, x::T) where T <: JLArray = x

function Base._reshape(parent::JLArray, dims::Dims)
  n = length(parent)
  prod(dims) == n || throw(DimensionMismatch("parent has $n elements, which is incompatible with size $dims"))
  return JLArray{eltype(parent),length(dims)}(reshape(parent.data, dims), dims)
end
function Base._reshape(parent::JLArray{T,1}, dims::Tuple{Int}) where T
  n = length(parent)
  prod(dims) == n || throw(DimensionMismatch("parent has $n elements, which is incompatible with size $dims"))
  return parent
end


## broadcast

using Base.Broadcast: BroadcastStyle, Broadcasted

struct JLArrayStyle{N} <: AbstractGPUArrayStyle{N} end
JLArrayStyle(::Val{N}) where N = JLArrayStyle{N}()
JLArrayStyle{M}(::Val{N}) where {N,M} = JLArrayStyle{N}()

BroadcastStyle(::Type{JLArray{T,N}}) where {T,N} = JLArrayStyle{N}()

# Allocating the output container
Base.similar(bc::Broadcasted{JLArrayStyle{N}}, ::Type{T}) where {N,T} =
    similar(JLArray{T}, axes(bc))
Base.similar(bc::Broadcasted{JLArrayStyle{N}}, ::Type{T}, dims...) where {N,T} =
    JLArray{T}(undef, dims...)


## math

for f in (:cos, :sin, :sqrt, :log)
    @eval GPUArrays.$f(ctx::JLKernelContext, x) = $f(x)
end

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


## GPUArrays interfaces

GPUArrays.device(x::JLArray) = JLDevice()

GPUArrays.backend(::Type{<:JLArray}) = JLBackend()

Adapt.adapt_storage(::Adaptor, x::JLArray{T,N}) where {T,N} =
  JLDeviceArray{T,N}(x.data, x.dims)

GPUArrays.unsafe_reinterpret(::Type{T}, A::JLArray, size::Tuple) where T =
    reshape(reinterpret(T, A.data), size)

function GPUArrays.mapreducedim!(f, op, R::JLArray, A::Union{AbstractArray,Broadcast.Broadcasted};
                                 init=nothing)
    if init !== nothing
        fill!(R, init)
    end
    @allowscalar Base.reducedim!(op, R.data, map(f, A))
end


## LinearAlgebra

using LinearAlgebra

for TR in (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular)
    @eval LinearAlgebra.ldiv!(x::$TR{T,<:JLArray{T,2}}, y::JLArray{T,2}) where T<:Union{Complex{Float32}, Complex{Float64}, Float32, Float64} = JLArray(LinearAlgebra.ldiv!($TR(parent(x).data),y.data))
    @eval LinearAlgebra.rdiv!(x::JLArray{T,2}, y::$TR{T,<:JLArray{T,2}}) where T<:Union{Complex{Float32}, Complex{Float64}, Float32, Float64} = JLArray(LinearAlgebra.rdiv!(x.data,$TR(parent(y).data)))
end

end
