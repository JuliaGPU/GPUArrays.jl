# Dense GPU Array
abstract type GPUArray{T, N} <: DenseArray{T, N} end

# Sampler type that acts like a texture/image and allows interpolated access
abstract type Sampler{T, N} <: DenseArray{T, N} end

const GPUVector{T} = GPUArray{T, 1}
const GPUMatrix{T} = GPUArray{T, 2}
const GPUVecOrMat{T} = Union{GPUArray{T, 1}, GPUArray{T, 2}}

#=
Interface for accessing the lower level
=#
buffer(A::GPUArray) = A.buffer
context(A::GPUArray) = A.context
default_buffer_type(typ, context) = error("Found unsupported context: $context")

# GPU Local Memory
immutable LocalMemory{T} <: GPUArray{T, 1}
    size::Int
end


"""
linear index in a GPU kernel
"""
function linear_index end


#=
AbstractArray interface
=#
Base.eltype{T}(::GPUArray{T}) = T
Base.size(A::GPUArray) = A.size

function Base.show(io::IO, mt::MIME"text/plain", A::GPUArray)
    show(io, mt, Array(A))
end
function Base.showcompact(io::IO, mt::MIME"text/plain", A::GPUArray)
    showcompact(io, mt, Array(A))
end

function Base.similar{T <: GPUArray}(x::T)
    similar(x, eltype(x), size(x))
end
function Base.similar{T <: GPUArray, ET}(x::T, ::Type{ET}; kw_args...)
    similar(x, ET, size(x); kw_args...)
end
function Base.similar{T <: GPUArray, N}(x::T, dims::NTuple{N, Int}; kw_args...)
    similar(x, eltype(x), dims; kw_args...)
end
function Base.similar{N, ET}(x::GPUArray, ::Type{ET}, sz::NTuple{N, Int}; kw_args...)
    similar(typeof(x), ET, sz, context = context(x); kw_args...)
end


function Base.similar{T <: GPUArray, ET, N}(
        ::Type{T}, ::Type{ET}, sz::NTuple{N, Int};
        context::Context = current_context(), kw_args...
    )
    bt = default_buffer_type(T, Tuple{ET, N}, context)
    GPUArray{ET, N, bt, typeof(context)}(sz; context = context)
end



#=
Host to Device data transfers
=#
function (::Type{A}){A <: GPUArray}(x::AbstractArray)
    A(collect(x))
end
function (::Type{A}){A <: GPUArray}(x::Array; kw_args...)
    out = similar(A, eltype(x), size(x); kw_args...)
    copy!(out, x)
    out
end
Base.convert{A <: GPUArray}(::Type{A}, x::AbstractArray) = A(x)
Base.convert{A <: GPUArray}(::Type{A}, x::A) = x

#=
Device to host data transfers
=#
function (::Type{Array}){T, N}(device_array::GPUArray{T, N})
    Array{T, N}(device_array)
end
function (AT::Type{Array{T, N}}){T, N}(device_array::GPUArray)
    convert(AT, Array(device_array))
end
function (AT::Type{Array{T, N}}){T, N}(device_array::GPUArray{T, N})
    hostarray = similar(AT, size(device_array))
    copy!(hostarray, device_array)
    hostarray
end


######################################
# Broadcast
include("broadcast.jl")

############################################
# serialization

const BaseSerializer = if isdefined(Base, :AbstractSerializer)
    Base.AbstractSerializer
elseif isdefined(Base, :SerializationState)
    Base.SerializationState
else
    error("No Serialization type found. Probably unsupported Julia version")
end

function Base.serialize{T <: GPUArray}(s::BaseSerializer, t::T)
    Base.serialize_type(s, T)
    serialize(s, Array(t))
end
function Base.deserialize{T <: GPUArray}(s::BaseSerializer, ::Type{T})
    A = deserialize(s)
    T(A)
end

@inline unpack_buffer(x) = x
@inline unpack_buffer(x::GPUArray) = buffer(x)
@inline unpack_buffer(x::Ref{<: GPUArray}) = unpack_buffer(x[])

function to_cartesian(A, indices::Tuple)
    start = CartesianIndex(ntuple(length(indices)) do i
        val = indices[i]
        isa(val, Integer) && return val
        isa(val, UnitRange) && return first(val)
        isa(val, Colon) && return 1
        error("GPU indexing only defined for integers or unit ranges. Found: $val")
    end)
    stop = CartesianIndex(ntuple(length(indices)) do i
        val = indices[i]
        isa(val, Integer) && return val
        isa(val, UnitRange) && return last(val)
        isa(val, Colon) && return size(A, i)
        error("GPU indexing only defined for integers or unit ranges. Found: $val")
    end)
    CartesianRange(start, stop)
end

crange(start, stop) = CartesianRange(CartesianIndex(start), CartesianIndex(stop))

#Hmmm... why is this not part of the Array constructors???
#TODO Figure out or issue THEM JULIA CORE PEOPLE SO HARD ... or PR? Who'd know
function array_convert{T, N}(t::Type{Array{T, N}}, x::Array)
    convert(t, x)
end


array_convert{T, N}(t::Type{Array{T, N}}, x::T) = [x]

function array_convert{T, N, T2}(t::Type{Array{T, N}}, x::T2)
    arr = collect(x) # iterator
    dims = ntuple(Val{N}) do i
        ifelse(ndims(arr) >= i, size(arr, i), 1)
    end
    return reshape(map(T, arr), dims) # broadcast dims
end

for (D, S) in ((GPUArray, AbstractArray), (AbstractArray, GPUArray), (GPUArray, GPUArray))
    @eval begin
        function copy!(
                dest::$D, doffset::Integer,
                src::$S, soffset::Integer, amount::Integer
            )
            copy!(
                unpack_buffer(dest), doffset,
                unpack_buffer(src), soffset, amount
            )
        end
        function copy!{T, N}(
                dest::$D{T, N}, rdest::NTuple{N, UnitRange},
                src::$S{T, N}, ssrc::NTuple{N, UnitRange},
            )
            drange = crange(start.(rdest), last.(rdest))
            srange = crange(start.(ssrc), last.(ssrc))
            copy!(dest, drange, src, srange)
        end
        function copy!{T}(
                dest::$D{T}, d_range::CartesianRange{CartesianIndex{1}},
                src::$S{T}, s_range::CartesianRange{CartesianIndex{1}},
            )
            amount = length(d_range)
            if length(s_range) != amount
                throw(ArgumentError("Copy range needs same length. Found: dest: $amount, src: $(length(s_range))"))
            end
            amount == 0 && return dest
            d_offset = first(d_range)[1]
            s_offset = first(s_range)[1]
            copy!(dest, d_offset, src, s_offset, amount)
        end
        # function copy!{T, N}(
        #         dest::$D{T, N}, rdest::CartesianRange{CartesianIndex{N}},
        #         src::$S{T, N}, ssrc::CartesianRange{CartesianIndex{N}},
        #     )
        #     copy!(unpack_buffer(dest), rdest, unpack_buffer(src), ssrc)
        # end
        function copy!{T, N}(
                dest::$D{T, N}, src::$S{T, N}
            )
            len = length(src)
            len == 0 && return dest
            if length(dest) > len
                throw(BoundsError(dest, length(src)))
            end
            copy!(dest, 1, src, 1, len)
        end
    end
end


function copy_kernel!(state, dest, dest_offsets, src, src_offsets, shape, shape_dest, shape_source, length)
    i = linear_index(dest, state)
    if i <= length
        # TODO can this be done faster and smarter?
        idx = gpu_ind2sub(shape, i)
        dest_idx = gpu_sub2ind(shape_dest, idx .+ dest_offsets)
        src_idx = gpu_sub2ind(shape_source, idx .+ src_offsets)
        @inbounds dest[dest_idx] = src[src_idx]
    end
    return
end

function copy!{T, N}(
        dest::GPUArray{T, N}, destcrange::CartesianRange{CartesianIndex{N}},
        src::GPUArray{T, N}, srccrange::CartesianRange{CartesianIndex{N}}
    )
    shape = size(destcrange)
    if shape != size(srccrange)
        throw(DimensionMismatch("Ranges don't match their size. Found: $shape, $(size(srccrange))"))
    end
    len = length(destcrange)
    dest_offsets = Cuint.(destcrange.start.I .- 1)
    src_offsets = Cuint.(srccrange.start.I .- 1)
    ui_shape = Cuint.(shape)
    gpu_call(
        copy_kernel!, dest,
        (dest, dest_offsets, src, src_offsets, ui_shape, Cuint.(size(dest)), Cuint.(size(src)), Cuint(len)),
        len
    )
    dest
end


function copy!{T, N}(
        dest::GPUArray{T, N}, destcrange::CartesianRange{CartesianIndex{N}},
        src::AbstractArray{T, N}, srccrange::CartesianRange{CartesianIndex{N}}
    )
    # Is this efficient? Maybe!
    # TODO: compare to a pure intrinsic copy implementation!
    # this would mean looping over linear sections of memory and
    # use copy!(dest, offset::Integer, buffer(src), offset::Integer, amout::Integer)
    src_gpu = typeof(dest)(map(idx-> src[idx], srccrange))
    nrange = CartesianRange(one(CartesianIndex{N}), CartesianIndex(size(src_gpu)))
    copy!(dest, destcrange, src_gpu, nrange)
    dest
end


function copy!{T, N}(
        dest::AbstractArray{T, N}, destcrange::CartesianRange{CartesianIndex{N}},
        src::GPUArray{T, N}, srccrange::CartesianRange{CartesianIndex{N}}
    )
    # Is this efficient? Maybe!
    dest_gpu = similar(src, size(destcrange))
    nrange = CartesianRange(one(CartesianIndex{N}), CartesianIndex(size(dest_gpu)))
    copy!(dest_gpu, nrange, src, srccrange)
    copy!(dest, destcrange, Array(dest_gpu), nrange)
    dest
end

Base.copy(x::GPUArray) = identity.(x)

#=
reinterpret taken from julia base/array.jl
Copyright (c) 2009-2016: Jeff Bezanson, Stefan Karpinski, Viral B. Shah, and other contributors:

https://github.com/JuliaLang/julia/contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
=#
import Base.reinterpret

"""
Unsafe reinterpret for backends to overload.
This makes it easier to do checks just on the high level.
"""
function unsafe_reinterpret end

function reinterpret(::Type{T}, a::GPUArray{S,1}) where T where S
    nel = Int(div(length(a)*sizeof(S),sizeof(T)))
    # TODO: maybe check that remainder is zero?
    return reinterpret(T, a, (nel,))
end

function reinterpret(::Type{T}, a::GPUArray{S}) where T where S
    if sizeof(S) != sizeof(T)
        throw(ArgumentError("result shape not specified"))
    end
    reinterpret(T, a, size(a))
end

function reinterpret(::Type{T}, a::GPUArray{S}, dims::NTuple{N,Int}) where T where S where N
    if !isbits(T)
        throw(ArgumentError("cannot reinterpret Array{$(S)} to ::Type{Array{$(T)}}, type $(T) is not a bits type"))
    end
    if !isbits(S)
        throw(ArgumentError("cannot reinterpret Array{$(S)} to ::Type{Array{$(T)}}, type $(S) is not a bits type"))
    end
    nel = div(length(a)*sizeof(S),sizeof(T))
    if prod(dims) != nel
        throw(DimensionMismatch("new dimensions $(dims) must be consistent with array size $(nel)"))
    end
    unsafe_reinterpret(T, a, dims)
end

function Base.reshape(a::GPUArray{T}, dims::NTuple{N,Int}) where T where N
    if prod(dims) != length(a)
        throw(DimensionMismatch("new dimensions $(dims) must be consistent with array size $(length(a))"))
    end
    unsafe_reinterpret(T, a, dims)
end
