abstract AbstractAccArray{T, N} <: DenseArray{T, N}

type GPUArray{T, N, B, C} <: AbstractAccArray{T, N}
    buffer::B
    size::NTuple{N, Int}
    context::C
end

#=
context interface
=#

buffer(A::AbstractAccArray) = A.buffer
context(A::AbstractAccArray) = A.context

#=
AbstractArray interface
=#

Base.eltype{B, T, N, C}(::Type{GPUArray{T, N, B, C}}) = T
Base.eltype{T}(::AbstractAccArray{T}) = T
Base.size(A::AbstractAccArray) = A.size
Base.size(A::AbstractAccArray, i::Int) = A.size[i]

function Base.show(io::IO, mt::MIME"text/plain", A::AbstractAccArray)
    println(io, "GPUArray with ctx: $(context(A)): ")
    show(io, mt, Array(A))
end
function Base.showcompact(io::IO, mt::MIME"text/plain", A::AbstractAccArray)
    showcompact(io, mt, Array(A))
end

function Base.similar{T <: AbstractAccArray}(x::T)
    simbuff = similar(buffer(x))
    T(simbuff, size(x), context = context(x))
end

#=
Host to Device data transfers
=#

# don't want to jump straight into refactor hell, so don't force GPU packges to inherit from GPUBuffer
function GPUArray(
        buffer#=::GPUBuffer=#, sz::Tuple;
        context::Context = current_context()
    )
    b, T, N = buffer, eltype(buffer), length(sz)
    GPUArray{T, N, typeof(b), typeof(context)}(buffer, sz, context)
end
function GPUArray{T, N}(
        ::Type{T}, sz::Vararg{Int, N};
        kw_args...
    )
    GPUArray(T, sz; kw_args...)
end
function GPUArray{T}(
        ::Type{T}, sz::Tuple;
        context::Context=current_context(), kw_args...
    )
    b = create_buffer(context, T, sz; kw_args...)
    GPUArray(b, sz, context=context)
end

function GPUArray{T, N}(
        host_array::AbstractArray{T, N};
        context::Context=current_context(), kw_args...
    )
    concrete_ha = convert(Array, host_array)
    gpu_array = GPUArray(T, size(concrete_ha), context=context)
    unsafe_copy!(gpu_array, concrete_ha)
    gpu_array
end

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
    unsafe_copy!(hostarray, device_array)
    hostarray
end


#=
Copying
=#

function Base.unsafe_copy!{T, N}(dest::Array{T, N}, source::GPUArray{T, N})
    Base.unsafe_copy!(dest, buffer(source))
end
function Base.unsafe_copy!{T, N}(dest::GPUArray{T, N}, source::Array{T, N})
    Base.unsafe_copy!(buffer(dest), source)
end

export buffer, context


# Helpers for lazy arrays, which can be used for e.g. indices
using StaticArrays, BenchmarkTools
immutable Grid{T, N, RT} <: AbstractArray{T, N}
    grid::NTuple{N, RT}
end
Base.size(p::Grid) = map(length, p.grid)
Base.@propagate_inbounds function Base.getindex{T, N, RT, ID <: Integer}(g::Grid{T, N, RT}, idx::Vararg{ID, N})
    T(ntuple(Val{N}) do i
        g.grid[i][idx[i]]
    end)
end
Base.@propagate_inbounds function Base.getindex{T, RT}(g::Grid{T, 3, RT}, x, y, z)
    T((
        g.grid[1][x],
        g.grid[2][y],
        g.grid[3][z],
    ))
end
function Grid{N, T}(ranges::Vararg{T, N})
    Grid(NTuple{N, eltype(first(ranges))}, ranges)
end
function Grid{N, ET, T <: AbstractVector}(::Type{ET}, ranges::NTuple{N, T})
    Grid{ET, N, T}(ranges)
end
function Grid{N, ET, T <: AbstractVector}(::Type{ET}, ranges::Vararg{T, N})
    Grid{ET, N, T}(ranges)
end
