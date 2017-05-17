import Base: copy!, splice!, append!, push!, setindex!, start, next, done
import Base: getindex, map, length, eltype, endof, ndims, size, resize!


function to_cartesian(indices::Tuple)
    start = CartesianIndex(map(indices) do i
        isa(val, Integer) && return val
        isa(val, UnitRange) && return first(val)
        error("GPU indexing only defined for integers or unit ranges. Found: $val")
    end)
    stop = CartesianIndex(map(indices) do i
        isa(val, Integer) && return val
        isa(val, UnitRange) && return last(val)
        error("GPU indexing only defined for integers or unit ranges. Found: $val")
    end)
    CartesianRange(start, stop)
end


#Hmmm... why is this not part of the Array constructors???
#TODO Figure out or issue THEM JULIA CORE PEOPLE SO HARD ... or PR? Who'd know
function array_convert{T, N}(t::Type{Array{T, N}}, x::AbstractArray)
    convert(t, x)
end
function array_convert{T, N}(t::Type{Array{T, N}}, x)
    # okay x is no array, so convert doesn't work.. Only chance is collect
    # this actually works nice for values, since they become 0 dim array, which we conveniently can use as a mem ref
    convert(t, collect(x))
end


function setindex!{T, N}(A::GPUArray{T, N}, value, indexes...)
    # similarly, value should always be a julia array
    v = array_convert(Array{T, N}, value)
    # We shouldn't really bother about checkbounds performance, since setindex/getindex will always be relatively slow
    checkbounds(A, ranges...)
    checkbounds(v, ranges...)
    # since you shouldn't update GPUArrays with single indices, we simplify the interface
    # by always mapping to ranges
    ranges_dest = to_cartesian(indexes)
    ranges_src = CartesianRange(size(value))

    copy!(A, ranges_dest, v, ranges_src)
    return
end

"""
resize! of GPUArrays. Tries to be inplace, but inplace can also mean changing the
buffer of the GPUArray
"""
function resize!(A::AbstractGPUArray, dims)
    error("Inplace resizing not implemented for $(typeof(A))")
end


function resize!{T, NDim}(A::GPUArray{T, NDim}, newdims::NTuple{NDim, Int})
    newdims == size(A) && return A
    A.buffer = resize!(buffer(A), newdims)
    A.size = newdims # we might need an interface for this
    A
end

function reshape!{T, NDim}(A::GPUArray{T, NDim}, newdims::NTuple{NDim, Int})
    size(A) == newdims && return A
    if prod(newdims) == length(A)
        throw(DimensionMismatch("new dimensions $newdims must be consistent with array length $(length(A))"))
    end
    GPUArray(buffer(A), context(A), newdims)
end

function reinterpret{T, S, N}(::Type{T}, A::GPUArray{S}, dims::NTuple{N, Int})
    if !isbits(T)
        throw(ArgumentError("cannot reinterpret Array{$(S)} to ::Type{Array{$(T)}}, type $(T) is not a bits type"))
    end
    if !isbits(S)
        throw(ArgumentError("cannot reinterpret Array{$(S)} to ::Type{Array{$(T)}}, type $(S) is not a bits type"))
    end
    nel = div(length(a) * sizeof(S), sizeof(T))
    if prod(dims) != nel
        throw(DimensionMismatch("new dimensions $(dims) must be consistent with array size $(nel)"))
    end
    GPUArray(unsafe_reinterpret(T, buffer(A), dims), dims, context(A))
end

"""
This updates an array, even if dimensions and sizes don't match.
Will resize accordingly!
"""
function update!{T, N}(A::GPUArray{T, N}, value::Array{T, N})
    size(A) != size(value) && resize!(A, size(value))
    copy!(A, value)
    return
end

function getindex{T, N}(A::GPUArray{T, N}, indices...)
    getindex(A, to_range.(indices))
end
function getindex{T, N}(A::GPUArray{T, N}, ranges::UnitRange...)
    checkbounds(A, ranges...)
    result = Array{T, N}(length.(ranges))
    copy!(result, A, ranges...)
end


function grow_dimensions(
        real_length::Int, _size::Int, additonal_size::Int,
        growfactor::Real = 1.5
    )
    new_dim = round(Int, real_length * growfactor)
    return max(new_dim, additonal_size + _size)
end


push!{T}(v::GPUVector{T}, x) = push!(v, convert(T, x))
push!{T}(v::GPUVector{T}, x::T) = append!(v, [x])
push!{T}(v::GPUVector{T}, x::T...) = append!(v, [x...])

function append!{T}(v::GPUVector{T}, value)
    x = reasonable_convert(Vector{T}, value)
    lv, lx = length(v), length(x)
    real_length = length(buffer(v))
    if (v.real_length < lv + lx)
        resize!(v, grow_dimensions(real_length, lv, lx))
    end
    v[(lv + 1) : (lv + lx)] = value
    v.size = (lv + lx,)
    v
end


function grow_at(v::GPUVector, index::Int, amount::Int)
    resize!(v, length(v) + amount)
    copy!(v, index, v, index + amount, amount)
end

splice!{T}(v::GPUVector{T}, index::Int, x::T) = (v[index] = x)
function splice!{T}(v::GPUVector{T}, index::Int, x::Vector = T[])
    splice!(v, index:index, map(T, x))
end

function splice!{T}(v::GPUVector{T}, index::UnitRange, x::Vector=T[])
    lenv = length(v)
    elements_to_grow = length(x) - length(index) # -1
    buffer = similar(v.buffer, length(v) + elements_to_grow)
    copy!(v.buffer, 1, buffer, 1, first(index) - 1) # copy first half
    copy!(v.buffer, last(index) + 1, buffer, first(index) + length(x), lenv - last(index)) # shift second half
    v.buffer = buffer
    v.real_length = length(buffer)
    v.size = (v.real_length,)
    copy!(x, 1, buffer, first(index), length(x)) # copy contents of insertion vector
    return
end
