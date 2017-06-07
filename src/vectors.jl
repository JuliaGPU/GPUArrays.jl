import Base: copy!, splice!, append!, push!, setindex!, start, next, done
import Base: getindex, map, length, eltype, endof, ndims, size, resize!
resize!(A::GPUArray, newdims::Int...) = resize!(A, newdims)
function resize!{T}(A::GPUArray{T, 1}, newdims::NTuple{1, Int})
    newdims == size(A) && return A
    newlength = newdims[1]
    real_length = length(buffer(A))
    if real_length >= newlength # underlying buffer already big enough
        A.size = newdims
        return A
    else
        B = similar(A, newdims)
        if length(A) > 0
            max_len = min(length(A), newlength) #might also shrink
            copy!(B, (1:max_len,), A, (1:max_len,))
        end
        A.size = newdims
        A.buffer = buffer(B)
        return A
    end
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


function grow_dimensions(
        real_length::Int, _size::Int, additonal_size::Int,
        growfactor::Real = 1.5
    )
    new_dim = round(Int, real_length * growfactor)
    return max(new_dim, additonal_size + _size)
end

const GPUVector{T} = GPUArray{T, 1}

push!{T}(v::GPUVector{T}, x) = push!(v, convert(T, x))
push!{T}(v::GPUVector{T}, x::T) = append!(v, [x])
push!{T}(v::GPUVector{T}, x::T...) = append!(v, [x...])

function append!{T}(v::GPUVector{T}, value)
    x = array_convert(Vector{T}, value)
    lv, lx = length(v), length(x)
    real_length = length(buffer(v))
    if real_length < (lv + lx)
        resize!(v, grow_dimensions(real_length, lv, lx))
    end
    v.size = (lv + lx,)
    v[(lv + 1) : (lv + lx)] = value
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
    buffer = similar(buffer(v), length(v) + elements_to_grow)
    copy!(v.buffer, 1, buffer, 1, first(index) - 1) # copy first half
    copy!(v.buffer, last(index) + 1, buffer, first(index) + length(x), lenv - last(index)) # shift second half
    v.buffer = buffer
    v.real_length = length(buffer)
    v.size = (v.real_length,)
    copy!(x, 1, buffer, first(index), length(x)) # copy contents of insertion vector
    return
end
