# common Base functionality

allequal(x) = true
allequal(x, y, z...) = x == y && allequal(y, z...)
function Base.map!(f, y::GPUArray, xs::GPUArray...)
    @assert allequal(size.((y, xs...))...)
    return y .= f.(xs...)
end
function Base.map(f, y::GPUArray, xs::GPUArray...)
    @assert allequal(size.((y, xs...))...)
    return f.(y, xs...)
end

# Break ambiguities with base
Base.map!(f, y::GPUArray) =
    invoke(map!, Tuple{Any,GPUArray,Vararg{GPUArray}}, f, y)
Base.map!(f, y::GPUArray, x::GPUArray) =
    invoke(map!, Tuple{Any,GPUArray, Vararg{GPUArray}}, f, y, x)
Base.map!(f, y::GPUArray, x1::GPUArray, x2::GPUArray) =
    invoke(map!, Tuple{Any,GPUArray, Vararg{GPUArray}}, f, y, x1, x2)


# Base functions that are sadly not fit for the the GPU yet (they only work for Int64)
Base.@pure @inline function gpu_ind2sub(A::AbstractArray, ind::T) where T
    _ind2sub(size(A), ind - T(1))
end
Base.@pure @inline function gpu_ind2sub(dims::NTuple{N}, ind::T) where {N, T}
    _ind2sub(NTuple{N, T}(dims), ind - T(1))
end
Base.@pure @inline _ind2sub(::Tuple{}, ind::T) where {T} = (ind + T(1),)
Base.@pure @inline function _ind2sub(indslast::NTuple{1}, ind::T) where T
    ((ind + T(1)),)
end
Base.@pure @inline function _ind2sub(inds, ind::T) where T
    r1 = inds[1]
    indnext = div(ind, r1)
    f = T(1); l = r1
    (ind-l*indnext+f, _ind2sub(Base.tail(inds), indnext)...)
end

Base.@pure function gpu_sub2ind(dims::NTuple{N}, I::NTuple{N2, T}) where {N, N2, T}
    Base.@_inline_meta
    _sub2ind(NTuple{N, T}(dims), T(1), T(1), I...)
end
_sub2ind(x, L, ind) = ind
function _sub2ind(::Tuple{}, L, ind, i::T, I::T...) where T
    Base.@_inline_meta
    ind + (i - T(1)) * L
end
function _sub2ind(inds, L, ind, i::IT, I::IT...) where IT
    Base.@_inline_meta
    r1 = inds[1]
    _sub2ind(Base.tail(inds), L * r1, ind + (i - IT(1)) * L, I...)
end

# This is pretty ugly, but I feel bad to add those to device arrays, since
# we're never bound checking... So getindex(a::GPUVector, 10, 10) would silently go unnoticed
# we need this here for easier implementation of repeat
@inline Base.@propagate_inbounds getidx_2d1d(x::AbstractVector, i, j) = x[i]
@inline Base.@propagate_inbounds getidx_2d1d(x::AbstractMatrix, i, j) = x[i, j]

function Base.repeat(a::GPUVecOrMat, m::Int, n::Int = 1)
    o, p = size(a, 1), size(a, 2)
    b = similar(a, o*m, p*n)
    gpu_call(a, (b, a, o, p, m, n), n) do state, b, a, o, p, m, n
        j = linear_index(state)
        j > n && return
        d = (j - 1) * p + 1
        @inbounds for i in 1:m
            c = (i - 1) * o + 1
            for r in 1:p
                for k in 1:o
                    b[k - 1 + c, r - 1 + d] = getidx_2d1d(a, k, r)
                end
            end
        end
        return
    end
    return b
end

function Base.repeat(a::GPUVector, m::Int)
    o = length(a)
    b = similar(a, o*m)
    gpu_call(a, (b, a, o, m), m) do state, b, a, o, m
        i = linear_index(state)
        i > m && return
        c = (i - 1)*o + 1
        @inbounds for i in 1:o
            b[c + i - 1] = a[i]
        end
        return
    end
    return b
end
