import Base: count, map!, permutedims!, cat_t, vcat, hcat
using Base: @pure

allequal(x) = true
allequal(x, y, z...) = x == y && allequal(y, z...)
function map!(f, y::GPUArray, xs::GPUArray...)
    @assert allequal(size.((y, xs...))...)
    return y .= f.(xs...)
end
function map(f, y::GPUArray, xs::GPUArray...)
    @assert allequal(size.((y, xs...))...)
    return f.(y, xs...)
end

# Break ambiguities with base
map!(f, y::GPUArray) =
    invoke(map!, Tuple{Any,GPUArray,Vararg{GPUArray}}, f, y)
map!(f, y::GPUArray, x::GPUArray) =
    invoke(map!, Tuple{Any,GPUArray, Vararg{GPUArray}}, f, y, x)
map!(f, y::GPUArray, x1::GPUArray, x2::GPUArray) =
    invoke(map!, Tuple{Any,GPUArray, Vararg{GPUArray}}, f, y, x1, x2)


# TODO find out why this segfaults julia without stack trace on AMD
# produces wrong results on Titan X and passes on GTX 950..........

# @generated function nindex(i::T, ls::NTuple{N}) where {T, N}
#     quote
#         Base.@_inline_meta
#         $(foldr(:($T(0), $T(0)), T(1):T(N)) do n, els
#             :(i ≤ ls[$n] ? ($T($n), i) : (i -= $T(ls[$n]); $els))
#         end)
#     end
# end
# function catindex(dim, I::NTuple{N, T}, shapes) where {T, N}
#     xi = nindex(I[dim], map(s-> s[dim], shapes))
#     x = xi[1]; i = xi[2]
#     x, ntuple(n -> n == dim ? i : I[n], Val{N})
# end
#
# function _cat(dim, dest, xs...)
#     gpu_call(dest, (UInt32(dim), dest, xs)) do state, dim, dest, xs
#         I = @cartesianidx dest state
#         nI = catindex(dim, I, size.(xs))
#         n = nI[1]; I′ = nI[2]
#         @inbounds dest[I...] = xs[n][I′...]
#         return
#     end
#     return dest
# end
#
# function cat_t(dims::Integer, T::Type, x::GPUArray, xs::GPUArray...)
#     catdims = Base.dims2cat(dims)
#     shape = Base.cat_shape(catdims, (), size.((x, xs...))...)
#     dest = Base.cat_similar(x, T, shape)
#     _cat(dims, dest, x, xs...)
# end
#
# vcat(xs::GPUArray...) = cat(1, xs...)
# hcat(xs::GPUArray...) = cat(2, xs...)


# Base functions that are sadly not fit for the the GPU yet (they only work for Int64)
@pure @inline function gpu_ind2sub(A::AbstractArray, ind::T) where T
    _ind2sub(size(A), ind - T(1))
end
@pure @inline function gpu_ind2sub(dims::NTuple{N}, ind::T) where {N, T}
    _ind2sub(NTuple{N, T}(dims), ind - T(1))
end
@pure @inline _ind2sub(::Tuple{}, ind::T) where {T} = (ind + T(1),)
@pure @inline function _ind2sub(indslast::NTuple{1}, ind::T) where T
    ((ind + T(1)),)
end
@pure @inline function _ind2sub(inds, ind::T) where T
    r1 = inds[1]
    indnext = div(ind, r1)
    f = T(1); l = r1
    (ind-l*indnext+f, _ind2sub(Base.tail(inds), indnext)...)
end

@pure function gpu_sub2ind(dims::NTuple{N}, I::NTuple{N2, T}) where {N, N2, T}
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

@inline Base.@propagate_inbounds getidx_2d1d(x::AbstractVector, i, j) = x[i]
@inline Base.@propagate_inbounds getidx_2d1d(x::AbstractMatrix, i, j) = x[i, j]

function Base.repmat(a::GPUVecOrMat, m::Int, n::Int = 1)
    o, p = size(a, 1), size(a, 2)
    b = similar(a, o*m, p*n)
    args = (b, a, UInt32.((o, p, m, n))...)
    gpu_call(a, args, n) do state, b, a, o, p, m, n
        j = linear_index(state)
        j > n && return
        ui1 = UInt32(1)
        d = (j - ui1) * p + ui1
        @inbounds for i in ui1:m
            c = (i - ui1) * o + ui1
            for r in ui1:p
                for k in ui1:o
                    b[k - ui1 + c, r - ui1 + d] = getidx_2d1d(a, k, r)
                end
            end
        end
        return
    end
    return b
end

function Base.repmat(a::GPUVector, m::Int)
    o = length(a)
    b = similar(a, o*m)
    gpu_call(a, (b, a, UInt32(o), UInt32(m)), m) do state, b, a, o, m
        i = linear_index(state)
        i > m && return
        ui1 = UInt32(1)
        c = (i - ui1)*o + ui1
        @inbounds for i in ui1:o
            b[c + i - ui1] = a[i]
        end
        return
    end
    return b
end
