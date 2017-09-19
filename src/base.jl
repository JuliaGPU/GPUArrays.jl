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
#     gpu_call(dest, (Cuint(dim), dest, xs)) do state, dim, dest, xs
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
@pure @inline function gpu_ind2sub{T}(A::AbstractArray, ind::T)
    _ind2sub(size(A), ind - T(1))
end
@pure @inline function gpu_ind2sub{N, T}(dims::NTuple{N}, ind::T)
    _ind2sub(NTuple{N, T}(dims), ind - T(1))
end
@pure @inline _ind2sub{T}(::Tuple{}, ind::T) = (ind + T(1),)
@pure @inline function _ind2sub{T}(indslast::NTuple{1}, ind::T)
    ((ind + T(1)),)
end
@pure @inline function _ind2sub{T}(inds, ind::T)
    r1 = inds[1]
    indnext = div(ind, r1)
    f = T(1); l = r1
    (ind-l*indnext+f, _ind2sub(Base.tail(inds), indnext)...)
end

@pure function gpu_sub2ind{N, N2, T}(dims::NTuple{N}, I::NTuple{N2, T})
    Base.@_inline_meta
    _sub2ind(NTuple{N, T}(dims), T(1), T(1), I...)
end
_sub2ind(x, L, ind) = ind
function _sub2ind{T}(::Tuple{}, L, ind, i::T, I::T...)
    Base.@_inline_meta
    ind + (i - T(1)) * L
end
function _sub2ind(inds, L, ind, i::IT, I::IT...) where IT
    Base.@_inline_meta
    r1 = inds[1]
    _sub2ind(Base.tail(inds), L * r1, ind + (i - IT(1)) * L, I...)
end
