import Base: count, map!, permutedims!, cat_t, vcat, hcat

count(pred, A::GPUArray) = Int(mapreduce(pred, +, Cuint(0), A))

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


function permutedims!(dest::GPUArray, src::GPUArray, perm)
    gpu_call(dest, (dest, src, perm)) do state, dest, src, perm
        I = @cartesianidx dest
        @inbounds dest[I...] = src[genperm(I, perm)...]
        return
    end
    return dest
end


@generated function nindex(i::Int, ls::NTuple{N}) where N
    quote
        Base.@_inline_meta
        $(foldr((n, els) -> :(i ≤ ls[$n] ? ($n, i) : (i -= ls[$n]; $els)), :(-1, -1), 1:N))
    end
end

function catindex(dim, I::NTuple{N}, shapes) where N
    @inbounds x, i = nindex(I[dim], getindex.(shapes, dim))
    x, ntuple(n -> n == dim ? Cuint(i) : I[n], Val{N})
end

function _cat(dim, dest, xs...)
    gpu_call(kernel, dest, (dim, dest, xs)) do state, dim, dest, xs
        I = @cartesianidx dest state
        n, I′ = catindex(dim, I, size.(xs))
        @inbounds dest[I...] = xs[n][I′...]
        return
    end
    return dest
end

function cat_t(dims::Integer, T::Type, x::GPUArray, xs::GPUArray...)
    catdims = Base.dims2cat(dims)
    shape = Base.cat_shape(catdims, (), size.((x, xs...))...)
    dest = Base.cat_similar(x, T, shape)
    _cat(dims, dest, x, xs...)
end

vcat(xs::GPUArray...) = cat(1, xs...)
hcat(xs::GPUArray...) = cat(2, xs...)
