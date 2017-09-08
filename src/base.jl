function Base.permutedims!(dest::GPUArray, src::GPUArray, perm)
  gpu_call(kernel, dest, (dest, src, perm)) do state, dest, src, perm
    I = @gpuindex dest
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
  x, ntuple(n -> n == dim ? i : I[n], Val{N})
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

function Base.cat_t(dims::Integer, T::Type, x::GPUArray, xs::GPUArray...)
  catdims = Base.dims2cat(dims)
  shape = Base.cat_shape(catdims, (), size.((x, xs...))...)
  dest = Base.cat_similar(x, T, shape)
  _cat(dims, dest, x, xs...)
end

Base.vcat(xs::GPUArray...) = cat(1, xs...)
Base.hcat(xs::GPUArray...) = cat(2, xs...)
