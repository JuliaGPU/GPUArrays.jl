# host-level indexing


# basic indexing with integers

Base.IndexStyle(::Type{<:AbstractGPUArray}) = Base.IndexLinear()

function Base.getindex(xs::AbstractGPUArray{T}, I::Integer...) where T
    assertscalar("getindex")
    i = Base._to_linear_index(xs, I...)
    x = Array{T}(undef, 1)
    copyto!(x, 1, xs, i, 1)
    return x[1]
end

function Base.setindex!(xs::AbstractGPUArray{T}, v::T, I::Integer...) where T
    assertscalar("setindex!")
    i = Base._to_linear_index(xs, I...)
    x = T[v]
    copyto!(xs, i, x, 1, 1)
    return xs
end

Base.setindex!(xs::AbstractGPUArray, v, I::Integer...) =
    setindex!(xs, convert(eltype(xs), v), I...)


# basic indexing with cartesian indices

Base.@propagate_inbounds Base.getindex(A::AbstractGPUArray, I::Union{Integer, CartesianIndex}...) =
    A[Base.to_indices(A, I)...]
Base.@propagate_inbounds Base.setindex!(A::AbstractGPUArray, v, I::Union{Integer, CartesianIndex}...) =
    (A[Base.to_indices(A, I)...] = v; A)


# generalized multidimensional indexing

Base.getindex(A::AbstractGPUArray, I...) = _getindex(A, to_indices(A, I)...)

function _getindex(src::AbstractGPUArray, Is...)
    shape = Base.index_shape(Is...)
    dest = similar(src, shape)
    any(isempty, Is) && return dest # indexing with empty array
    idims = map(length, Is)

    AT = typeof(src).name.wrapper
    # NOTE: we are pretty liberal here supporting non-GPU indices...
    gpu_call(getindex_kernel, dest, src, idims, adapt(AT, Is)...)
    return dest
end

@generated function getindex_kernel(ctx::AbstractKernelContext, dest, src, idims,
                                    Is::Vararg{Any,N}) where {N}
    quote
        i = @linearidx dest
        is = @inbounds CartesianIndices(idims)[i]
        @nexprs $N i -> I_i = @inbounds(Is[i][is[i]])
        val = @ncall $N getindex src i -> I_i
        @inbounds dest[i] = val
        return
    end
end

Base.setindex!(A::AbstractGPUArray, v, I...) = _setindex!(A, v, to_indices(A, I)...)

function _setindex!(dest::AbstractGPUArray, src, Is...)
    isempty(Is) && return dest
    idims = length.(Is)
    len = prod(idims)
    len==0 && return dest

    AT = typeof(dest).name.wrapper
    # NOTE: we are pretty liberal here supporting non-GPU sources and indices...
    gpu_call(setindex_kernel, dest, adapt(AT, src), idims, len, adapt(AT, Is)...;
             elements=len)
    return dest
end

@generated function setindex_kernel(ctx::AbstractKernelContext, dest, src, idims, len,
                                    Is::Vararg{Any,N}) where {N}
    quote
        i = linear_index(ctx)
        i > len && return
        is = @inbounds CartesianIndices(idims)[i]
        @nexprs $N i -> I_i = @inbounds(Is[i][is[i]])
        @ncall $N setindex! dest src[i] i -> I_i
        return
    end
end


## find*

function Base.findfirst(f::Function, xs::AnyGPUArray)
    indx = ndims(xs) == 1 ? (eachindex(xs), 1) :
    (CartesianIndices(xs), CartesianIndex{ndims(xs)}())
    function g(t1, t2)
        (x, i), (y, j) = t1, t2
        if i > j
            t1, t2 = t2, t1
            (x, i), (y, j) = t1, t2
        end
        x && return t1
        y && return t2
        return (false, indx[2])
    end

    res = mapreduce((x, y)->(f(x), y), g, xs, indx[1]; init = (false, indx[2]))
    res[1] === true && return res[2]
    return nothing
end

Base.findfirst(xs::AnyGPUArray{Bool}) = findfirst(identity, xs)

function findminmax(minmax, binop, a::AnyGPUArray; init, dims)
    function f(t1::Tuple{<:AbstractFloat,<:Any}, t2::Tuple{<:AbstractFloat,<:Any})
        (x, i), (y, j) = t1, t2
        if i > j
            t1, t2 = t2, t1
            (x, i), (y, j) = t1, t2
        end

        # Check for NaN first because NaN == NaN is false
        isnan(x) && return t1
        isnan(y) && return t2
        minmax(x, y) == x && return t1
        return t2
    end

    function f(t1, t2)
        (x, i), (y, j) = t1, t2

        binop(x, y) && return t1
        x == y && return (x, min(i, j))
        return t2
    end

    indx = ndims(a) == 1 ? (eachindex(a), 1) :
                           (CartesianIndices(a), CartesianIndex{ndims(a)}())
    if dims == Colon()
        mapreduce(tuple, f, a, indx[1]; init = (init, indx[2]))
    else
        res = mapreduce(tuple, f, a, indx[1];
                        init = (init, indx[2]), dims=dims)
        vals = map(x->x[1], res)
        inds = map(x->x[2], res)
        return (vals, inds)
    end
end

Base.findmax(a::AnyGPUArray; dims=:) = findminmax(max, >, a; init=typemin(eltype(a)), dims)
Base.findmin(a::AnyGPUArray; dims=:) = findminmax(min, <, a; init=typemax(eltype(a)), dims)
