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
    if length(dest)!=length(src)
        if length(src)==1
            throw(ArgumentError("indexed assignment with a single value to possibly many locations is not supported; perhaps use broadcasting `.=` instead?"))
        else
            throw(ArgumentError("indexed assignment with different lengths not supported; array sizes "*string(size(src))*" and "*string(size(dest)))) 
        end
    end
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

# simple array type that returns the index used to access an element, while
# retaining the dimensionality of the original array. this can be used to
# broadcast or reduce an array together with its indices, whereas normally
# combining e.g. a 2x2 array with its 4-element eachindex array would result
# in a 4x4 broadcast or reduction.
struct EachIndex{T,N,IS} <: AbstractArray{T,N}
    dims::NTuple{N,Int}
    indices::IS
end
EachIndex(xs::AbstractArray) =
    EachIndex{typeof(firstindex(xs)), ndims(xs), typeof(eachindex(xs))}(
              size(xs), eachindex(xs))
Base.size(ei::EachIndex) = ei.dims
Base.getindex(ei::EachIndex, i::Int) = ei.indices[i]
Base.IndexStyle(::Type{<:EachIndex}) = Base.IndexLinear()

function Base.findfirst(f::Function, xs::AnyGPUArray)
    indices = EachIndex(xs)
    dummy_index = first(indices)

    # given two pairs of (istrue, index), return the one with the smallest index
    function reduction(t1, t2)
        (x, i), (y, j) = t1, t2
        if i > j
            t1, t2 = t2, t1
            (x, i), (y, j) = t1, t2
        end
        x && return t1
        y && return t2
        return (false, dummy_index)
    end

    res = mapreduce((x, y)->(f(x), y), reduction, xs, indices;
                    init = (false, dummy_index))
    if res[1]
        # out of consistency with Base.findarray, return a CartesianIndex
        # when the input is a multidimensional array
        ndims(xs) == 1 && return res[2]
        return CartesianIndices(xs)[res[2]]
    else
        return nothing
    end
end

Base.findfirst(xs::AnyGPUArray{Bool}) = findfirst(identity, xs)

function findminmax(binop, xs::AnyGPUArray; init, dims)
    indices = EachIndex(xs)
    dummy_index = firstindex(xs)

    function reduction(t1, t2)
        (x, i), (y, j) = t1, t2

        binop(x, y) && return t2
        x == y && return (x, min(i, j))
        return t1
    end

    @static if VERSION < v"1.7.0-DEV.119"
    # before JuliaLang/julia#35316, isless/isgreated did not order NaNs last
    function reduction(t1::Tuple{<:AbstractFloat,<:Any}, t2::Tuple{<:AbstractFloat,<:Any})
        (x, i), (y, j) = t1, t2

        isnan(x) && return t1
        isnan(y) && return t2

        binop(x, y) && return t2
        x == y && return (x, min(i, j))
        return t1
    end
    end

    if dims == Colon()
        res = mapreduce(tuple, reduction, xs, indices; init = (init, dummy_index))

        # out of consistency with Base.findarray, return a CartesianIndex
        # when the input is a multidimensional array
        return (res[1], ndims(xs) == 1 ? res[2] : CartesianIndices(xs)[res[2]])
    else
        res = mapreduce(tuple, reduction, xs, indices;
                        init = (init, dummy_index), dims=dims)
        vals = map(x->x[1], res)
        inds = map(x->ndims(xs) == 1 ? x[2] : CartesianIndices(xs)[x[2]], res)
        return (vals, inds)
    end
end

Base.findmax(a::AnyGPUArray; dims=:) = findminmax(Base.isless, a; init=typemin(eltype(a)), dims)
Base.findmin(a::AnyGPUArray; dims=:) = findminmax(Base.isgreater, a; init=typemax(eltype(a)), dims)
