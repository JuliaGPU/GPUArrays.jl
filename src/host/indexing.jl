# host-level indexing


# indexing operators

Base.IndexStyle(::Type{<:AbstractGPUArray}) = Base.IndexLinear()

vectorized_indices(Is::Union{Integer,CartesianIndex}...) = Val{false}()
vectorized_indices(Is...) = Val{true}()

# TODO: re-use Base functionality for the conversion of indices to a linear index,
#       by only implementing `getindex(A, ::Int)` etc. this is difficult due to
#       ambiguities with the vectorized method that can take any index type.

Base.@propagate_inbounds Base.getindex(A::AbstractGPUArray, Is...) =
    _getindex(vectorized_indices(Is...), A, to_indices(A, Is)...)
Base.@propagate_inbounds _getindex(::Val{false}, A::AbstractGPUArray, Is...) =
    scalar_getindex(A, to_indices(A, Is)...)
Base.@propagate_inbounds _getindex(::Val{true}, A::AbstractGPUArray, Is...) =
    vectorized_getindex(A, to_indices(A, Is)...)

Base.@propagate_inbounds Base.setindex!(A::AbstractGPUArray, v, Is...) =
    _setindex!(vectorized_indices(Is...), A, v, to_indices(A, Is)...)
Base.@propagate_inbounds _setindex!(::Val{false}, A::AbstractGPUArray, v, Is...) =
    scalar_setindex!(A, v, to_indices(A, Is)...)
Base.@propagate_inbounds _setindex!(::Val{true}, A::AbstractGPUArray, v, Is...) =
    vectorized_setindex!(A, v, to_indices(A, Is)...)

## scalar indexing

function scalar_getindex(A::AbstractGPUArray{T}, Is...) where T
    @boundscheck checkbounds(A, Is...)
    I = Base._to_linear_index(A, Is...)
    getindex(A, I)
end

function scalar_setindex!(A::AbstractGPUArray{T}, v, Is...) where T
    @boundscheck checkbounds(A, Is...)
    I = Base._to_linear_index(A, Is...)
    setindex!(A, v, I)
end

# we still dispatch to `Base.getindex(a, ::Int)` etc so that there's a single method to
# override when a back-end (e.g. with unified memory) wants to allow scalar indexing.

function Base.getindex(A::AbstractGPUArray{T}, I::Int) where T
    @boundscheck checkbounds(A, I)
    assertscalar("getindex")
    x = Array{T}(undef, 1)
    copyto!(x, 1, A, I, 1)
    return x[1]
end

function Base.setindex!(A::AbstractGPUArray{T}, v, I::Int) where T
    @boundscheck checkbounds(A, I)
    assertscalar("setindex!")
    x = T[v]
    copyto!(A, I, x, 1, 1)
    return A
end

## vectorized indexing

function vectorized_checkbounds(src, Is)
    # Base's boundscheck accesses the indices, so make sure they reside on the CPU.
    # this is expensive, but it's a bounds check after all.
    Is_cpu = map(I->adapt(BackToCPU(), I), Is)
    checkbounds(src, Is_cpu...)
end

function vectorized_getindex(src::AbstractGPUArray, Is...)
    @boundscheck vectorized_checkbounds(src, Is)
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

function vectorized_setindex!(dest::AbstractGPUArray, src, Is...)
    @boundscheck vectorized_checkbounds(dest, Is)
    isempty(Is) && return dest
    idims = length.(Is)
    len = prod(idims)
    len==0 && return dest
    if length(src) != len
        if length(src) == 1
            throw(ArgumentError("indexed assignment with a single value to possibly many locations is not supported; perhaps use broadcasting `.=` instead?"))
        else
            throw(DimensionMismatch("dimensions must match: a has "*string(length(src))*" elements, b has  "*string(len)))
        end
    end

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


# find*

# simple array type that returns the index used to access an element, while
# retaining the dimensionality of the original array. this can be used to
# broadcast or reduce an array together with its indices, whereas normally
# combining e.g. a 2x2 array with its 4-element eachindex array would result
# in a 4x4 broadcast or reduction.
struct EachIndex{T,N,IS} <: AbstractArray{T,N}
    dims::NTuple{N,Int}
    indices::IS
end
EachIndex(A::AbstractArray) =
    EachIndex{typeof(firstindex(A)), ndims(A), typeof(eachindex(A))}(
              size(A), eachindex(A))
Base.size(ei::EachIndex) = ei.dims
Base.getindex(ei::EachIndex, i::Int) = ei.indices[i]
Base.IndexStyle(::Type{<:EachIndex}) = Base.IndexLinear()

function Base.findfirst(f::Function, A::AnyGPUArray)
    indices = EachIndex(A)
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

    res = mapreduce((x, y)->(f(x), y), reduction, A, indices;
                    init = (false, dummy_index))
    if res[1]
        # out of consistency with Base.findarray, return a CartesianIndex
        # when the input is a multidimensional array
        ndims(A) == 1 && return res[2]
        return CartesianIndices(A)[res[2]]
    else
        return nothing
    end
end

Base.findfirst(A::AnyGPUArray{Bool}) = findfirst(identity, A)

function findminmax(binop, A::AnyGPUArray; init, dims)
    indices = EachIndex(A)
    dummy_index = firstindex(A)

    function reduction(t1, t2)
        (x, i), (y, j) = t1, t2

        binop(x, y) && return t2
        x == y && return (x, min(i, j))
        return t1
    end

    if dims == Colon()
        res = mapreduce(tuple, reduction, A, indices; init = (init, dummy_index))

        # out of consistency with Base.findarray, return a CartesianIndex
        # when the input is a multidimensional array
        return (res[1], ndims(A) == 1 ? res[2] : CartesianIndices(A)[res[2]])
    else
        res = mapreduce(tuple, reduction, A, indices;
                        init = (init, dummy_index), dims=dims)
        vals = map(x->x[1], res)
        inds = map(x->ndims(A) == 1 ? x[2] : CartesianIndices(A)[x[2]], res)
        return (vals, inds)
    end
end

Base.findmax(a::AnyGPUArray; dims=:) = findminmax(Base.isless, a; init=typemin(eltype(a)), dims)
Base.findmin(a::AnyGPUArray; dims=:) = findminmax(Base.isgreater, a; init=typemax(eltype(a)), dims)
