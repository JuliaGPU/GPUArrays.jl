# host-level indexing

using Base: @propagate_inbounds


# indexing operators

Base.IndexStyle(::Type{<:AbstractGPUArray}) = Base.IndexLinear()

vectorized_indices(Is::Union{Integer,CartesianIndex}...) = Val{false}()
vectorized_indices(Is...) = Val{true}()

# TODO: re-use Base functionality for the conversion of indices to a linear index,
#       by only implementing `getindex(A, ::Int)` etc. this is difficult due to
#       ambiguities with the vectorized method that can take any index type.

@propagate_inbounds Base.getindex(A::AbstractGPUArray, Is...) =
    _getindex(vectorized_indices(Is...), A, to_indices(A, Is)...)
@propagate_inbounds _getindex(::Val{false}, A::AbstractGPUArray, Is...) =
    scalar_getindex(A, to_indices(A, Is)...)
@propagate_inbounds _getindex(::Val{true}, A::AbstractGPUArray, Is...) =
    vectorized_getindex(A, to_indices(A, Is)...)

@propagate_inbounds Base.setindex!(A::AbstractGPUArray, v, Is...) =
    _setindex!(vectorized_indices(Is...), A, v, to_indices(A, Is)...)
@propagate_inbounds _setindex!(::Val{false}, A::AbstractGPUArray, v, Is...) =
    scalar_setindex!(A, v, to_indices(A, Is)...)
@propagate_inbounds _setindex!(::Val{true}, A::AbstractGPUArray, v, Is...) =
    vectorized_setindex!(A, v, to_indices(A, Is)...)

## scalar indexing

@propagate_inbounds function scalar_getindex(A::AbstractGPUArray{T}, Is...) where T
    @boundscheck checkbounds(A, Is...)
    I = Base._to_linear_index(A, Is...)
    getindex(A, I)
end

@propagate_inbounds function scalar_setindex!(A::AbstractGPUArray{T}, v, Is...) where T
    @boundscheck checkbounds(A, Is...)
    I = Base._to_linear_index(A, Is...)
    setindex!(A, v, I)
end

# we still dispatch to `Base.getindex(a, ::Int)` etc so that there's a single method to
# override when a back-end (e.g. with unified memory) wants to allow scalar indexing.

@propagate_inbounds function Base.getindex(A::AbstractGPUArray{T}, I::Int) where T
    @boundscheck checkbounds(A, I)
    assertscalar("getindex")
    x = Array{T}(undef, 1)
    copyto!(x, 1, A, I, 1)
    return x[1]
end

@propagate_inbounds function Base.setindex!(A::AbstractGPUArray{T}, v, I::Int) where T
    @boundscheck checkbounds(A, I)
    assertscalar("setindex!")
    x = T[v]
    copyto!(A, I, x, 1, 1)
    return A
end

## vectorized indexing

@propagate_inbounds function vectorized_getindex!(dest::AbstractGPUArray,
                                                  src::AbstractArray, Is...)
    any(isempty, Is) && return dest # indexing with empty array
    idims = map(length, Is)

    # NOTE: we are pretty liberal here supporting non-GPU indices...
    Is = map(adapt(ToGPU(dest)), Is)
    @boundscheck checkbounds(src, Is...)

    getindex_kernel(get_backend(dest))(dest, src, idims, Is...; ndrange=size(dest))
    return dest
end

@propagate_inbounds function vectorized_getindex(src::AbstractGPUArray, Is...)
    shape = Base.index_shape(Is...)
    dest = similar(src, shape)
    return vectorized_getindex!(dest, src, Is...)
end

@kernel function getindex_kernel(dest, src, idims, Is...)
    i = @index(Global, Linear)
    getindex_generated(dest, src, idims, i, Is...)
end
@generated function getindex_generated(dest, src, idims, i, Is::Vararg{Any,N}) where {N}
    quote
        is = @inbounds CartesianIndices(idims)[i]
        @nexprs $N i -> I_i = @inbounds(Is[i][is[i]])
        val = @ncall $N getindex src i -> I_i
        @inbounds dest[i] = val
    end
end

@propagate_inbounds function vectorized_setindex!(dest::AbstractArray, src, Is...)
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

    # NOTE: we are pretty liberal here supporting non-GPU indices...
    Is = map(adapt(ToGPU(dest)), Is)
    @boundscheck checkbounds(dest, Is...)

    # one work item per index, not per destination element: with repeated indices
    # there can be more assignments than elements, and each of them must be performed
    setindex_kernel(get_backend(dest))(dest, adapt(ToGPU(dest), src), idims, len, Is...;
             ndrange = len)
    return dest
end

@kernel function setindex_kernel(dest, src, idims, len, Is...)
    i = @index(Global, Linear)
    setindex_generated(dest, src, idims, len, i, Is...)
end
@generated function setindex_generated(dest, src, idims, len, i, Is::Vararg{Any,N}) where {N}
    quote
        i > len && return
        is = @inbounds CartesianIndices(idims)[i]
        @nexprs $N i -> I_i = @inbounds(Is[i][is[i]])
        @ncall $N setindex! dest src[i] i -> I_i
        return
    end
end


# bounds checking

# indices residing on the GPU should be bounds-checked on the GPU to avoid iteration.

# not all wrapped GPU arrays make sense as indices, so we use a subset of `AnyGPUArray`
const IndexGPUArray{T} = Union{AbstractGPUArray{T},
                               SubArray{T, <:Any, <:AbstractGPUArray},
                               LinearAlgebra.Adjoint{T}}

@inline function Base.checkindex(::Type{Bool}, inds::AbstractUnitRange, I::IndexGPUArray)
    all(broadcast(I) do i
        Base.checkindex(Bool, inds, i)
    end)
end
# ... except for a logical mask, which must have the indexed axes (Base's rule)
Base.checkindex(::Type{Bool}, inds::AbstractUnitRange, I::IndexGPUArray{Bool}) =
    ndims(I) == 1 && Base.axes1(I) == inds
Base.checkindex(::Type{Bool}, inds::Tuple, I::IndexGPUArray{Bool}) =
    length(inds) == ndims(I) && all(map(==, inds, axes(I)))

@inline function Base.checkindex(::Type{Bool}, inds::Tuple,
                                 I::IndexGPUArray{<:CartesianIndex})
    all(broadcast(I) do i
        Base.checkbounds_indices(Bool, inds, (i,))
    end)
end

## Vectorized index overloading for `WrappedGPUArray`
# We'd better not to overload `getindex`/`setindex!` directly as otherwise
# the ambiguities from the default scalar fallback become a mess.
# The default `getindex` for `AbstractArray` follows a `similar`-`copyto!` style.
# Thus we only dispatch the `copyto!` part (`Base._unsafe_getindex!`) to our implement.
function Base._unsafe_getindex!(dest::AbstractGPUArray, src::AbstractArray, Is::Vararg{Union{Real, AbstractArray}, N}) where {N}
    return vectorized_getindex!(dest, src, Base.ensure_indexable(Is)...)
end
# Similar for `setindex!`, its default fallback is equivalent to `copyto!`.
# We only dispatch the `copyto!` part (`Base._unsafe_setindex!`) to our implement.
# Also cover the outer `ReshapedArray` that `_maybe_reshape` produces when the parent
# is a `WrappedGPUArray`. Keeping this in the same `Union` as `WrappedGPUArray` (rather
# than as a second method) avoids the dispatch ambiguity from #587: the two signatures
# would otherwise overlap (`WrappedGPUArray` already includes some `ReshapedArray`s)
# without either being a strict subtype of the other.
function Base._unsafe_setindex!(::IndexStyle, A::Union{
            WrappedGPUArray,
            Base.ReshapedArray{<:Any, <:Any, <:WrappedGPUArray},
        }, x, Is::Vararg{Union{Real,AbstractArray}, N}) where N
    return vectorized_setindex!(A, x, Base.ensure_indexable(Is)...)
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

function findfirstlast_reduction(op_and_dummy, t1, t2)
    op, dummy_index = op_and_dummy
    (x, i), (y, j) = t1, t2
    if op(i, j)
        t1, t2 = t2, t1
        (x, i), (y, j) = t1, t2
    end
    x && return t1
    y && return t2
    return (false, dummy_index)
end

for (find_f, op, dummy_f) in ((:(Base.findfirst), :>, :first), (:(Base.findlast), :<, :last))
    @eval begin
        function $find_f(f::Function, A::AnyGPUArray)
            isempty(A) && return nothing
            indices = EachIndex(A)
            dummy_index = $dummy_f(indices)

            # given two pairs of (istrue, index), return the one with the smallest index
            res = mapreduce((x, y)->(f(x), y), (a, b)->findfirstlast_reduction(($op, dummy_index), a, b), A, indices;
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
    end
end
Base.findfirst(A::AnyGPUArray{Bool}) = findfirst(identity, A)
Base.findlast(A::AnyGPUArray{Bool})  = findlast(identity, A)

# findall, implemented by AcceleratedKernels, which selects `items` of the array's length: Base's
# indices, `keys(A)`, which are linear for vectors and Cartesian otherwise, except that Base's
# predicate form makes those of a 0-dimensional array linear
Base.findall(bools::AnyGPUArray{Bool}) = AK.findall(bools)
Base.findall(f::Function, A::AnyGPUArray) = AK.findall(f, A; items=_findall_items(A))
Base.findall(f::Base.Fix2{typeof(in)}, A::AnyGPUArray) =                    # (Base: `keys(A)`)
    AK.findall(f, A)
_findall_items(A) = ndims(A) == 0 ? LinearIndices(A) : keys(A)

# logical indexing: Base's `LogicalIndex` iterates, so the mask becomes the indices it selects.
# Those no longer carry the mask's shape, so a single mask is checked against the array first, as
# Base does; a mask mixed with other indices is not (as before).
Base.to_index(::AnyGPUArray, I::AbstractArray{Bool}) = findall(I)
@static if VERSION >= v"1.11.0-DEV.1157"
    Base.to_indices(A::AnyGPUArray, I::Tuple{AbstractArray{Bool}}) =
        (checkbounds(A, I[1]); (Base.to_index(A, I[1]),))
else
    # (also reached for the last of several indices, whose `inds` are then not all of `A`'s)
    _check_mask(A, inds, mask) = length(inds) == ndims(A) ? checkbounds(A, mask) : nothing
    Base.to_indices(A::AnyGPUArray, inds,
                    I::Tuple{Union{Array{Bool,N}, BitArray{N}}}) where {N} =
        (_check_mask(A, inds, I[1]); (Base.to_index(A, I[1]),))
    Base.to_indices(A::AnyGPUArray, inds, I::Tuple{AbstractArray{Bool}}) =
        (_check_mask(A, inds, I[1]); (Base.to_index(A, I[1]),))
end
# ... except that a mask of the array's shape selects the values themselves, in one pass
function Base.getindex(A::AbstractGPUArray, mask::AnyGPUArray{Bool})
    checkbounds(A, mask)
    axes(mask) == axes(A) || return invoke(getindex, Tuple{AbstractGPUArray, Vararg{Any}}, A, mask)
    return AK.findall(mask; items=A)
end

# `findmin` and `findmax` reduce `(f(x), i)` pairs, with `i` the position of `x`: without an
# `init`, as partial results start from their first element. Indices are Base's, `keys(A)`; so are
# the errors of empty inputs, from a host stand-in.
struct _FindPair{F}
    f::F
end
(p::_FindPair)(x, i) = (p.f(x), i)


function findminmax(binop, f, A::AnyGPUArray; dims)
    function reduction(t1, t2)
        (x, i), (y, j) = t1, t2

        binop(x, y) && return t2
        isequal(x, y) && return (x, min(i, j))
        return t1
    end

    if isempty(A)
        h = (binop === Base.isless ? findmax : findmin)(f, Array{eltype(A)}(undef, size(A)); dims)
        return dims === Colon() ? h : (copyto!(similar(A, eltype(h[1]), size(h[1])), h[1]),
                                       copyto!(similar(A, eltype(h[2]), size(h[2])), h[2]))
    end

    # (along valid `dims`, a 0-dimensional array reduces nothing; Julia 1.10's `reduced_indices`
    # cannot take it)
    rdims = ndims(A) == 0 && !(dims isa Colon) && all(d -> d isa Integer && d >= 1, dims) ? () : dims
    res = mapreduce(_FindPair(f), reduction, A, LinearIndices(A); dims=rdims)
    I = keys(A)
    if dims === Colon()
        return (res[1], I[res[2]])
    else
        # (`map!`, as `map` of a 0-dimensional array would give a scalar)
        vals = map!(first, similar(res, fieldtype(eltype(res), 1)), res)
        inds = map!(x -> I[x[2]], similar(res, eltype(I)), res)
        return (vals, inds)
    end
end

Base.findmax(a::AnyGPUArray; dims=:) = findminmax(Base.isless, identity, a; dims)
Base.findmin(a::AnyGPUArray; dims=:) = findminmax(Base.isgreater, identity, a; dims)
Base.findmax(f::Function, a::AnyGPUArray; dims=:) = findminmax(Base.isless, f, a; dims)
Base.findmin(f::Function, a::AnyGPUArray; dims=:) = findminmax(Base.isgreater, f, a; dims)

# the element that minimizes or maximizes `f` (Base iterates)
Base.argmax(f::Function, a::AnyGPUArray) = @allowscalar a[findmax(f, a)[2]]
Base.argmin(f::Function, a::AnyGPUArray) = @allowscalar a[findmin(f, a)[2]]
