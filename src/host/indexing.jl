# host-level indexing

export allowscalar, @allowscalar, @disallowscalar, assertscalar


# mechanism to disallow scalar operations

@enum ScalarIndexing ScalarAllowed ScalarWarn ScalarWarned ScalarDisallowed

"""
    allowscalar(allow=true, warn=true)
    allowscalar(allow=true, warn=true) do end

Configure whether scalar indexing is allowed depending on the value of `allow`.

If allowed, `warn` can be set to throw a single warning instead. Calling this function will
reset the state of the warning, and throw a new warning on subsequent scalar iteration.

For temporary changes, use the do-block version, or [`@allowscalar`](@ref).
"""
function allowscalar(allow::Bool=true, warn::Bool=true)
    val = if allow && !warn
        ScalarAllowed
    elseif allow
        ScalarWarn
    else
        ScalarDisallowed
    end

    task_local_storage(:ScalarIndexing, val)
    return
end

@doc (@doc allowscalar) ->
function allowscalar(f::Base.Callable, allow::Bool=true, warn::Bool=false)
    val = if allow && !warn
        ScalarAllowed
    elseif allow
        ScalarWarn
    else
        ScalarDisallowed
    end

    task_local_storage(f, :ScalarIndexing, val)
end

"""
    assertscalar(op::String)

Assert that a certain operation `op` performs scalar indexing. If this is not allowed, an
error will be thrown ([`allowscalar`](@ref)).
"""
function assertscalar(op = "operation")
    val = get(task_local_storage(), :ScalarIndexing, ScalarWarn)
    if val == ScalarDisallowed
        error("$op is disallowed")
    elseif val == ScalarWarn
        @warn "Performing scalar operations on GPU arrays: This is very slow, consider disallowing these operations with `allowscalar(false)`"
        task_local_storage(:ScalarIndexing, ScalarWarned)
    end
    return
end

"""
    @allowscalar ex...
    @disallowscalar ex...
    allowscalar(::Function, ...)

Temporarily allow or disallow scalar iteration.

Note that this functionality is intended for functionality that is known and allowed to use
scalar iteration (or not), i.e., there is no option to throw a warning. Only use this on
fine-grained expressions.
"""
macro allowscalar(ex)
    quote
        task_local_storage(:ScalarIndexing, ScalarAllowed) do
            $(esc(ex))
        end
    end
end

@doc (@doc @allowscalar) ->
macro disallowscalar(ex)
    quote
        task_local_storage(:ScalarIndexing, ScalarDisallowed) do
            $(esc(ex))
        end
    end
end


# basic indexing with integers

Base.IndexStyle(::Type{<:AbstractGPUArray}) = Base.IndexLinear()

function Base.getindex(xs::AbstractGPUArray{T}, I::Integer...) where T
    assertscalar("scalar getindex")
    i = Base._to_linear_index(xs, I...)
    x = Array{T}(undef, 1)
    copyto!(x, 1, xs, i, 1)
    return x[1]
end

function Base.setindex!(xs::AbstractGPUArray{T}, v::T, I::Integer...) where T
    assertscalar("scalar setindex!")
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
                                    Is::Vararg{<:Any,N}) where {N}
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
             total_threads=len)
    return dest
end

@generated function setindex_kernel(ctx::AbstractKernelContext, dest, src, idims, len,
                                    Is::Vararg{<:Any,N}) where {N}
    quote
        i = linear_index(ctx)
        i > len && return
        is = @inbounds CartesianIndices(idims)[i]
        @nexprs $N i -> I_i = @inbounds(Is[i][is[i]])
        @ncall $N setindex! dest src[i] i -> I_i
        return
    end
end
