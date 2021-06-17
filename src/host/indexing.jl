# host-level indexing

export allowscalar, @allowscalar, assertscalar


# mechanism to disallow scalar operations

@enum ScalarIndexing ScalarAllowed ScalarWarn ScalarWarned ScalarDisallowed

# if the user explicitly calls allowscalar, use that setting for all new tasks
# XXX: use context variables to inherit the parent task's setting, once available.
const default_scalar_indexing = Ref{Union{Nothing,ScalarIndexing}}(nothing)

"""
    allowscalar() do
        # code that can use scalar indexing
    end

Denote which operations can use scalar indexing.

See also: [`@allowscalar`](@ref).
"""
function allowscalar(f::Base.Callable)
    task_local_storage(f, :ScalarIndexing, ScalarAllowed)
end

function allowscalar(allow::Bool=true)
    if allow
        Base.depwarn("allowscalar([true]) is deprecated, use `allowscalar() do end` or `@allowscalar` to denote exactly which operations can use scalar operations.", :allowscalar)
    end
    setting = allow ? ScalarAllowed : ScalarDisallowed
    task_local_storage(:ScalarIndexing, setting)
    default_scalar_indexing[] = setting
    return
end

"""
    assertscalar(op::String)

Assert that a certain operation `op` performs scalar indexing. If this is not allowed, an
error will be thrown ([`allowscalar`](@ref)).
"""
function assertscalar(op = "operation")
    val = get!(task_local_storage(), :ScalarIndexing) do
        something(default_scalar_indexing[], isinteractive() ? ScalarWarn : ScalarDisallowed)
    end
    desc = """Invocation of $op resulted in scalar indexing of a GPU array.
              This is typically caused by calling an iterating implementation of a method.
              Such implementations *do not* execute on the GPU, but very slowly on the CPU,
              and therefore are only permitted from the REPL for prototyping purposes.
              If you did intend to index this array, annotate the caller with @allowscalar."""
    if val == ScalarDisallowed
        error("""Scalar indexing is disallowed.
                 $desc""")
    elseif val == ScalarWarn
        @warn("""Performing scalar indexing on task $(current_task()).
                 $desc""")
        task_local_storage(:ScalarIndexing, ScalarWarned)
    end
    return
end

"""
    @allowscalar() begin
        # code that can use scalar indexing
    end

Denote which operations can use scalar indexing.

See also: [`allowscalar`](@ref).
"""
macro allowscalar(ex)
    quote
        task_local_storage(:ScalarIndexing, true) do
            $(esc(ex))
        end
    end
end


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
             total_threads=len)
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
