# host-level indexing

export allowscalar, @allowscalar, @disallowscalar, assertscalar


# mechanism to disallow scalar operations

@enum ScalarIndexing ScalarAllowed ScalarWarned ScalarDisallowed

const scalar_allowed = Ref(ScalarWarned)
const scalar_warned = Ref(false)

"""
    allowscalar(allow=true, warn=true)

Configure whether scalar indexing is allowed depending on the value of `allow`.

If allowed, `warn` can be set to throw a single warning instead. Calling this function will
reset the state of the warning, and throw a new warning on subsequent scalar iteration.
"""
function allowscalar(allow::Bool=true, warn::Bool=true)
    scalar_warned[] = false
    scalar_allowed[] = if allow && !warn
        ScalarAllowed
    elseif allow
        ScalarWarned
    else
        ScalarDisallowed
    end
    return
end

"""
    assertscalar(op::String)

Assert that a certain operation `op` performs scalar indexing. If this is not allowed, an
error will be thrown ([`allowscalar`](@ref)).
"""
function assertscalar(op = "operation")
    if scalar_allowed[] == ScalarDisallowed
        error("$op is disallowed")
    elseif scalar_allowed[] == ScalarWarned && !scalar_warned[]
        @warn "Performing scalar operations on GPU arrays: This is very slow, consider disallowing these operations with `allowscalar(false)`"
        scalar_warned[] = true
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
        local prev = scalar_allowed[]
        scalar_allowed[] = ScalarAllowed
        local ret = $(esc(ex))
        scalar_allowed[] = prev
        ret
    end
end

@doc (@doc @allowscalar) ->
macro disallowscalar(ex)
    quote
        local prev = scalar_allowed[]
        scalar_allowed[] = ScalarDisallowed
        local ret = $(esc(ex))
        scalar_allowed[] = prev
        ret
    end
end

@doc (@doc @allowscalar) ->
function allowscalar(f::Base.Callable, allow::Bool=true, warn::Bool=false)
    prev = scalar_allowed[]
    allowscalar(allow, warn)
    ret = f()
    scalar_allowed[] = prev
    ret
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

function _getindex(src::AbstractGPUArray, Is::Vararg{<:Any,N}) where {N}
    shape = Base.index_shape(Is...)
    dest = similar(src, shape)
    any(isempty, Is) && return dest # indexing with empty array
    idims = map(length, Is)

    function kernel(ctx::AbstractKernelContext, dest::AbstractArray, src::AbstractArray, idims, Is)
        i = @linearidx dest
        @inbounds begin
            is = CartesianIndices(idims)[i]
            idx = ntuple(dim -> @inbounds(Is[dim][is[dim]]), N)
            dest[i] = getindex(src, idx...)
        end
        return
    end

    AT = typeof(src).name.wrapper
    # NOTE: we are pretty liberal here supporting non-GPU indices...
    gpu_call(kernel, dest, src, idims, adapt(AT, Is); name="getindex!")
    return dest
end

Base.setindex!(A::AbstractGPUArray, v, I...) = _setindex!(A, v, to_indices(A, I)...)

function _setindex!(dest::AbstractGPUArray, src, Is::Vararg{<:Any,N}) where {N}
    isempty(Is) && return dest
    idims = length.(Is)
    len = prod(idims)
    len==0 && return dest

    function kernel(ctx::AbstractKernelContext, dest, src, idims, len, Is)
        i = linear_index(ctx)
        i > len && return
        @inbounds begin
            is = CartesianIndices(idims)[i]
            idx = ntuple(dim -> @inbounds(Is[dim][is[dim]]), N)
            setindex!(dest, src[i], idx...)
        end
        return
    end

    AT = typeof(dest).name.wrapper
    # NOTE: we are pretty liberal here supporting non-GPU sources and indices...
    gpu_call(kernel, dest, adapt(AT, src), idims, len, adapt(AT, Is);
             total_threads=len, name="setindex!")
    return dest
end
