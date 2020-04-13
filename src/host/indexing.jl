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


# basic indexing

Base.IndexStyle(::Type{<:AbstractGPUArray}) = Base.IndexLinear()

function Base.getindex(xs::AbstractGPUArray{T}, i::Integer) where T
    assertscalar("scalar getindex")
    x = Array{T}(undef, 1)
    copyto!(x, 1, xs, i, 1)
    return x[1]
end

function Base.setindex!(xs::AbstractGPUArray{T}, v::T, i::Integer) where T
    assertscalar("scalar setindex!")
    x = T[v]
    copyto!(xs, i, x, 1, 1)
    return xs
end

Base.setindex!(xs::AbstractGPUArray, v, i::Integer) = xs[i] = convert(eltype(xs), v)


# Vector indexing

to_index(a, x) = x
to_index(a::A, x::Array{ET}) where {A, ET} = copyto!(similar(a, ET, size(x)...), x)
to_index(a, x::UnitRange{<: Integer}) = convert(UnitRange{Int}, x)
to_index(a, x::Base.LogicalIndex) = error("Logical indexing not implemented")

@generated function index_kernel(ctx::AbstractKernelContext, dest::AbstractArray, src::AbstractArray, idims, Is)
    N = length(Is.parameters)
    quote
        i = @linearidx dest
        is = CartesianIndices(idims)[i]
        @nexprs $N i -> @inbounds I_i = Is[i][is[i]]
        @inbounds dest[i] = @ncall $N getindex src i -> I_i
        return
    end
end

function Base._unsafe_getindex!(dest::AbstractGPUArray, src::AbstractGPUArray, Is::Union{Real, AbstractArray}...)
    if any(isempty, Is) # indexing with empty array
        return dest
    end
    idims = map(length, Is)
    gpu_call(index_kernel, dest, src, idims, map(x-> to_index(dest, x), Is))
    return dest
end

# FIXME: simple broadcast getindex like function... reuse from Base
@inline bgetindex(x::AbstractArray, i) = x[i]
@inline bgetindex(x, i) = x

@generated function setindex_kernel!(ctx::AbstractKernelContext, dest::AbstractArray, src, idims, Is, len)
    N = length(Is.parameters)
    idx = ntuple(i-> :(Is[$i][is[$i]]), N)
    quote
        i = linear_index(ctx)
        i > len && return
        is = CartesianIndices(idims)[i]
        @inbounds setindex!(dest, bgetindex(src, i), $(idx...))
        return
    end
end

function Base._unsafe_setindex!(::IndexStyle, dest::T, src, Is::Union{Real, AbstractArray}...) where T <: AbstractGPUArray
    if length(Is) == 1 && isa(first(Is), Array) && isempty(first(Is)) # indexing with empty array
        return dest
    end
    idims = length.(Is)
    len = prod(idims)
    src_gpu = adapt(T, src)
    gpu_call(setindex_kernel!, dest, src_gpu, idims, map(x-> to_index(dest, x), Is), len;
             total_threads=len)
    return dest
end
