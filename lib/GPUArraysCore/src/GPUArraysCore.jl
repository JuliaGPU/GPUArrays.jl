module GPUArraysCore

using Adapt


## essential types

export AbstractGPUArray, AbstractGPUVector, AbstractGPUMatrix, AbstractGPUVecOrMat,
       WrappedGPUArray, AnyGPUArray, AbstractGPUArrayStyle,
       AnyGPUArray, AnyGPUVector, AnyGPUMatrix

"""
    AbstractGPUArray{T, N} <: DenseArray{T, N}

Supertype for `N`-dimensional GPU arrays (or array-like types) with elements of type `T`.
Instances of this type are expected to live on the host, see [`AbstractDeviceArray`](@ref)
for device-side objects.
"""
abstract type AbstractGPUArray{T, N} <: DenseArray{T, N} end

const AbstractGPUVector{T} = AbstractGPUArray{T, 1}
const AbstractGPUMatrix{T} = AbstractGPUArray{T, 2}
const AbstractGPUVecOrMat{T} = Union{AbstractGPUArray{T, 1}, AbstractGPUArray{T, 2}}

# convenience aliases for working with wrapped arrays
const WrappedGPUArray{T,N} = WrappedArray{T,N,AbstractGPUArray,AbstractGPUArray{T,N}}
const AnyGPUArray{T,N} = Union{AbstractGPUArray{T,N}, WrappedGPUArray{T,N}}
const AnyGPUVector{T} = AnyGPUArray{T, 1}
const AnyGPUMatrix{T} = AnyGPUArray{T, 2}


## broadcasting

"""
Abstract supertype for GPU array styles. The `N` parameter is the dimensionality.

Downstream implementations should provide a concrete array style type that inherits from
this supertype.
"""
abstract type AbstractGPUArrayStyle{N} <: Base.Broadcast.AbstractArrayStyle{N} end


## scalar iteration

export allowscalar, @allowscalar, assertscalar

@enum ScalarIndexing ScalarAllowed ScalarWarn ScalarWarned ScalarDisallowed

# if the user explicitly calls allowscalar, use that setting for all new tasks
# XXX: use context variables to inherit the parent task's setting, once available.
const default_scalar_indexing = Ref{Union{Nothing,ScalarIndexing}}(nothing)

"""
    assertscalar(op::String)

Assert that a certain operation `op` performs scalar indexing. If this is not allowed, an
error will be thrown ([`allowscalar`](@ref)).
"""
function assertscalar(op = "operation")
    # try to detect the REPL
    @static if VERSION >= v"1.10.0-DEV.444" || v"1.9-beta4" <= VERSION < v"1.10-"
        if isdefined(Base, :active_repl) && current_task() == Base.active_repl.frontend_task
            # we always allow scalar iteration on the REPL's frontend task,
            # where we often trigger scalar indexing by displaying GPU objects.
            return false
        end
        default_behavior = ScalarDisallowed
    else
        # we can't detect the REPL, but it will only be used in interactive sessions,
        # so default to allowing scalar indexing there (but warn).
        default_behavior = isinteractive() ? ScalarWarn : ScalarDisallowed
    end

    val = get!(task_local_storage(), :ScalarIndexing) do
        something(default_scalar_indexing[], default_behavior)
    end
    desc = """Invocation of $op resulted in scalar indexing of a GPU array.
              This is typically caused by calling an iterating implementation of a method.
              Such implementations *do not* execute on the GPU, but very slowly on the CPU,
              and therefore should be avoided.

              If you want to allow scalar iteration, use `allowscalar` or `@allowscalar`
              to enable scalar iteration globally or for the operations in question."""
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

# Like a try-finally block, except without introducing the try scope
# NOTE: This is deprecated and should not be used from user logic. A proper solution to
# this problem will be introduced in https://github.com/JuliaLang/julia/pull/39217
macro __tryfinally(ex, fin)
    Expr(:tryfinally,
       :($(esc(ex))),
       :($(esc(fin)))
       )
end

"""
    allowscalar([true])
    allowscalar([true]) do
        ...
    end

Use this function to allow or disallow scalar indexing, either globall or for the
duration of the do block.

See also: [`@allowscalar`](@ref).
"""
allowscalar

function allowscalar(f::Base.Callable)
    task_local_storage(f, :ScalarIndexing, ScalarAllowed)
end

function allowscalar(allow::Bool=true)
    if allow
        @warn """It's not recommended to use allowscalar([true]) to allow scalar indexing.
                 Instead, use `allowscalar() do end` or `@allowscalar` to denote exactly which operations can use scalar operations.""" maxlog=1
    end
    setting = allow ? ScalarAllowed : ScalarDisallowed
    task_local_storage(:ScalarIndexing, setting)
    default_scalar_indexing[] = setting
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
        local tls_value = get(task_local_storage(), :ScalarIndexing, nothing)
        task_local_storage(:ScalarIndexing, ScalarAllowed)
        @__tryfinally($(esc(ex)),
                      isnothing(tls_value) ? delete!(task_local_storage(), :ScalarIndexing)
                                           : task_local_storage(:ScalarIndexing, tls_value))
    end
end


## other

"""
    backend(T::Type)
    backend(x)

Gets the GPUArrays back-end responsible for managing arrays of type `T`.
"""
backend(::Type) = error("This object is not a GPU array") # COV_EXCL_LINE
backend(x) = backend(typeof(x))

backend(::Type{WA}) where WA<:WrappedArray = backend(parent(WA)) # WrappedArray from Adapt for Base wrappers.

end # module GPUArraysCore
