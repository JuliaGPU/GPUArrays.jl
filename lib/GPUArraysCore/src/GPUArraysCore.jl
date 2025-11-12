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

"""
    AbstractGPUVector{T}

Shortcut for `AbstractGPUArray{T, 1}`.
"""
const AbstractGPUVector{T} = AbstractGPUArray{T, 1}

"""
    AbstractGPUMatrixT}

Shortcut for `AbstractGPUArray{T, 2}`.
"""
const AbstractGPUMatrix{T} = AbstractGPUArray{T, 2}

"""
    AbstractGPUVecOrMat{T}

Shortcut for `Union{AbstractGPUArray{T, 1}, AbstractGPUArray{T, 2}}`.
"""
const AbstractGPUVecOrMat{T} = Union{AbstractGPUArray{T, 1}, AbstractGPUArray{T, 2}}

# convenience aliases for working with wrapped arrays

"""
    WrappedGPUArray{T, N}

Convenience alias for working with wrapped arrays from [Adapt.jl](https://github.com/JuliaGPU/Adapt.jl).
"""
const WrappedGPUArray{T,N} = WrappedArray{T,N,AbstractGPUArray,AbstractGPUArray{T,N}}

"""
    AnyGPUArray{T, N}

Shortcut for `Union{AbstractGPUArray{T,N}, WrappedGPUArray{T,N}}`.
"""
const AnyGPUArray{T,N} = Union{AbstractGPUArray{T,N}, WrappedGPUArray{T,N}}

"""
    AnyGPUVector{T}

Shortcut for `AnyGPUArray{T, 1}`.
"""
const AnyGPUVector{T} = AnyGPUArray{T, 1}

"""
    AnyGPUMatrix{T}

Shortcut for `AnyGPUArray{T, 2}`.
"""
const AnyGPUMatrix{T} = AnyGPUArray{T, 2}


## broadcasting

"""
    AbstractGPUArrayStyle{N} <: Base.Broadcast.AbstractArrayStyle{N}

Abstract supertype for GPU array broadcasting styles. The `N` parameter is the dimensionality.

Downstream implementations should provide a concrete array style type that inherits from
this supertype.
"""
abstract type AbstractGPUArrayStyle{N} <: Base.Broadcast.AbstractArrayStyle{N} end


## scalar iteration

export allowscalar, @allowscalar, assertscalar

@enum ScalarIndexing ScalarAllowed ScalarWarn ScalarWarned ScalarDisallowed

# if the user explicitly calls allowscalar, use that setting for all new tasks
# XXX: use context variables to inherit the parent task's setting, once available.
const requested_scalar_indexing = Ref{Union{Nothing,ScalarIndexing}}(nothing)

const _repl_frontend_task = Ref{Union{Nothing,Missing,Task}}()
function repl_frontend_task()
    if !isassigned(_repl_frontend_task)
        _repl_frontend_task[] = get_repl_frontend_task()
    end
    _repl_frontend_task[]
end
@noinline function get_repl_frontend_task()
    if isdefined(Base, :active_repl)
        Base.active_repl.frontend_task
    else
        missing
    end
end

@noinline function default_scalar_indexing()
    if isinteractive()
        # try to detect the REPL
        repl_task = repl_frontend_task()
        if repl_task isa Task
            if repl_task === current_task()
                # we always allow scalar iteration on the REPL's frontend task,
                # where we often trigger scalar indexing by displaying GPU objects.
                ScalarAllowed
            else
                ScalarDisallowed
            end
        else
            # we couldn't detect a REPL in this interactive session, so default to a warning
            ScalarWarn
        end
    else
        # non-interactively, we always disallow scalar iteration
        ScalarDisallowed
    end
end

"""
    assertscalar(op::String)

Assert that a certain operation `op` performs scalar indexing. If this is not allowed, an
error will be thrown ([`allowscalar`](@ref)).
"""
function assertscalar(op::String)
    behavior = get(task_local_storage(), :ScalarIndexing, nothing)
    if behavior === nothing
        behavior = requested_scalar_indexing[]
        if behavior === nothing
            behavior = default_scalar_indexing()
        end
        task_local_storage(:ScalarIndexing, behavior)
    end

    behavior = behavior::ScalarIndexing
    if behavior === ScalarAllowed
        # fast path
        return
    end

    _assertscalar(op, behavior)
end

@noinline function _assertscalar(op, behavior)
    desc = """Invocation of '$op' resulted in scalar indexing of a GPU array.
              This is typically caused by calling an iterating implementation of a method.
              Such implementations *do not* execute on the GPU, but very slowly on the CPU,
              and therefore should be avoided.

              If you want to allow scalar iteration, use `allowscalar` or `@allowscalar`
              to enable scalar iteration globally or for the operations in question."""
    if behavior == ScalarDisallowed
        errorscalar(op)
    elseif behavior == ScalarWarn
        warnscalar(op)
        task_local_storage(:ScalarIndexing, ScalarWarned)
    end

    return
end

function scalardesc(op)
    desc = """Invocation of $op resulted in scalar indexing of a GPU array.
              This is typically caused by calling an iterating implementation of a method.
              Such implementations *do not* execute on the GPU, but very slowly on the CPU,
              and therefore should be avoided.

              If you want to allow scalar iteration, use `allowscalar` or `@allowscalar`
              to enable scalar iteration globally or for the operations in question."""
end

@noinline function warnscalar(op)
    desc = scalardesc(op)
    @warn("""Performing scalar indexing on task $(current_task()).
             $desc""")
end

@noinline function errorscalar(op)
    desc = scalardesc(op)
    error("""Scalar indexing is disallowed.
             $desc""")
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
    requested_scalar_indexing[] = setting
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

end # module GPUArraysCore
