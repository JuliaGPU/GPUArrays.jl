# sorting

# JLArrays are backed by a regular Array, so the sorting primitives defer to Base on that storage.
# This keeps the reference backend independent of AcceleratedKernels' GPU kernels. Only the mutating
# primitives are overridden; `sort`, `sortperm` and `partialsort` route through them. `alg` (an
# AcceleratedKernels algorithm) is ignored here, since the reference only needs to match Base.

function Base.sort!(v::AnyJLArray; dims=:, alg=nothing, lt=isless, by=identity,
                    rev::Bool=false, order::Base.Order.Ordering=Base.Order.Forward)
    if dims === Colon()
        sort!(vec(typed_data(v)); lt, by, rev, order)
    else
        sort!(typed_data(v); dims, lt, by, rev, order)
    end
    return v
end

function Base.sortperm!(ix::AnyJLArray, v::AnyJLArray; dims=:, alg=nothing, lt=isless,
                        by=identity, rev::Bool=false,
                        order::Base.Order.Ordering=Base.Order.Forward, initialized::Bool=false)
    axes(ix) == axes(v) ||
        throw(ArgumentError("index array must have the same axes as the array being sorted"))
    if dims === Colon()
        sortperm!(vec(typed_data(ix)), vec(typed_data(v)); lt, by, rev, order, initialized)
    else
        sortperm!(typed_data(ix), typed_data(v); dims, lt, by, rev, order, initialized)
    end
    return ix
end
