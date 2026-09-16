# sorting

# GPUArrays' sorting methods forward to backend-defined `GPUArrays.sort!` and
# `GPUArrays.sortperm!` primitives, in the same fashion as `mapreducedim!` for reductions: a
# backend implements the primitive for its array type, and the generic `Base.sort` / `Base.sortperm`
# methods below build on it. `dims === :` sorts the whole array as one flat vector; an integer
# `dims` sorts each 1-D slice along that dimension.

sort!(A::AnyGPUArray; dims=:, lt=isless, by=identity, rev=false,
      order::Base.Order.Ordering=Base.Order.Forward) = error("Not implemented") # COV_EXCL_LINE

sortperm!(ix::AnyGPUArray, A::AnyGPUArray; dims=:, lt=isless, by=identity, rev=false,
          order::Base.Order.Ordering=Base.Order.Forward,
          initialized=false) = error("Not implemented") # COV_EXCL_LINE


function Base.sort!(A::AnyGPUArray; dims=:, lt=isless, by=identity,
                    rev::Union{Bool,Nothing}=nothing, order::Base.Order.Ordering=Base.Order.Forward)
    GPUArrays.sort!(A; dims, lt, by, rev=something(rev, false), order)
end

Base.sort(A::AnyGPUArray; kwargs...) = Base.sort!(copy(A); kwargs...)

function Base.sortperm!(ix::AnyGPUArray, A::AnyGPUArray; dims=:, lt=isless, by=identity,
                        rev::Union{Bool,Nothing}=nothing,
                        order::Base.Order.Ordering=Base.Order.Forward, initialized::Bool=false)
    axes(ix) == axes(A) ||
        throw(ArgumentError("index array must have the same axes as the array being sorted"))
    GPUArrays.sortperm!(ix, A; dims, lt, by, rev=something(rev, false), order, initialized)
end

function Base.sortperm(A::AnyGPUArray; kwargs...)
    ix = similar(A, Int)
    Base.sortperm!(ix, A; kwargs..., initialized=false)
end
