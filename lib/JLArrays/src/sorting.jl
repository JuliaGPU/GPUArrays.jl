# sorting and reversing

# GPUArrays implements these with AcceleratedKernels, whose kernels do not run on JLArrays'
# back-end. The reference implementation applies Base to the arrays' storage instead, with the
# same method shapes, and accepts the same `alg` values as GPU arrays (`GPUArrays._akalg`).

_host(A) = adapt(ArrayNoCopy(), A)

function Base.sort!(v::AnyJLVector; alg=nothing, lt=isless, by=identity, rev=nothing,
                    order::Base.Order.Ordering=Base.Order.Forward, scratch=nothing)
    GPUArrays._akalg(alg)
    sort!(_host(v); lt, by, rev, order)
    return v
end
function Base.sort!(A::AnyJLArray; dims::Integer, alg=nothing, scratch=nothing, kwargs...)
    GPUArrays._akalg(alg)
    sort!(_host(A); dims, kwargs...)
    return A
end

function Base.sortperm!(ix::AnyJLArray{<:Integer}, v::AnyJLVector; alg=nothing,
                        scratch=nothing, initialized::Bool=false, dims=nothing, kwargs...)
    dims === nothing || throw(ArgumentError("sortperm! of a vector does not accept `dims`"))
    axes(ix) == axes(v) ||
        throw(ArgumentError("index array must have the same axes as the source array"))
    GPUArrays._akalg(alg)
    sortperm!(_host(ix), _host(v); kwargs...)
    return ix
end
function Base.sortperm!(ix::AnyJLArray{<:Integer}, A::AnyJLArray; dims::Integer, alg=nothing,
                        scratch=nothing, initialized::Bool=false, kwargs...)
    axes(ix) == axes(A) ||
        throw(ArgumentError("index array must have the same axes as the source array"))
    GPUArrays._akalg(alg)
    sortperm!(_host(ix), _host(A); dims, kwargs...)
    return ix
end

function Base.reverse!(A::AnyJLArray; dims=:)
    reverse!(_host(A); dims)
    return A
end
Base.reverse(A::AnyJLArray; dims=:) = reverse!(copy(A); dims)
Base.reverse!(v::AnyJLVector; dims=:) = invoke(reverse!, Tuple{AbstractVector}, v; dims)
Base.reverse(v::AnyJLVector; dims=:) = invoke(reverse, Tuple{AbstractVector}, v; dims)
function Base.reverse!(v::AnyJLVector, start::Integer, stop::Integer=lastindex(v))
    reverse!(_host(v), start, stop)
    return v
end
