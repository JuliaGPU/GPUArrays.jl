# Custom GPU-compatible `Number` interface.
struct GPUNumber{T <: AbstractGPUArray} <: AN.AbstractNumber{T}
    val::T

    function GPUNumber(val::T) where T <: AbstractGPUArray
        length(val) != 1 && error(
            "`GPUNumber` accepts only 1-element GPU arrays, " *
            "instead `$(length(val))`-element array was given.")
        new{T}(val)
    end
end

AN.number(g::GPUNumber) = @allowscalar g.val[]

maybe_number(g::GPUNumber) = AN.number(g)
maybe_number(g) = g

number_type(::GPUNumber{T}) where T = eltype(T)

# When operations involve other `::Number` types,
# do not convert back to `GPUNumber`.
AN.like(::Type{<: GPUNumber}, x) = x

# When broadcasting, just pass the array itself.
Base.broadcastable(g::GPUNumber) = g.val

# Overload to avoid copies.
Base.one(g::GPUNumber) = one(number_type(g))
Base.one(::Type{GPUNumber{T}}) where T = one(eltype(T))
Base.zero(g::GPUNumber) = zero(number_type(g))
Base.zero(::Type{GPUNumber{T}}) where T = zero(eltype(T))
Base.identity(g::GPUNumber) = g
