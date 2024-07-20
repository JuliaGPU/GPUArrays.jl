# Custom GPU-compatible `Number` interface.
struct AsyncNumber{T <: AbstractGPUArray} <: AbstractNumbers.AbstractNumber{T}
    val::T

    function AsyncNumber(val::T) where T <: AbstractGPUArray
        length(val) != 1 && error(
            "`AsyncNumber` accepts only 1-element GPU arrays, " *
            "instead `$(length(val))`-element array was given.")
        new{T}(val)
    end
end

AbstractNumbers.number(g::AsyncNumber) = @allowscalar g.val[]
maybe_number(g::AsyncNumber) = AbstractNumbers.number(g)
maybe_number(g) = g

number_type(::AsyncNumber{T}) where T = eltype(T)

# When operations involve other `::Number` types,
# do not convert back to `AsyncNumber`.
AbstractNumbers.like(::Type{<: AsyncNumber}, x) = x

# When broadcasting, just pass the array itself.
Base.broadcastable(g::AsyncNumber) = g.val

# Overload to avoid copies.
Base.one(g::AsyncNumber) = one(number_type(g))
Base.one(::Type{AsyncNumber{T}}) where T = one(eltype(T))
Base.zero(g::AsyncNumber) = zero(number_type(g))
Base.zero(::Type{AsyncNumber{T}}) where T = zero(eltype(T))
Base.identity(g::AsyncNumber) = g

Base.getindex(g::AsyncNumber) = AbstractNumbers.number(g)

Base.isequal(g::AsyncNumber, v::Number) = isequal(g[], v)
Base.isequal(v::Number, g::AsyncNumber) = isequal(v, g[])

Base.nextpow(a, x::AsyncNumber) = nextpow(a, x[])
Base.nextpow(a::AsyncNumber, x) = nextpow(a[], x)
Base.nextpow(a::AsyncNumber, x::AsyncNumber) = nextpow(a[], x[])

Base.convert(::Type{Number}, g::AsyncNumber) = g[]
