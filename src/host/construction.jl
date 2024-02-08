# convenience and indirect construction

# conversions from CPU arrays rely on constructors
Base.convert(::Type{T}, a::AbstractArray) where {T<:AbstractGPUArray} = a isa T ? a : T(a)
# TODO: can we implement constructors to and from ::AbstractArray here? by calling the undef
#       constructor and doing a `copyto!`. this is tricky, due to ambiguities, and no easy
#       way to go from <:AbstractGPUArray{T,N} to e.g. CuArray{S,N}


## convenience constructors

function Base.fill!(A::AnyGPUArray{T}, x) where T
    length(A) == 0 && return A
    @kernel fill!(a, val)
        idx = @index(Linear, Global)
        @inbounds a[idx] = val
    end
    kernel = fill!(backend(A))
    kernel(A, x)
    A
end


## identity matrices

@kernel function identity_kernel(res::AbstractArray{T}, stride, val) where T
    i = @index(Global, Linear)
    ilin = (stride * (i - 1)) + i
    ilin > length(res) && return
    @inbounds res[ilin] = val
end

function (T::Type{<: AnyGPUArray{U}})(s::UniformScaling, dims::Dims{2}) where {U}
    res = similar(T, dims)
    fill!(res, zero(U))
    kernel = identity_kernel(backend(res))
    kernel(res, size(res, 1), s.λ; ndrange=minimum(dims))
    res
end

(T::Type{<: AnyGPUArray})(s::UniformScaling{U}, dims::Dims{2}) where U = T{U}(s, dims)

(T::Type{<: AnyGPUArray})(s::UniformScaling, m::Integer, n::Integer) = T(s, Dims((m, n)))

function Base.copyto!(A::AbstractGPUMatrix{T}, s::UniformScaling) where T
    fill!(A, zero(T))
    kernel = identity_kernel(backend(A))
    kernel(A, size(A, 1), s.λ; ndrange=minimum(size(A)))
    A
end

function _one(unit::T, x::AbstractGPUMatrix) where {T}
    m,n = size(x)
    m==n || throw(DimensionMismatch("multiplicative identity defined only for square matrices"))
    I = similar(x, T)
    fill!(I, zero(T))
    kernel = identity_kernel(backend(I))
    kernel(I, m, unit; ndrange=m)
    I
end

Base.one(x::AbstractGPUMatrix{T}) where {T} = _one(one(T), x)
Base.oneunit(x::AbstractGPUMatrix{T}) where {T} = _one(oneunit(T), x)
