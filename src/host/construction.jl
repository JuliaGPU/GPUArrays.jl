# convenience and indirect construction

# conversions from CPU arrays rely on constructors
Base.convert(::Type{T}, a::AbstractArray) where {T<:AbstractGPUArray} = a isa T ? a : T(a)
# TODO: can we implement constructors to and from ::AbstractArray here? by calling the undef
#       constructor and doing a `copyto!`. this is tricky, due to ambiguities, and no easy
#       way to go from <:AbstractGPUArray{T,N} to e.g. CuArray{S,N}


## convenience constructors

function Base.fill!(A::AnyGPUArray{T}, x) where T
    length(A) == 0 && return A
    gpu_call(A, convert(T, x)) do ctx, a, val
        idx = @linearidx(a)
        @inbounds a[idx] = val
        return
    end
    A
end


## identity matrices

function identity_kernel(ctx::AbstractKernelContext, res::AbstractArray{T}, stride, val) where T
    i = linear_index(ctx)
    i > stride && return
    ilin = (stride * (i - 1)) + i
    @inbounds res[ilin] = val
    return
end

function (T::Type{<: AnyGPUArray{U}})(s::UniformScaling, dims::Dims{2}) where {U}
    res = similar(T, dims)
    fill!(res, zero(U))
    gpu_call(identity_kernel, res, size(res, 1), s.λ; elements=minimum(dims))
    res
end

(T::Type{<: AnyGPUArray})(s::UniformScaling{U}, dims::Dims{2}) where U = T{U}(s, dims)

(T::Type{<: AnyGPUArray})(s::UniformScaling, m::Integer, n::Integer) = T(s, Dims((m, n)))

function Base.copyto!(A::AbstractGPUMatrix{T}, s::UniformScaling) where T
    fill!(A, zero(T))
    gpu_call(identity_kernel, A, size(A, 1), s.λ; elements=minimum(size(A)))
    A
end

function _one(unit::T, x::AbstractGPUMatrix) where {T}
    m,n = size(x)
    m==n || throw(DimensionMismatch("multiplicative identity defined only for square matrices"))
    I = similar(x, T)
    fill!(I, zero(T))
    gpu_call(identity_kernel, I, m, unit; elements=m)
    I
end

Base.one(x::AbstractGPUMatrix{T}) where {T} = _one(one(T), x)
Base.oneunit(x::AbstractGPUMatrix{T}) where {T} = _one(oneunit(T), x)
