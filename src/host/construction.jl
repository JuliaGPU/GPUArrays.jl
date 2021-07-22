# convenience and indirect construction

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


## collect & convert

function indexstyle(x::T) where T
    style = try
        Base.IndexStyle(x)
    catch
        nothing
    end
    style
end

function collect_kernel(ctx::AbstractKernelContext, A, iter, ::IndexCartesian)
    idx = @cartesianidx(A)
    @inbounds A[idx...] = iter[idx...]
    return
end

function collect_kernel(ctx::AbstractKernelContext, A, iter, ::IndexLinear)
    idx = linear_index(ctx)
    @inbounds A[idx] = iter[idx]
    return
end

eltype_or(::Type{<: AbstractGPUArray}, or) = or
eltype_or(::Type{<: AbstractGPUArray{T}}, or) where T = T
eltype_or(::Type{<: AbstractGPUArray{T, N}}, or) where {T, N} = T

function Base.convert(AT::Type{<: AbstractGPUArray}, iter)
    isize = Base.IteratorSize(iter)
    style = indexstyle(iter)
    ettrait = Base.IteratorEltype(iter)
    if isbits(iter) && isa(isize, Base.HasShape) && style != nothing && isa(ettrait, Base.HasEltype)
        # We can collect on the GPU
        A = similar(AT, eltype_or(AT, eltype(iter)), size(iter))
        gpu_call(collect_kernel, A, iter, style)
        A
    else
        convert(AT, collect(iter))
    end
end

function Base.convert(AT::Type{<: AbstractGPUArray{T, N}}, A::DenseArray{T, N}) where {T, N}
    copyto!(AT(undef, size(A)), A)
end

function Base.convert(AT::Type{<: AbstractGPUArray{T1}}, A::DenseArray{T2, N}) where {T1, T2, N}
    copyto!(similar(AT, size(A)), convert(Array{T1, N}, A))
end

function Base.convert(AT::Type{<: AbstractGPUArray}, A::DenseArray{T2, N}) where {T2, N}
    copyto!(similar(AT{T2}, size(A)), A)
end

function Base.convert(AT::Type{Array{T, N}}, A::AbstractGPUArray{CT, CN}) where {T, N, CT, CN}
    convert(AT, copyto!(Array{CT, CN}(undef, size(A)), A))
end
