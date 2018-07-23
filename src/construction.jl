import Base: fill!, similar, zeros, ones, fill
import LinearAlgebra: eye


function fill(X::Type{<: GPUArray}, val, dims::Integer...)
    fill(X, val, dims)
end
function fill(X::Type{<: GPUArray}, val::T, dims::NTuple{N, Integer}) where {T, N}
    res = similar(X, T, dims)
    fill!(res, val)
end

zeros(T::Type{<: GPUArray}, dims::NTuple{N, Integer}) where N = fill(T, zero(eltype(T)), dims)
ones(T::Type{<: GPUArray}, dims::NTuple{N, Integer}) where N = fill(T, one(eltype(T)), dims)

function eyekernel(state, res::AbstractArray{T}, stride) where T
    i = linear_index(state)
    i > stride && return
    ilin = (stride * (i - UInt32(1))) + i
    @inbounds res[ilin] = one(T)
    return
end

eye(T::Type{<: GPUArray}, i1::Integer) = eye(T, (i1, i1))
eye(T::Type{<: GPUArray}, i1::Integer, i2::Integer) = eye(T, (i1, i2))
function eye(T::Type{<: GPUArray}, dims::NTuple{2, Integer})
    res = zeros(T, dims)
    gpu_call(eyekernel, res, (res, UInt32(size(res, 1))), minimum(dims))
    res
end

(T::Type{<: GPUArray})(x) = convert(T, x)
(T::Type{<: GPUArray})(dims::Integer...) = T(dims)
(T::Type{<: GPUArray})(dims::NTuple{N, Base.OneTo{Int}}) where N = T(undef, length.(dims))
(T::Type{<: GPUArray{X} where X})(dims::NTuple{N, Integer}) where N = similar(T, eltype(T), dims)
(T::Type{<: GPUArray{X} where X})(::UndefInitializer, dims::NTuple{N, Integer}) where N = similar(T, eltype(T), dims)

similar(x::X, ::Type{T}, size::Base.Dims{N}) where {X <: GPUArray, T, N} = similar(X, T, size)
similar(::Type{X}, ::Type{T}, size::NTuple{N, Base.OneTo{Int}}) where {X <: GPUArray, T, N} = similar(X, T, length.(size))

convert(AT::Type{<: GPUArray{T, N}}, A::GPUArray{T, N}) where {T, N} = A

function indexstyle(x::T) where T
    style = try
        Base.IndexStyle(x)
    catch
        nothing
    end
    style
end

function collect_kernel(state, A, iter, ::IndexCartesian)
    idx = @cartesianidx(A, state)
    @inbounds A[idx...] = iter[idx...]
    return
end

function collect_kernel(state, A, iter, ::IndexLinear)
    idx = linear_index(state)
    @inbounds A[idx] = iter[idx]
    return
end

eltype_or(::Type{<: GPUArray}, or) = or
eltype_or(::Type{<: GPUArray{T}}, or) where T = T
eltype_or(::Type{<: GPUArray{T, N}}, or) where {T, N} = T

function convert(AT::Type{<: GPUArray}, iter)
    isize = Base.IteratorSize(iter)
    style = indexstyle(iter)
    ettrait = Base.IteratorEltype(iter)
    if isbits(iter) && isa(isize, Base.HasShape) && style != nothing && isa(ettrait, Base.HasEltype)
        # We can collect on the GPU
        A = similar(AT, eltype_or(AT, eltype(iter)), size(iter))
        gpu_call(collect_kernel, A, (A, iter, style))
        A
    else
        convert(AT, collect(iter))
    end
end

function convert(AT::Type{<: GPUArray{T, N}}, A::DenseArray{T, N}) where {T, N}
    copyto!(AT(Base.size(A)), A)
end

function convert(AT::Type{<: GPUArray{T1}}, A::DenseArray{T2, N}) where {T1, T2, N}
    copyto!(similar(AT, T1, size(A)), convert(Array{T1, N}, A))
end
function convert(AT::Type{<: GPUArray}, A::DenseArray{T2, N}) where {T2, N}
    copyto!(similar(AT, T2, size(A)), A)
end

function convert(AT::Type{Array{T, N}}, A::GPUArray{CT, CN}) where {T, N, CT, CN}
    convert(AT, copyto!(Array{CT, CN}(undef, Int.(Base.size(A))), A))
end
