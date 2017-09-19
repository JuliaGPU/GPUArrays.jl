import Base: fill!, similar, eye, zeros, fill


function fill(X::Type{<: GPUArray}, val, dims::Integer...)
    fill(X, val, dims)
end
function fill(X::Type{<: GPUArray}, val::T, dims::NTuple{N, Integer}) where {T, N}
    res = similar(X, T, dims)
    fill!(res, val)
end

function fill!{T, N}(A::GPUArray{T, N}, val)
    valconv = T(val)
    gpu_call(const_kernel2, A, (A, valconv, Cuint(length(A))))
    A
end

zeros(T::Type{<: GPUArray}, dims::NTuple{N, Integer}) where N = fill(T, zero(eltype(T)), dims)

function eyekernel(state, res::AbstractArray{T}, stride) where T
    i = linear_index(state)
    ilin = (stride * (i - 1)) + i
    @inbounds res[ilin] = one(T)
    return
end

eye(T::Type{<: GPUArray}, i1::Integer) = eye(T, (i1, i1))
eye(T::Type{<: GPUArray}, i1::Integer, i2::Integer) = eye(T, (i1, i2))
function eye(T::Type{<: GPUArray}, dims::NTuple{2, Integer})
    res = zeros(T, dims)
    gpu_call(eyekernel, res, (res, Cuint(size(res, 1))), minimum(dims))
    res
end

(T::Type{<: GPUArray})(dims::Integer...) = T(dims)
(T::Type{<: GPUArray{X} where X})(dims::NTuple{N, Integer}) where N = similar(T, eltype(T), dims)

similar(x::X, ::Type{T}, size::Base.Dims{N}) where {X <: GPUArray, T, N} = similar(X, T, size)

function convert(AT::Type{<: GPUArray{T, N}}, A::DenseArray{T, N}) where {T, N}
    copy!(AT(Base.size(A)), A)
end
function convert(AT::Type{<: GPUArray{T1}}, A::DenseArray{T2}) where {T1, T2}
    copy!(similar(AT, T1, size(A)), T1.(A))
end
function convert(AT::Type{<: GPUArray}, A::DenseArray{T2, N}) where {T2, N}
    copy!(similar(AT, T2, size(A)), A)
end

function convert(AT::Type{Array{T, N}}, A::GPUArray{CT, CN}) where {T, N, CT, CN}
    convert(AT, copy!(Array{CT, CN}(Int.(Base.size(A))), A))
end