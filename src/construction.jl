function Base.fill!{T, N}(A::GPUArray{T, N}, val)
    valconv = T(val)
    gpu_call(const_kernel2, A, (A, valconv, Cuint(length(A))))
    A
end
function Base.rand{T <: GPUArray, ET}(::Type{T}, ::Type{ET}, size...)
    T(rand(ET, size...))
end
