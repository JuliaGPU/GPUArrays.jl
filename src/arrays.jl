type GPUArray{B, T, N, C} <: DenseArray{T, N}
  buffer::B
  size::NTuple{N, Int}
  context::C
end

function GPUArray(buffer, sz, ctx=compute_context)
  b, T, N = buffer, eltype(buffer), length(sz)
  GPUArray{typeof(b), T, N, typeof(ctx)}(buffer, sz, ctx)
end

function GPUArray{T, N}(A::AbstractArray{T, N}, flag=:rw, ctx=compute_context)
  b = create_buffer(compute_context, A, flag)
  GPUArray{typeof(b), T, N, typeof(ctx)}(
    b, size(A), ctx
  )
end

Base.eltype{B, T, N, C}(::Type{GPUArray{B, T, N, C}}) = T
Base.eltype{B, T}(::GPUArray{B, T}) = T
Base.size(A::GPUArray) = A.size
Base.size(A::GPUArray, i::Int) = A.size[i]
buffer(A::GPUArray) = A.buffer

export buffer