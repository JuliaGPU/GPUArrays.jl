import NNlib: adapt, adapt_

adapt_(::Type{<:GPUArray}, xs::AbstractArray) =
  isbits(xs) ? xs : convert(GPUArray, xs)

adapt_(::Type{<:GPUArray{T}}, xs::AbstractArray{<:Real}) where T <: AbstractFloat =
  isbits(xs) ? xs : convert(GPUArray{T}, xs)

# Should go in CLArrays
# cl(xs) = adapt(CLArray{Float32}, xs)
