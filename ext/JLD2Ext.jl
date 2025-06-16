module JLD2Ext

using GPUArrays: AbstractGPUArray
using JLD2: JLD2

JLD2.writeas(::Type{<:AbstractGPUArray{T, N}}) where {T, N} = Array{T, N}

end
