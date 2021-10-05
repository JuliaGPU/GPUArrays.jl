# common Base functionality

function Base.repeat(a::AbstractGPUVecOrMat, m::Int, n::Int = 1)
    o, p = size(a, 1), size(a, 2)
    b = similar(a, o*m, p*n)
    if length(b) == 0
        return b
    end
    gpu_call(b, a, o, p, m, n; elements=n) do ctx, b, a, o, p, m, n
        j = linear_index(ctx)
        j > n && return
        d = (j - 1) * p + 1
        @inbounds for i in 1:m
            c = (i - 1) * o + 1
            for r in 1:p
                for k in 1:o
                    b[k - 1 + c, r - 1 + d] = a[k, r]
                end
            end
        end
        return
    end
    return b
end

function Base.repeat(a::AbstractGPUVector, m::Int)
    o = length(a)
    b = similar(a, o*m)
    if length(b) == 0
        return b
    end
    gpu_call(b, a, o, m; elements=m) do ctx, b, a, o, m
        i = linear_index(ctx)
        i > m && return
        c = (i - 1)*o + 1
        @inbounds for i in 1:o
            b[c + i - 1] = a[i]
        end
        return
    end
    return b
end

## PermutedDimsArrays

using Base: PermutedDimsArrays

# PermutedDimsArrays' custom copyto! doesn't know how to deal with GPU arrays
function PermutedDimsArrays._copy!(dest::PermutedDimsArray{T,N,<:Any,<:Any,<:AbstractGPUArray}, src) where {T,N}
    dest .= src
    dest
end

## concatenation

# hacky overloads to make simple vcat and hcat with numbers work as expected.
# we can't really make this work in general without Base providing
# a dispatch mechanism for output container type.
@inline Base.vcat(a::Number, b::AbstractGPUArray) =
    vcat(fill!(similar(b, typeof(a), (1,size(b)[2:end]...)), a), b)
@inline Base.hcat(a::Number, b::AbstractGPUArray) =
    hcat(fill!(similar(b, typeof(a), (size(b)[1:end-1]...,1)), a), b)
