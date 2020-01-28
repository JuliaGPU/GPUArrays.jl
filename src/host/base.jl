# common Base functionality

allequal(x) = true
allequal(x, y, z...) = x == y && allequal(y, z...)
function Base.map!(f, y::AbstractGPUArray, xs::AbstractGPUArray...)
    @assert allequal(size.((y, xs...))...)
    return y .= f.(xs...)
end
function Base.map(f, y::AbstractGPUArray, xs::AbstractGPUArray...)
    @assert allequal(size.((y, xs...))...)
    return f.(y, xs...)
end

# Break ambiguities with base
Base.map!(f, y::AbstractGPUArray) =
    invoke(map!, Tuple{Any,AbstractGPUArray,Vararg{AbstractGPUArray}}, f, y)
Base.map!(f, y::AbstractGPUArray, x::AbstractGPUArray) =
    invoke(map!, Tuple{Any,AbstractGPUArray, Vararg{AbstractGPUArray}}, f, y, x)
Base.map!(f, y::AbstractGPUArray, x1::AbstractGPUArray, x2::AbstractGPUArray) =
    invoke(map!, Tuple{Any,AbstractGPUArray, Vararg{AbstractGPUArray}}, f, y, x1, x2)

# This is pretty ugly, but I feel bad to add those to device arrays, since
# we're never bound checking... So getindex(a::AbstractGPUVector, 10, 10) would silently go unnoticed
# we need this here for easier implementation of repeat
@inline Base.@propagate_inbounds getidx_2d1d(x::AbstractVector, i, j) = x[i]
@inline Base.@propagate_inbounds getidx_2d1d(x::AbstractMatrix, i, j) = x[i, j]

function Base.repeat(a::AbstractGPUVecOrMat, m::Int, n::Int = 1)
    o, p = size(a, 1), size(a, 2)
    b = similar(a, o*m, p*n)
    gpu_call(b, a, o, p, m, n; total_threads=n) do ctx, b, a, o, p, m, n
        j = linear_index(ctx)
        j > n && return
        d = (j - 1) * p + 1
        @inbounds for i in 1:m
            c = (i - 1) * o + 1
            for r in 1:p
                for k in 1:o
                    b[k - 1 + c, r - 1 + d] = getidx_2d1d(a, k, r)
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
    gpu_call(b, a, o, m; total_threads=m) do ctx, b, a, o, m
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
