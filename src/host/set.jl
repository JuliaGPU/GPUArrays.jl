# unique elements, by sorting
#
# Sorting puts elements that are `isequal` next to each other, as long as the order agrees with
# `isequal`. `isless` does, and so does `isless` on the real and imaginary parts of complex
# numbers. To return elements in the order Base does, the first element of each group is marked
# at its original position, and the marked elements are selected with a mask.

# Mixed with complex numbers, a real number sorts as a complex number with zero imaginary part.
complex_key(x::Complex) = reim(x)
complex_key(x) = (x, zero(x))
complex_isless(x, y) = isless(complex_key(x), complex_key(y))

has_complex(T) = T isa Union ? any(has_complex, Base.uniontypes(T)) : T <: Complex

# Plain `isless` where possible, because AcceleratedKernels can only use its faster sorting
# algorithms with the default order.
set_isless(::Type{T}) where {T} = has_complex(T) ? complex_isless : isless

# Returns the sorting permutation of `keys` and the sorted keys. The sort is stable, so the
# first element of each group of equal keys is the one that occurs first in `keys`.
function sorted_keys(keys::AbstractVector)
    lt = set_isless(eltype(keys))
    p = sortperm(keys; lt)
    return p, keys[p]
end

# For each element of the sorted `s`, whether it differs from the one before it
function group_starts(s::AbstractVector)
    n = length(s)
    head = similar(s, Bool, n)
    n == 0 && return head
    head[1:1] .= true
    head[2:n] .= .!isequal.(view(s, 2:n), view(s, 1:n-1))
    return head
end

# Moves the flags `heads`, which are in sorted order, back to the original positions
function scatter_mask(p::AbstractVector, heads::AbstractVector{Bool})
    mask = similar(heads)
    mask[p] = heads
    return mask
end

function Base.unique(f, A::AnyGPUArray)
    v = vec(A)
    length(v) <= 1 && return copy(v)
    p, s = sorted_keys(f === identity ? v : map(f, v))
    return v[scatter_mask(p, group_starts(s))]
end

function Base.unique(A::AnyGPUArray; dims::Union{Colon, Integer}=:)
    dims isa Colon && return unique(identity, A)
    # Base returns a copy when `dims` is out of range
    1 <= dims <= ndims(A) || return copy(A)
    return unique_dims(A, Int(dims))
end

# The slices along `d` are compared in a 3-d reshape of `A`, whose middle dimension is `d`
function unique_dims(A::AbstractArray, d::Int)
    n = size(A, d)
    n <= 1 && return copy(A)
    B = reshape(A, prod(size(A)[1:d-1]), n, prod(size(A)[d+1:end]))
    ltel = set_isless(eltype(A))
    lt = let B = B, ltel = ltel
        (i, j) -> begin
            @inbounds for k in axes(B, 3), l in axes(B, 1)
                a, b = B[l, i, k], B[l, j, k]
                ltel(a, b) && return true
                ltel(b, a) && return false
            end
            return false
        end
    end
    eq = let B = B
        (i, j) -> begin
            @inbounds for k in axes(B, 3), l in axes(B, 1)
                isequal(B[l, i, k], B[l, j, k]) || return false
            end
            return true
        end
    end
    p = similar(A, Int, n)
    p .= 1:n
    sort!(p; lt)
    head = similar(A, Bool, n)
    head[1:1] .= true
    head[2:n] .= .!eq.(view(p, 1:n-1), view(p, 2:n))
    mask = scatter_mask(p, head)
    return A[ntuple(i -> i == d ? mask : Colon(), ndims(A))...]
end

function Base.unique!(f, v::AnyGPUVector)
    u = unique(f, v)
    copyto!(v, firstindex(v), u, 1, length(u))
    return resize!(v, length(u))
end
Base.unique!(v::AnyGPUVector) = unique!(identity, v)

function Base.allunique(A::AnyGPUArray)
    n = length(A)
    n <= 1 && return true
    lt = set_isless(eltype(A))
    s = sort(vec(A); lt)
    return !any(isequal.(view(s, 2:n), view(s, 1:n-1)))
end
# Base has `allunique(f, A)` since Julia 1.11
if hasmethod(allunique, Tuple{Any, Any})
    Base.allunique(f, A::AnyGPUArray) = allunique(map(f, vec(A)))
end

