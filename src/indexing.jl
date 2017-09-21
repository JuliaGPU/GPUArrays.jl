const _allowslow = Ref(true)

allowslow(flag = true) = (_allowslow[] = flag)

function assertslow(op = "Operation")
    _allowslow[] || error("$op is disabled")
    return
end

Base.IndexStyle(::Type{<:GPUArray}) = IndexLinear()

function _getindex(xs::GPUArray{T}, i::Integer) where T
    x = Array{T}(1)
    copy!(x, 1, xs, i, 1)
    return x[1]
end

function Base.getindex(xs::GPUArray{T}, i::Integer) where T
    assertslow("getindex")
    _getindex(xs, i)
end

function _setindex!(xs::GPUArray{T}, v::T, i::Integer) where T
    x = T[v]
    copy!(xs, i, x, 1, 1)
    return v
end

function Base.setindex!(xs::GPUArray{T}, v::T, i::Integer) where T
    assertslow("setindex!")
    _setindex!(xs, v, i)
end

Base.setindex!(xs::GPUArray, v, i::Integer) = xs[i] = convert(eltype(xs), v)

# Vector indexing

using Base.Cartesian
to_index(a, x) = x
to_index(::A, x::Array{ET}) where {A, ET} = copy!(similar(A, ET, size(x)), x)
to_index(a, x::UnitRange{<: Integer}) = convert(UnitRange{UInt32}, x)
to_index(a, x::Base.LogicalIndex) = error("Logical indexing not implemented")

@generated function index_kernel(state, dest::AbstractArray, src::AbstractArray, idims, Is)
    N = length(Is.parameters)
    quote
        i = linear_index(state)
        i > length(dest) && return
        is = gpu_ind2sub(idims, i)
        @nexprs $N i -> @inbounds I_i = Is[i][Int(is[i])]
        @inbounds dest[i] = @ncall $N getindex src i -> I_i
        return
    end
end


function Base._unsafe_getindex!(dest::GPUArray, src::GPUArray, Is::Union{Real, AbstractArray}...)
    if length(Is) == 1 && isa(first(Is), Array) && isempty(first(Is)) # indexing with empty array
        return dest
    end
    idims = map(length, Is)
    gpu_call(index_kernel, dest, (dest, src, UInt32.(idims), map(x-> to_index(dest, x), Is)))
    return dest
end
