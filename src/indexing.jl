const _allowslow = Ref(true)

allowslow(flag = true) = (_allowslow[] = flag)

function assertslow(op = "Operation")
    # _allowslow[] || error("$op is disabled")
    return
end

Base.IndexStyle(::Type{<:GPUArray}) = Base.IndexLinear()

function _getindex(xs::GPUArray{T}, i::Integer) where T
    x = Array{T}(undef, 1)
    copyto!(x, 1, xs, i, 1)
    return x[1]
end

function Base.getindex(xs::GPUArray{T}, i::Integer) where T
    assertslow("getindex")
    _getindex(xs, i)
end

function _setindex!(xs::GPUArray{T}, v::T, i::Integer) where T
    x = T[v]
    copyto!(xs, i, x, 1, 1)
    return v
end

function Base.setindex!(xs::GPUArray{T}, v::T, i::Integer) where T
    assertslow("setindex!")
    _setindex!(xs, v, i)
end

Base.setindex!(xs::GPUArray, v, i::Integer) = xs[i] = convert(eltype(xs), v)

# Vector indexing

to_index(a, x) = x
to_index(::A, x::Array{ET}) where {A, ET} = copyto!(similar(A, ET, size(x)), x)
to_index(a, x::UnitRange{<: Integer}) = convert(UnitRange{Int}, x)
to_index(a, x::Base.LogicalIndex) = error("Logical indexing not implemented")

@generated function index_kernel(state, dest::AbstractArray, src::AbstractArray, idims, Is)
    N = length(Is.parameters)
    quote
        i = linear_index(state)
        i > length(dest) && return
        is = gpu_ind2sub(idims, i)
        @nexprs $N i -> @inbounds I_i = Is[i][is[i]]
        @inbounds dest[i] = @ncall $N getindex src i -> I_i
        return
    end
end


function Base._unsafe_getindex!(dest::GPUArray, src::GPUArray, Is::Union{Real, AbstractArray}...)
    if length(Is) == 1 && isa(first(Is), Array) && isempty(first(Is)) # indexing with empty array
        return dest
    end
    idims = map(length, Is)
    gpu_call(index_kernel, dest, (dest, src, idims, map(x-> to_index(dest, x), Is)))
    return dest
end

# simple broadcast getindex like function... could reuse another?
@inline bgetindex(x::AbstractArray, i) = x[i]
@inline bgetindex(x, i) = x

@generated function setindex_kernel!(state, dest::AbstractArray, src, idims, Is, len)
    N = length(Is.parameters)
    idx = ntuple(i-> :(Is[$i][is[$i]]), N)
    quote
        i = linear_index(state)
        i > len && return
        is = gpu_ind2sub(idims, i)
        @inbounds setindex!(dest, bgetindex(src, i), $(idx...))
        return
    end
end


#TODO this should use adapt, but I currently don't have time to figure out it's intended usage

gpu_convert(GPUType, x::GPUArray) = x
function gpu_convert(GPUType, x::AbstractArray)
    isbits(x) ? x : convert(GPUType, x)
end
function gpu_convert(GPUType, x)
    isbits(x) ? x : error("Only isbits types are allowed for indexing. Found: $(typeof(x))")
end

function Base._unsafe_setindex!(::IndexStyle, dest::T, src, Is::Union{Real, AbstractArray}...) where T <: GPUArray
    if length(Is) == 1 && isa(first(Is), Array) && isempty(first(Is)) # indexing with empty array
        return dest
    end
    idims = length.(Is)
    len = prod(idims)
    src_gpu = gpu_convert(T, src)
    gpu_call(setindex_kernel!, dest, (dest, src_gpu, idims, map(x-> to_index(dest, x), Is), len), len)
    return dest
end
