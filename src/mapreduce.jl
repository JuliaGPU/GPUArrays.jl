#############################
# reduce
# functions in base implemented with a direct loop need to be overloaded to use mapreduce
any(pred, A::GPUArray) = Bool(mapreduce(pred, |, Cint(0), (u)))
count(pred, A::GPUArray) = Int(mapreduce(pred, +, Cuint(0), A))


# hack to get around of fetching the first element of the GPUArray
# as a startvalue, which is a bit complicated with the current reduce implementation
function startvalue(f, T)
    error("Please supply a starting value for mapreduce. E.g: mapreduce($f, $op, 1, A)")
end
startvalue(::typeof(+), T) = zero(T)
startvalue(::typeof(*), T) = one(T)
startvalue(::typeof(Base.scalarmin), T) = typemax(T)
startvalue(::typeof(Base.scalarmax), T) = typemin(T)

# TODO widen and support Int64 and use Base.r_promote_type
gpu_promote_type{T}(op, ::Type{T}) = T
gpu_promote_type{T<:Base.WidenReduceResult}(op, ::Type{T}) = T
gpu_promote_type{T<:Base.WidenReduceResult}(::typeof(+), ::Type{T}) = T
gpu_promote_type{T<:Base.WidenReduceResult}(::typeof(*), ::Type{T}) = T
gpu_promote_type{T<:Number}(::typeof(+), ::Type{T}) = typeof(zero(T)+zero(T))
gpu_promote_type{T<:Number}(::typeof(*), ::Type{T}) = typeof(one(T)*one(T))
gpu_promote_type{T<:Base.WidenReduceResult}(::typeof(Base.scalarmax), ::Type{T}) = T
gpu_promote_type{T<:Base.WidenReduceResult}(::typeof(Base.scalarmin), ::Type{T}) = T
gpu_promote_type{T<:Base.WidenReduceResult}(::typeof(max), ::Type{T}) = T
gpu_promote_type{T<:Base.WidenReduceResult}(::typeof(min), ::Type{T}) = T

function Base.mapreduce{T, N}(f::Function, op::Function, A::GPUArray{T, N})
    OT = gpu_promote_type(op, T)
    v0 = startvalue(op, OT) # TODO do this better
    mapreduce(f, op, v0, A)
end
function acc_mapreduce end
function Base.mapreduce(f, op, v0, A::GPUArray, B::GPUArray, C::Number)
    acc_mapreduce(f, op, v0, A, (B, C))
end
function Base.mapreduce(f, op, v0, A::GPUArray, B::GPUArray)
    acc_mapreduce(f, op, v0, A, (B,))
end
function Base.mapreduce(f, op, v0, A::GPUArray)
    acc_mapreduce(f, op, v0, A, ())
end


function mapreducedim_kernel(state, f, op, R::AbstractArray{T1, N}, A::AbstractArray{T, N}, slice_size, sizeA, dim) where {T1, T, N}
    ilin = Cuint(linear_index(state))
    accum = zero(T1)
    @inbounds for i = Cuint(1):slice_size
        idx = N == dim ? (ilin, i) : (i, ilin)
        i2d = gpu_sub2ind(sizeA, idx)
        accum = op(accum, f(A[i2d]))
    end
    R[ilin] = accum
    return
end
function Base._mapreducedim!(f, op, R::GPUArray, A::GPUArray)
    sizeR = size(R)
    if all(x-> x == 1, sizeR)
        x = mapreduce(f, op, A)
        copy!(R, reshape([x], sizeR))
        return R
    end
    @assert count(x-> x == 1, sizeR) == (ndims(R) - 1) "Not implemented"
    dim = findfirst(x-> x == 1, sizeR)
    slice_size = size(A, dim)
    gpu_call(mapreducedim_kernel, R, (f, op, R, A, Cuint(slice_size), Cuint.(size(A)), Cuint(dim)))
    return R
end
