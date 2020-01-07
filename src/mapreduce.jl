#############################
# reduce
# functions in base implemented with a direct loop need to be overloaded to use mapreduce


Base.any(A::GPUArray{Bool}) = mapreduce(identity, |, A; init = false)
Base.all(A::GPUArray{Bool}) = mapreduce(identity, &, A; init = true)

Base.any(f::Function, A::GPUArray) = mapreduce(f, |, A; init = false)
Base.all(f::Function, A::GPUArray) = mapreduce(f, &, A; init = true)
Base.count(pred::Function, A::GPUArray) = Int(mapreduce(pred, +, A; init = 0))

Base.:(==)(A::GPUArray, B::GPUArray) = Bool(mapreduce(==, &, A, B; init = true))

LinearAlgebra.ishermitian(A::GPUMatrix) = acc_mapreduce(==, &, true, A, (adjoint(A),))

# hack to get around of fetching the first element of the GPUArray
# as a startvalue, which is a bit complicated with the current reduce implementation
_initerror(f) = error("Please supply a neutral element for $f. E.g: mapreduce(f, $f, A; init = 1)")
startvalue(f, T) = _initerror(f)
for op = (+, Base.add_sum, *, Base.mul_prod, max, min)
    @eval startvalue(::typeof($op), ::Type{Any}) = _initerror($op)
end

startvalue(::typeof(+), T) = zero(T)
startvalue(::typeof(Base.add_sum), T) = zero(T)
startvalue(::typeof(*), T) = one(T)
startvalue(::typeof(Base.mul_prod), T) = one(T)

startvalue(::typeof(max), T) = typemin(T)
startvalue(::typeof(min), T) = typemax(T)

# TODO mirror base

if Int === Int32
const SmallSigned = Union{Int8,Int16}
const SmallUnsigned = Union{UInt8,UInt16}
else
const SmallSigned = Union{Int8,Int16,Int32}
const SmallUnsigned = Union{UInt8,UInt16,Int}
end

const CommonReduceResult = Union{UInt64,UInt128,Int64,Int128,Float16,Float32,Float64}
const WidenReduceResult = Union{SmallSigned, SmallUnsigned}


# TODO widen and support Int64 and use Base.r_promote_type
gpu_promote_type(op, ::Type{T}) where {T} = T
gpu_promote_type(op, ::Type{T}) where {T<: WidenReduceResult} = T
gpu_promote_type(::typeof(+), ::Type{T}) where {T<: WidenReduceResult} = T
gpu_promote_type(::typeof(*), ::Type{T}) where {T<: WidenReduceResult} = T
gpu_promote_type(::typeof(Base.add_sum), ::Type{T}) where {T<:WidenReduceResult} = typeof(Base.add_sum(zero(T), zero(T)))
gpu_promote_type(::typeof(Base.mul_prod), ::Type{T}) where {T<:WidenReduceResult} = typeof(Base.mul_prod(one(T), one(T)))
gpu_promote_type(::typeof(+), ::Type{T}) where {T<:Number} = typeof(zero(T)+zero(T))
gpu_promote_type(::typeof(*), ::Type{T}) where {T<:Number} = typeof(one(T)*one(T))
gpu_promote_type(::typeof(Base.add_sum), ::Type{T}) where {T<:Number} = typeof(Base.add_sum(zero(T), zero(T)))
gpu_promote_type(::typeof(Base.mul_prod), ::Type{T}) where {T<:Number} = typeof(Base.mul_prod(one(T), one(T)))
gpu_promote_type(::typeof(max), ::Type{T}) where {T<: WidenReduceResult} = T
gpu_promote_type(::typeof(min), ::Type{T}) where {T<: WidenReduceResult} = T
gpu_promote_type(::typeof(abs), ::Type{Complex{T}}) where {T} = T
gpu_promote_type(::typeof(abs2), ::Type{Complex{T}}) where {T} = T

import Base.Broadcast: Broadcasted, ArrayStyle
const GPUSrcArray = Union{Broadcasted{ArrayStyle{AT}}, GPUArray{T, N}} where {T, N, AT<:GPUArray}

function Base.mapreduce(f::Function, op::Function, A::GPUSrcArray; dims = :, init...)
    mapreduce_impl(f, op, init.data, A, dims)
end

function mapreduce_impl(f, op, ::NamedTuple{()}, A::GPUSrcArray, ::Colon)
    OT = gpu_promote_type(op, gpu_promote_type(f, eltype(A)))
    v0 = startvalue(op, OT) # TODO do this better
    acc_mapreduce(f, op, v0, A, ())
end

function mapreduce_impl(f, op, nt::NamedTuple{(:init,)}, A::GPUSrcArray, ::Colon)
    acc_mapreduce(f, op, nt.init, A, ())
end

function mapreduce_impl(f, op, nt, A::GPUSrcArray, dims)
    Base._mapreduce_dim(f, op, nt, A, dims)
end

function acc_mapreduce end
function Base.mapreduce(f, op, A::GPUSrcArray, B::GPUSrcArray, C::Number; init)
    acc_mapreduce(f, op, init, A, (B, C))
end
function Base.mapreduce(f, op, A::GPUSrcArray, B::GPUSrcArray; init)
    acc_mapreduce(f, op, init, A, (B,))
end

@generated function mapreducedim_kernel(state, f, op, R, A, range::NTuple{N, Any}) where N
    types = (range.parameters...,)
    indices = ntuple(i-> Symbol("I_$i"), N)
    Iexpr = ntuple(i-> :(I[$i]), N)
    body = :(@inbounds R[$(Iexpr...)] = op(R[$(Iexpr...)], f(A[$(indices...)])))
    for i = N:-1:1
        idxsym = indices[i]
        if types[i] == Nothing
            body = quote
                $idxsym = I[$i]
                $body
            end
        else
            rsym = Symbol("r_$i")
            body = quote
                $(rsym) = range[$i]
                for $idxsym in Int(first($rsym)):Int(last($rsym))
                    $body
                end
            end
        end
        body
    end
    quote
        I = @cartesianidx R state
        $body
        return
    end
end

function Base._mapreducedim!(f, op, R::GPUArray, A::GPUSrcArray)
    range = ifelse.(length.(axes(R)) .== 1, axes(A), nothing)
    gpu_call(mapreducedim_kernel, R, (f, op, R, A, range))
    return R
end

@inline simple_broadcast_index(A::AbstractArray, i...) = @inbounds A[i...]
@inline simple_broadcast_index(x, i...) = x

for i = 0:10
    args = ntuple(x-> Symbol("arg_", x), i)
    fargs = ntuple(x-> :(simple_broadcast_index($(args[x]), cartesian_global_index...)), i)
    @eval begin
        # http://developer.amd.com/resources/articles-whitepapers/opencl-optimization-case-study-simple-reductions/
        function reduce_kernel(state, f, op, v0::T, A, ::Val{LMEM}, result, $(args...)) where {T, LMEM}
            tmp_local = @LocalMemory(state, T, LMEM)
            global_index = linear_index(state)
            acc = v0
            # # Loop sequentially over chunks of input vector
            @inbounds while global_index <= length(A)
                cartesian_global_index = Tuple(CartesianIndices(axes(A))[global_index])
                @inbounds element = f(A[cartesian_global_index...], $(fargs...))
                acc = op(acc, element)
                global_index += global_size(state)
            end
            # Perform parallel reduction
            local_index = threadidx_x(state) - 1
            @inbounds tmp_local[local_index + 1] = acc
            synchronize_threads(state)

            offset = blockdim_x(state) ÷ 2
            @inbounds while offset > 0
                if (local_index < offset)
                    other = tmp_local[local_index + offset + 1]
                    mine = tmp_local[local_index + 1]
                    tmp_local[local_index + 1] = op(mine, other)
                end
                synchronize_threads(state)
                offset = offset ÷ 2
            end
            if local_index == 0
                @inbounds result[blockidx_x(state)] = tmp_local[1]
            end
            return
        end
    end

end

function acc_mapreduce(f, op, v0::OT, A::GPUSrcArray, rest::Tuple) where {OT}
    blocksize = 80
    threads = 256
    if length(A) <= blocksize * threads
        args = zip(convert_to_cpu(A), convert_to_cpu.(rest)...)
        return mapreduce(x-> f(x...), op, args, init = v0)
    end
    out = similar(A, OT, (blocksize,))
    fill!(out, v0)
    args = (f, op, v0, A, Val{threads}(), out, rest...)
    gpu_call(reduce_kernel, out, args, ((blocksize,), (threads,)))
    reduce(op, Array(out))
end

"""
Same as Base.isapprox, but without keyword args and without nans
"""
function fast_isapprox(x::Number, y::Number, rtol::Real = Base.rtoldefault(x, y), atol::Real=0)
    x == y || (isfinite(x) && isfinite(y) && abs(x-y) <= max(atol, rtol*max(abs(x), abs(y))))
end

Base.isapprox(A::GPUArray{T1}, B::GPUArray{T2}, rtol::Real = Base.rtoldefault(T1, T2, 0), atol::Real=0) where {T1, T2} = all(fast_isapprox.(A, B, T1(rtol)|>real, T1(atol)|>real))
Base.isapprox(A::AbstractArray{T1}, B::GPUArray{T2}, rtol::Real = Base.rtoldefault(T1, T2, 0), atol::Real=0) where {T1, T2} = all(fast_isapprox.(A, Array(B), T1(rtol)|>real, T1(atol)|>real))
Base.isapprox(A::GPUArray{T1}, B::AbstractArray{T2}, rtol::Real = Base.rtoldefault(T1, T2, 0), atol::Real=0) where {T1, T2} = all(fast_isapprox.(Array(A), B, T1(rtol)|>real, T1(atol)|>real))
