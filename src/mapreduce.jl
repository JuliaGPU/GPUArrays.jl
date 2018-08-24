#############################
# reduce
# functions in base implemented with a direct loop need to be overloaded to use mapreduce


Base.any(A::GPUArray{Bool}) = mapreduce(identity, |, A; init = false)
Base.all(A::GPUArray{Bool}) = mapreduce(identity, &, A; init = true)

Base.any(f::Function, A::GPUArray) = mapreduce(f, |, A; init = false)
Base.all(f::Function, A::GPUArray) = mapreduce(f, &, A; init = true)
Base.count(pred::Function, A::GPUArray) = Int(mapreduce(pred, +, A; init = 0))

Base.:(==)(A::GPUArray, B::GPUArray) = Bool(mapreduce(==, &, A, B; init = true))

# hack to get around of fetching the first element of the GPUArray
# as a startvalue, which is a bit complicated with the current reduce implementation
function startvalue(f, T)
    error("Please supply a starting value for mapreduce. E.g: mapreduce(func, $f, A; init = 1)")
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

function Base.mapreduce(f::Function, op::Function, A::GPUArray{T, N}; dims = :, init...) where {T, N}
    mapreduce_impl(f, op, init.data, A, dims)
end

function mapreduce_impl(f, op, ::NamedTuple{()}, A::GPUArray{T, N}, ::Colon) where {T, N}
    OT = gpu_promote_type(op, T)
    v0 = startvalue(op, OT) # TODO do this better
    acc_mapreduce(f, op, v0, A, ())
end

function mapreduce_impl(f, op, nt::NamedTuple{(:init,)}, A::GPUArray{T, N}, ::Colon) where {T, N}
    acc_mapreduce(f, op, nt.init, A, ())
end

function mapreduce_impl(f, op, nt, A::GPUArray{T, N}, dims) where {T, N}
    Base._mapreduce_dim(f, op, nt, A, dims)
end

function acc_mapreduce end
function Base.mapreduce(f, op, A::GPUArray, B::GPUArray, C::Number; init)
    acc_mapreduce(f, op, init, A, (B, C))
end
function Base.mapreduce(f, op, A::GPUArray, B::GPUArray; init)
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

function Base._mapreducedim!(f, op, R::GPUArray, A::GPUArray)
    range = ifelse.(length.(axes(R)) .== 1, axes(A), nothing)
    gpu_call(mapreducedim_kernel, R, (f, op, R, A, range))
    return R
end

simple_broadcast_index(A::AbstractArray, i) = A[i]
simple_broadcast_index(x, i) = x

for i = 0:10
    args = ntuple(x-> Symbol("arg_", x), i)
    fargs = ntuple(x-> :(simple_broadcast_index($(args[x]), global_index)), i)
    @eval begin
        # http://developer.amd.com/resources/articles-whitepapers/opencl-optimization-case-study-simple-reductions/
        function reduce_kernel(state, f, op, v0::T, A, ::Val{LMEM}, result, $(args...)) where {T, LMEM}
            tmp_local = @LocalMemory(state, T, LMEM)
            global_index = linear_index(state)
            acc = v0
            # # Loop sequentially over chunks of input vector
            while global_index <= length(A)
                element = f(A[global_index], $(fargs...))
                acc = op(acc, element)
                global_index += global_size(state)
            end
            # Perform parallel reduction
            local_index = threadidx_x(state) - 1
            tmp_local[local_index + 1] = acc
            synchronize_threads(state)

            offset = blockdim_x(state) ÷ 2
            while offset > 0
                if (local_index < offset)
                    other = tmp_local[local_index + offset + 1]
                    mine = tmp_local[local_index + 1]
                    tmp_local[local_index + 1] = op(mine, other)
                end
                synchronize_threads(state)
                offset = offset ÷ 2
            end
            if local_index == 0
                result[blockidx_x(state)] = tmp_local[1]
            end
            return
        end
    end

end

to_cpu(x) = x
to_cpu(x::GPUArray) = Array(x)

function acc_mapreduce(
        f, op, v0::OT, A::GPUArray{T, N}, rest::Tuple
    ) where {T, OT, N}
    dev = device(A)
    blocksize = 80
    threads = 256
    if length(A) <= blocksize * threads
        args = zip(Array(A), to_cpu.(rest)...)
        return mapreduce(x-> f(x...), op, args, init = v0)
    end
    out = similar(A, OT, (blocksize,))
    fill!(out, v0)
    args = (f, op, v0, A, Val{threads}(), out, rest...)
    gpu_call(reduce_kernel, A, args, ((blocksize,), (threads,)))
    reduce(op, Array(out))
end

"""
Same as Base.isapprox, but without keyword args and without nans
"""
function fast_isapprox(x::Number, y::Number, rtol::Real = Base.rtoldefault(x, y), atol::Real=0)
    x == y || (isfinite(x) && isfinite(y) && abs(x - y) <= atol + rtol*max(abs(x), abs(y)))
end

Base.isapprox(A::GPUArray{T1}, B::GPUArray{T2}, rtol::Real = Base.rtoldefault(T1, T2, 0), atol::Real=0) where {T1, T2} = all(fast_isapprox.(A, B, T1(rtol), T1(atol)))
Base.isapprox(A::AbstractArray{T1}, B::GPUArray{T2}, rtol::Real = Base.rtoldefault(T1, T2, 0), atol::Real=0) where {T1, T2} = all(fast_isapprox.(A, Array(B), T1(rtol), T1(atol)))
Base.isapprox(A::GPUArray{T1}, B::AbstractArray{T2}, rtol::Real = Base.rtoldefault(T1, T2, 0), atol::Real=0) where {T1, T2} = all(fast_isapprox.(Array(A), B, T1(rtol), T1(atol)))
