function TausStep(z::Unsigned, S1::Integer, S2::Integer, S3::Integer, M::Unsigned)
    b = (((z << S1) ⊻ z) >> S2)
    return (((z & M) << S3) ⊻ b)
end

LCGStep(z::Unsigned, A::Unsigned, C::Unsigned) = A * z + C

make_rand_num(::Type{Float64}, tmp) = 2.3283064365387e-10 * Float64(tmp)
make_rand_num(::Type{Float32}, tmp) = 2.3283064f-10 * Float32(tmp)


function next_rand(::Type{FT}, state::NTuple{4, T}) where {FT, T <: Unsigned}
    state = (
        TausStep(state[1], Cint(13), Cint(19), Cint(12), T(4294967294)),
        TausStep(state[2], Cint(2), Cint(25), Cint(4), T(4294967288)),
        TausStep(state[3], Cint(3), Cint(11), Cint(17), T(4294967280)),
        LCGStep(state[4], T(1664525), T(1013904223))
    )
    tmp = (state[1] ⊻ state[2] ⊻ state[3] ⊻ state[4])
    return (
        state,
        make_rand_num(FT, tmp)
    )
end

function gpu_rand(::Type{T}, state, randstate::AbstractVector{NTuple{4, UInt32}}) where T
    threadid = GPUArrays.threadidx_x(state)
    stateful_rand = next_rand(T, randstate[threadid])
    randstate[threadid] = stateful_rand[1]
    return stateful_rand[2]
end

floattype(::Type{T}) where T <: Union{Int64, UInt64} = Float64
floattype(::Type{T}) where T <: Union{Int32, UInt32} = Float32

to_number_range(x::AbstractFloat, ::Type{T}) where T <: Unsigned = T(round(x * typemax(T)))
to_number_range(x::F, ::Type{T}) where {T <: Signed, F <: AbstractFloat} = T(round(((x - F(0.5)) * typemax(T)) * T(2)))

function gpu_rand(::Type{T}, state, randstate::AbstractVector{NTuple{4, UInt32}}) where T <: Integer
    f = gpu_rand(floattype(T), state, randstate)
    return to_number_range(f, T)
end

global cached_state, clear_cache
let rand_state_dict = Dict()
    clear_cache() = (empty!(rand_state_dict); return)
    function cached_state(x)
        dev = GPUArrays.device(x)
        get!(rand_state_dict, dev) do
            N = GPUArrays.threads(dev)
            res = similar(x, NTuple{4, UInt32}, N)
            copyto!(res, [ntuple(i-> rand(UInt32), 4) for i=1:N])
            res
        end
    end
end
function rand!(A::GPUArray{T}) where T <: Number
    rstates = cached_state(A)
    gpu_call(A, (rstates, A,)) do state, randstates, a
        idx = linear_index(state)
        idx > length(a) && return
        @inbounds a[idx] = gpu_rand(T, state, randstates)
        return
    end
    A
end

rand(X::Type{<: GPUArray}, i::Integer...) = rand(X, Float32, i...)
rand(X::Type{<: GPUArray}, size::NTuple{N, Int}) where N = rand(X, Float32, size...)
rand(X::Type{<: GPUArray{T}}, i::Integer...) where T = rand(X, T, i...)
rand(X::Type{<: GPUArray{T}}, size::NTuple{N, Int}) where {T, N} = rand(X, T, size...)
rand(X::Type{<: GPUArray{T, N}}, size::NTuple{N, Integer}) where {T, N} = rand(X, T, size...)
rand(X::Type{<: GPUArray{T, N}}, size::NTuple{N, Int}) where {T, N} = rand(X, T, size...)

function rand(X::Type{<: GPUArray}, ::Type{ET}, size::Integer...) where ET
    A = similar(X, ET, size)
    rand!(A)
end
