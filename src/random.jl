## device interface

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

to_number_range(x::F, ::Type{T}) where {T <: Signed, F <: AbstractFloat} = Base.unsafe_trunc(T, round(((x - F(0.5)) * typemax(T)) * T(2)))

function gpu_rand(::Type{T}, state, randstate::AbstractVector{NTuple{4, UInt32}}) where T <: Integer
    f = gpu_rand(floattype(T), state, randstate)
    return to_number_range(f, T)
end


## host interface

struct RNG <: AbstractRNG
    state::GPUArray{NTuple{4,UInt32},1}

    function RNG(A::GPUArray)
        dev = GPUArrays.device(A)
        N = GPUArrays.threads(dev)
        state = similar(A, NTuple{4, UInt32}, N)
        copyto!(state, [ntuple(i-> rand(UInt32), 4) for i=1:N])
        new(state)
    end
end

const GLOBAL_RNGS = Dict()
function global_rng(A::GPUArray)
    dev = GPUArrays.device(A)
    get!(GLOBAL_RNGS, dev) do
        RNG(A)
    end
end

function Random.rand!(rng::RNG, A::GPUArray{T}) where T <: Number
    gpu_call(A, (rng.state, A,)) do state, randstates, a
        idx = linear_index(state)
        idx > length(a) && return
        @inbounds a[idx] = gpu_rand(T, state, randstates)
        return
    end
    A
end

Random.rand!(A::GPUArray) = rand!(global_rng(A), A)
