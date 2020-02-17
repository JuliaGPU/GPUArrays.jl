# integration with Random stdlib

## device interface

# hybrid Tausworthe and Linear Congruent generator from
# https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch37.html
#
# only generates Float32 or Float64 numbers, conversion happens elsewhere

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
    return state, make_rand_num(FT, tmp)
end

function gpu_rand(::Type{T}, ctx::AbstractKernelContext, randstate::AbstractVector{NTuple{4, UInt32}}) where T
    threadid = GPUArrays.threadidx(ctx)
    stateful_rand = next_rand(T, randstate[threadid])
    randstate[threadid] = stateful_rand[1]
    return stateful_rand[2]
end

# support for integers

floattype(::Type{T}) where T <: Union{Int64, UInt64} = Float64
floattype(::Type{T}) where T <: Union{Int32, UInt32} = Float32

to_number_range(x::AbstractFloat, ::Type{T}) where T <: Unsigned = T(round(x * typemax(T)))

to_number_range(x::F, ::Type{T}) where {T <: Signed, F <: AbstractFloat} =
    Base.unsafe_trunc(T, round(((x - F(0.5)) * typemax(T)) * T(2)))

function gpu_rand(::Type{T}, ctx::AbstractKernelContext, randstate::AbstractVector{NTuple{4, UInt32}}) where T <: Integer
    f = gpu_rand(floattype(T), ctx, randstate)
    return to_number_range(f, T)
end

# support for complex numbers

function gpu_rand(::Type{Complex{T}}, ctx::AbstractKernelContext, randstate::AbstractVector{NTuple{4, UInt32}}) where T
    re = gpu_rand(T, ctx, randstate)
    im = gpu_rand(T, ctx, randstate)
    return complex(re, im)
end


## host interface

struct RNG <: AbstractRNG
    state::AbstractGPUArray{NTuple{4,UInt32},1}
end

const GLOBAL_RNGS = Dict()
function global_rng(AT::Type{<:AbstractGPUArray}, dev)
    get!(GLOBAL_RNGS, dev) do
        N = GPUArrays.threads(dev)
        AT = Base.typename(AT).wrapper
        state = AT{NTuple{4, UInt32}}(undef, N)
        rng = RNG(state)
        Random.seed!(rng)
        rng
    end
end
global_rng(A::AT) where {AT <: AbstractGPUArray} = global_rng(AT, GPUArrays.device(A))

make_seed(rng::RNG) = make_seed(rng, rand(UInt))
function make_seed(rng::RNG, n::Integer)
    rand(MersenneTwister(n), UInt32, sizeof(rng.state)÷sizeof(UInt32))
end

Random.seed!(rng::RNG) = Random.seed!(rng, make_seed(rng))
Random.seed!(rng::RNG, seed::Integer) = Random.seed!(rng, make_seed(rng, seed))
function Random.seed!(rng::RNG, seed::Vector{UInt32})
    copyto!(rng.state, reinterpret(NTuple{4, UInt32}, seed))
    return
end

function Random.rand!(rng::RNG, A::AbstractGPUArray{T}) where T <: Number
    gpu_call(A, rng.state) do ctx, a, randstates
        idx = linear_index(ctx)
        idx > length(a) && return
        @inbounds a[idx] = gpu_rand(T, ctx, randstates)
        return
    end
    A
end

Random.rand!(A::AbstractGPUArray) = rand!(global_rng(A), A)
