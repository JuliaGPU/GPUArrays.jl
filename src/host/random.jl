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

function next_rand(state::NTuple{4, T}) where {T <: Unsigned}
    state = (
        TausStep(state[1], Cint(13), Cint(19), Cint(12), T(4294967294)),
        TausStep(state[2], Cint(2), Cint(25), Cint(4), T(4294967288)),
        TausStep(state[3], Cint(3), Cint(11), Cint(17), T(4294967280)),
        LCGStep(state[4], T(1664525), T(1013904223))
    )
    tmp = (state[1] ⊻ state[2] ⊻ state[3] ⊻ state[4])
    return state, tmp
end

function gpu_rand(::Type{T}, ctx::AbstractKernelContext, randstate::AbstractVector{NTuple{4, UInt32}}) where T
    threadid = GPUArrays.threadidx(ctx)
    stateful_rand = next_rand(randstate[threadid])
    randstate[threadid] = stateful_rand[1]
    return make_rand_num(T, stateful_rand[2])
end

function gpu_rand(::Type{T}, ctx::AbstractKernelContext, randstate::AbstractVector{NTuple{4, UInt32}}) where T <: Integer
    threadid = GPUArrays.threadidx(ctx)
    result = T(0)
    for _ in 1:sizeof(T) >> 2
        randstate[threadid], y = next_rand(randstate[threadid])
        result = reinterpret(T, (|)(promote(result << 32, y)...))
    end
    result
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

function Random.randn!(rng::RNG, A::AbstractGPUArray{T}) where T <: Number
    threads = (length(A) - 1) ÷ 2 + 1
    length(A) == 0 && return
    gpu_call(A, rng.state; total_threads = threads) do ctx, a, randstates
        idx = 2*(linear_index(ctx) - 1) + 1
        U1 = gpu_rand(T, ctx, randstates)
        U2 = gpu_rand(T, ctx, randstates)
        Z0 = sqrt(ctx, T(-2.0)*log(ctx, U1))*cos(ctx, T(2pi)*U2)
        Z1 = sqrt(ctx, T(-2.0)*log(ctx, U1))*sin(ctx, T(2pi)*U2)
        @inbounds a[idx] = Z0
        idx + 1 > length(a) && return
        @inbounds a[idx + 1] = Z1
        return
    end
    A
end

Random.rand!(A::AbstractGPUArray) = rand!(global_rng(A), A)
Random.randn!(A::AbstractGPUArray) = randn!(global_rng(A), A)
