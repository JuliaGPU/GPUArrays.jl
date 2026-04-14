# integration with Random stdlib


## Philox4x32-10 counter-based RNG
#
# Stateless: (counter, key) → 4 UInt32 outputs. Each unique counter gives independent
# random values with no shared memory or global state needed.
#
# Reference: Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3" (2011)

const PHILOX_M4x32_0 = 0xD2511F53
const PHILOX_M4x32_1 = 0xCD9E8D57
const PHILOX_W32_0   = 0x9E3779B9
const PHILOX_W32_1   = 0xBB67AE85

@inline function philox4x32round(ctr::NTuple{4,UInt32}, key::NTuple{2,UInt32})
    mul0 = widemul(PHILOX_M4x32_0, ctr[1])
    mul1 = widemul(PHILOX_M4x32_1, ctr[3])
    hi0 = (mul0 >> 32) % UInt32
    hi1 = (mul1 >> 32) % UInt32
    lo0 = mul0 % UInt32
    lo1 = mul1 % UInt32
    (hi1 ⊻ ctr[2] ⊻ key[1], lo1, hi0 ⊻ ctr[4] ⊻ key[2], lo0)
end

@inline function philox4x32bumpkey(key::NTuple{2,UInt32})
    (key[1] + PHILOX_W32_0, key[2] + PHILOX_W32_1)
end

@inline function philox4x32_10(ctr::NTuple{4,UInt32}, key::NTuple{2,UInt32})
    ctr = philox4x32round(ctr, key); key = philox4x32bumpkey(key)
    ctr = philox4x32round(ctr, key); key = philox4x32bumpkey(key)
    ctr = philox4x32round(ctr, key); key = philox4x32bumpkey(key)
    ctr = philox4x32round(ctr, key); key = philox4x32bumpkey(key)
    ctr = philox4x32round(ctr, key); key = philox4x32bumpkey(key)
    ctr = philox4x32round(ctr, key); key = philox4x32bumpkey(key)
    ctr = philox4x32round(ctr, key); key = philox4x32bumpkey(key)
    ctr = philox4x32round(ctr, key); key = philox4x32bumpkey(key)
    ctr = philox4x32round(ctr, key); key = philox4x32bumpkey(key)
    ctr = philox4x32round(ctr, key)
    ctr
end


## Float conversions: unsigned integer → uniform float in (0, 1]

@inline function u01(::Type{Float32}, u::UInt32)
    fma(Float32(u), Float32(2)^(-32), Float32(2)^(-33))
end

@inline function u01(::Type{Float64}, u::UInt64)
    fma(Float64(u), Float64(2)^(-64), Float64(2)^(-65))
end


## Box-Muller transform using @fastmath log + sincospi
#
# Each backend provides optimized intrinsics via @device_override:
# CUDA: __nv_fast_logf, __nv_sincospif, etc.

using Base.FastMath

@inline function boxmuller(::Type{T}, u1::T, u2::T) where T
    r = sqrt(T(-2) * FastMath.log_fast(u1))
    s, c = sincospi(2 * u2)
    (r * s, r * c)
end


## RNG type

mutable struct RNG <: AbstractRNG
    seed::UInt32
    counter::UInt32
end

RNG() = RNG(rand(Random.RandomDevice(), UInt32), UInt32(0))
RNG(seed::Integer) = RNG(seed % UInt32, UInt32(0))

# return an instance of GPUArrays.RNG suitable for the requested array type
default_rng(::Type{<:AnyGPUArray}) = error("Not implemented") # COV_EXCL_LINE

Random.seed!(rng::RNG) = (rng.seed = rand(Random.RandomDevice(), UInt32); rng.counter = 0; rng)
Random.seed!(rng::RNG, seed::Integer) = (rng.seed = seed % UInt32; rng.counter = 0; rng)

function advance_counter!(rng::RNG)
    new_counter = Int64(rng.counter) + 1
    overflow, remainder = fldmod(new_counter, typemax(UInt32))
    rng.seed += overflow % UInt32
    rng.counter = remainder % UInt32
end


## Specialized rand! kernels for common types
#
# Each Philox4x32 call produces 4 UInt32 outputs. Specialized kernels batch
# multiple values per work item to use all 4 outputs efficiently.
# Types that consume 1 UInt32 each get 4 values per call; types that need
# 2 UInt32s (UInt64/Float64) get 2 values per call; complex types that need
# 4 UInt32s get 1 value per call.

# Convert Philox UInt32 outputs to N values of type T
@inline philox_to_vals(::Type{Float16}, a1, a2, a3, a4) =
    (Float16(u01(Float32, a1)), Float16(u01(Float32, a2)),
     Float16(u01(Float32, a3)), Float16(u01(Float32, a4)))
@inline philox_to_vals(::Type{Float32}, a1, a2, a3, a4) =
    (u01(Float32, a1), u01(Float32, a2), u01(Float32, a3), u01(Float32, a4))
for T in (UInt8, UInt16, UInt32, Int8, Int16, Int32, Bool)
    @eval @inline philox_to_vals(::Type{$T}, a1, a2, a3, a4) =
        (a1 % $T, a2 % $T, a3 % $T, a4 % $T)
end
@inline philox_to_vals(::Type{Float64}, a1, a2, a3, a4) =
    (u01(Float64, UInt64(a1) | UInt64(a2) << 32),
     u01(Float64, UInt64(a3) | UInt64(a4) << 32))
for T in (UInt64, Int64)
    @eval @inline philox_to_vals(::Type{$T}, a1, a2, a3, a4) =
        ((UInt64(a1) | UInt64(a2) << 32) % $T,
         (UInt64(a3) | UInt64(a4) << 32) % $T)
end
@inline philox_to_vals(::Type{Complex{Float32}}, a1, a2, a3, a4) =
    (complex(u01(Float32, a1), u01(Float32, a2)),
     complex(u01(Float32, a3), u01(Float32, a4)))
@inline philox_to_vals(::Type{Complex{Float64}}, a1, a2, a3, a4) =
    (complex(u01(Float64, UInt64(a1) | UInt64(a2) << 32),
             u01(Float64, UInt64(a3) | UInt64(a4) << 32)),)

# Number of output values per Philox4x32 call
vals_per_call(::Type{T}) where T = sizeof(T) <= 4 ? 4 : sizeof(T) <= 8 ? 2 : 1
vals_per_call(::Type{Complex{T}}) where T = sizeof(T) <= 4 ? 2 : 1

# Batched kernel: N values per work item from one Philox call
@kernel function rand_batched_kernel!(@Const(seed), @Const(counter), A::AbstractArray{T}) where T
    gid = @index(Global, Linear)
    N = vals_per_call(T)
    idx = N * gid
    len = length(A)
    if idx <= len
        vals = philox_to_vals(T, philox4x32_10(
            (gid % UInt32, UInt32(0), counter, UInt32(0)),
            (seed, UInt32(0)))...)
        for j in 1:N
            @inbounds A[idx - N + j] = vals[j]
        end
    elseif idx - N < len
        vals = philox_to_vals(T, philox4x32_10(
            (gid % UInt32, UInt32(0), counter, UInt32(0)),
            (seed, UInt32(0)))...)
        for j in 1:min(N, len - idx + N)
            @inbounds A[idx - N + j] = vals[j]
        end
    end
end

# Types with specialized batched kernels
const BatchedRandTypes = Union{
    Float16, Float32, Float64, Complex{Float32}, Complex{Float64},
    Bool, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64}

function Random.rand!(rng::RNG, A::AnyGPUArray{T}) where T <: BatchedRandTypes
    isempty(A) && return A
    N = vals_per_call(T)
    rand_batched_kernel!(get_backend(A))(rng.seed, rng.counter, A;
                                         ndrange=cld(length(A), N))
    advance_counter!(rng)
    A
end


## Generic rand! fallback via ElementRNG
#
# For types without specialized kernels (Float16, integers, Bool, UInt128, etc.),
# an immutable ElementRNG wraps a Philox4x32 call as an AbstractRNG. Julia's Random
# stdlib then handles all type conversions automatically via `rand(rng, T)`.
#
# State (subcounter) lives in a stack-allocated Ref; the struct holds a pointer to
# it. This avoids mutable struct heap allocation on GPU.

struct ElementRNG <: AbstractRNG
    seed::UInt32
    counter::UInt32
    gid::UInt32
    subctr_ptr::Ptr{UInt32}
end

@inline Random.rng_native_52(::ElementRNG) = UInt64

@inline function Random.rand(rng::ElementRNG, ::Random.SamplerType{UInt64})
    sc = unsafe_load(rng.subctr_ptr) + UInt32(1)
    unsafe_store!(rng.subctr_ptr, sc)
    a1, a2, _, _ = philox4x32_10(
        (rng.gid, sc, rng.counter, UInt32(0)),
        (rng.seed, UInt32(0)))
    UInt64(a1) | UInt64(a2) << 32
end

@inline function Random.rand(rng::ElementRNG, ::Random.SamplerType{UInt128})
    UInt128(rand(rng, Random.SamplerType{UInt64}())) |
    UInt128(rand(rng, Random.SamplerType{UInt64}())) << 64
end

@inline Random.rand(rng::ElementRNG, ::Random.SamplerType{T}) where T <: Union{Bool,Base.BitInteger} =
    rand(rng, Random.SamplerType{UInt64}()) % T

# @inline overrides for Random stdlib entry points that aren't inlined by default.
# Without these, the dispatch chain stays opaque and the Ref won't be optimized.
@inline Random.rand(rng::ElementRNG, ::Type{T}) where {T} =
    rand(rng, Random.Sampler(typeof(rng), T, Val(1)))
@inline Random.rand(rng::ElementRNG, ::Random.SamplerType{Complex{T}}) where {T<:Real} =
    complex(rand(rng, T), rand(rng, T))

@kernel function rand_generic_kernel!(@Const(seed), @Const(counter), A::AbstractArray{T}) where T
    gid = @index(Global, Linear)
    if gid <= length(A)
        subctr = Ref{UInt32}(0)
        rng = ElementRNG(seed, counter, gid % UInt32,
                          Base.unsafe_convert(Ptr{UInt32}, subctr))
        @inbounds A[gid] = rand(rng, T)
    end
end

function Random.rand!(rng::RNG, A::AnyGPUArray{T}) where T <: Number
    isempty(A) && return A
    rand_generic_kernel!(get_backend(A))(rng.seed, rng.counter, A; ndrange=length(A))
    advance_counter!(rng)
    A
end


## randn! kernels
#
# Unlike rand!, randn! uses specialized batched kernels instead of the generic
# ElementRNG approach. This is because:
#
# 1. Random's default `randn` uses the Ziggurat algorithm with table lookups
#    (`ki`, `wi`, `fi` arrays) that aren't accessible on GPU without device
#    overlays (which are backend-specific, not available in KernelAbstractions).
#
# 2. The generic fallback `randn(rng, T::AbstractFloat)` uses Marsaglia's polar
#    Box-Muller variant with a rejection loop (`while true ... 0 < s < 1`),
#    which causes severe warp divergence on GPU (~2x slower).
#
# 3. Direct (non-rejection) Box-Muller naturally produces value *pairs* from
#    sincospi. Batching these (4 Float32 / 2 Float64 per Philox call) is key
#    to staying memory-bandwidth-bound; a 1-per-work-item generic kernel
#    discards half the output.
#
# Box-Muller: each Philox call produces 2 normal values from 2 uniform values.
# For <=32-bit floats: 4 values per call (2 Box-Muller pairs from 4 UInt32).
# For >32-bit floats: 2 values per call (1 Box-Muller pair from 2 UInt64).

# Box-Muller for complex: sqrt(-log(U)) not sqrt(-2*log(U)),
# so each component has variance 1/2
@inline function boxmuller_complex(::Type{T}, u1::T, u2::T) where T
    r = sqrt(FastMath.neg_float_fast(FastMath.log_fast(u1)))
    s, c = sincospi(2 * u2)
    complex(r * s, r * c)
end

@kernel function randn_small_kernel!(@Const(seed), @Const(counter), A::AbstractArray{T}) where T
    gid = @index(Global, Linear)
    idx = 4 * gid
    len = length(A)
    if idx <= len
        a1, a2, a3, a4 = philox4x32_10(
            (gid % UInt32, UInt32(0), counter, UInt32(0)),
            (seed, UInt32(0)))
        n1, n2 = boxmuller(T, T(u01(Float32, a1)), T(u01(Float32, a2)))
        n3, n4 = boxmuller(T, T(u01(Float32, a3)), T(u01(Float32, a4)))
        @inbounds A[idx - 3] = n1
        @inbounds A[idx - 2] = n2
        @inbounds A[idx - 1] = n3
        @inbounds A[idx]     = n4
    elseif idx - 3 <= len
        a1, a2, a3, a4 = philox4x32_10(
            (gid % UInt32, UInt32(0), counter, UInt32(0)),
            (seed, UInt32(0)))
        n1, n2 = boxmuller(T, T(u01(Float32, a1)), T(u01(Float32, a2)))
        n3, n4 = boxmuller(T, T(u01(Float32, a3)), T(u01(Float32, a4)))
        vals = (n1, n2, n3, n4)
        for j in 1:min(4, len - idx + 4)
            @inbounds A[idx - 4 + j] = vals[j]
        end
    end
end

@kernel function randn_large_kernel!(@Const(seed), @Const(counter), A::AbstractArray{T}) where T
    gid = @index(Global, Linear)
    idx = 2 * gid
    len = length(A)
    if idx <= len
        a1, a2, a3, a4 = philox4x32_10(
            (gid % UInt32, UInt32(0), counter, UInt32(0)),
            (seed, UInt32(0)))
        n1, n2 = boxmuller(T,
            T(u01(Float64, UInt64(a1) | UInt64(a2) << 32)),
            T(u01(Float64, UInt64(a3) | UInt64(a4) << 32)))
        @inbounds A[idx - 1] = n1
        @inbounds A[idx]     = n2
    elseif idx - 1 <= len
        a1, a2, a3, a4 = philox4x32_10(
            (gid % UInt32, UInt32(0), counter, UInt32(0)),
            (seed, UInt32(0)))
        n1, _ = boxmuller(T,
            T(u01(Float64, UInt64(a1) | UInt64(a2) << 32)),
            T(u01(Float64, UInt64(a3) | UInt64(a4) << 32)))
        @inbounds A[len] = n1
    end
end

@kernel function randn_complex_small_kernel!(@Const(seed), @Const(counter), A::AbstractArray{Complex{T}}) where T
    gid = @index(Global, Linear)
    idx = 2 * gid
    len = length(A)
    if idx <= len
        a1, a2, a3, a4 = philox4x32_10(
            (gid % UInt32, UInt32(0), counter, UInt32(0)),
            (seed, UInt32(0)))
        @inbounds A[idx - 1] = boxmuller_complex(T, T(u01(Float32, a1)), T(u01(Float32, a2)))
        @inbounds A[idx]     = boxmuller_complex(T, T(u01(Float32, a3)), T(u01(Float32, a4)))
    elseif idx - 1 <= len
        a1, a2, _, _ = philox4x32_10(
            (gid % UInt32, UInt32(0), counter, UInt32(0)),
            (seed, UInt32(0)))
        @inbounds A[len] = boxmuller_complex(T, T(u01(Float32, a1)), T(u01(Float32, a2)))
    end
end

@kernel function randn_complex_large_kernel!(@Const(seed), @Const(counter), A::AbstractArray{Complex{T}}) where T
    gid = @index(Global, Linear)
    if gid <= length(A)
        a1, a2, a3, a4 = philox4x32_10(
            (gid % UInt32, UInt32(0), counter, UInt32(0)),
            (seed, UInt32(0)))
        U1 = T(u01(Float64, UInt64(a1) | UInt64(a2) << 32))
        U2 = T(u01(Float64, UInt64(a3) | UInt64(a4) << 32))
        @inbounds A[gid] = boxmuller_complex(T, U1, U2)
    end
end

function Random.randn!(rng::RNG, A::AnyGPUArray{T}) where T <: AbstractFloat
    isempty(A) && return A
    if sizeof(T) <= 4
        randn_small_kernel!(get_backend(A))(rng.seed, rng.counter, A; ndrange=cld(length(A), 4))
    else
        randn_large_kernel!(get_backend(A))(rng.seed, rng.counter, A; ndrange=cld(length(A), 2))
    end
    advance_counter!(rng)
    A
end

function Random.randn!(rng::RNG, A::AnyGPUArray{Complex{T}}) where T <: AbstractFloat
    isempty(A) && return A
    if sizeof(T) <= 4
        randn_complex_small_kernel!(get_backend(A))(rng.seed, rng.counter, A; ndrange=cld(length(A), 2))
    else
        randn_complex_large_kernel!(get_backend(A))(rng.seed, rng.counter, A; ndrange=length(A))
    end
    advance_counter!(rng)
    A
end

# allow use of CPU RNGs without scalar iteration
Random.rand!(rng::AbstractRNG, A::AnyGPUArray) =
    copyto!(A, rand(rng, eltype(A), size(A)...))
Random.randn!(rng::AbstractRNG, A::AnyGPUArray) =
    copyto!(A, randn(rng, eltype(A), size(A)...))
