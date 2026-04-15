# integration with Random stdlib

## Philox4x32-10 counter-based RNG

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


## Float conversions: unsigned integer → uniform float, strictly positive
## (Float32 can round up to exactly 1.0; Float64 stays strictly below 1.0.)

@inline function u01(::Type{Float32}, u::UInt32)
    fma(Float32(u), Float32(2)^(-32), Float32(2)^(-33))
end

@inline function u01(::Type{Float64}, u::UInt64)
    # Bit-pattern construction avoids the expensive Float64(::UInt64) conversion
    # and fma on consumer GPUs (where FP64 throughput is as low as 1:64). The
    # low bit of the mantissa is forced set so the result is strictly in (0, 1),
    # which is required by Box-Muller's log(u).
    reinterpret(Float64, ((u >> 12) | 0x1) | 0x3ff0000000000000) - 1.0
end


## Box-Muller transform using @fastmath log + sincospi

# Each backend provides optimized intrinsics via @device_override:
# CUDA: __nv_fast_logf, __nv_sincospif, etc.

using Base.FastMath

@inline function boxmuller(::Type{T}, u1::T, u2::T) where T <: AbstractFloat
    r = sqrt(T(-2) * FastMath.log_fast(u1))
    s, c = sincospi(2 * u2)
    (r * s, r * c)
end

# For complex normals each component has variance 1/2, so the radius is
# sqrt(-log(U)) rather than sqrt(-2*log(U)).
@inline function boxmuller(::Type{Complex{T}}, u1::T, u2::T) where T <: AbstractFloat
    r = sqrt(FastMath.neg_float_fast(FastMath.log_fast(u1)))
    s, c = sincospi(2 * u2)
    complex(r * s, r * c)
end


## RNG type

mutable struct RNG <: AbstractRNG
    seed::UInt64
    counter::UInt32
end

RNG() = RNG(rand(Random.RandomDevice(), UInt64), UInt32(0))
RNG(seed::Integer) = RNG(seed % UInt64, UInt32(0))

# return an instance of GPUArrays.RNG suitable for the requested array type
default_rng(::Type{<:AnyGPUArray}) = error("Not implemented") # COV_EXCL_LINE

Random.seed!(rng::RNG) = (rng.seed = rand(Random.RandomDevice(), UInt64); rng.counter = 0; rng)
Random.seed!(rng::RNG, seed::Integer) = (rng.seed = seed % UInt64; rng.counter = 0; rng)

function advance_counter!(rng::RNG)
    rng.counter += one(UInt32)
    rng.counter == 0 && (rng.seed += one(UInt64))
    rng
end

# Split the 64-bit seed into the two 32-bit lanes of the Philox key.
@inline philox_key(seed::UInt64) = (seed % UInt32, (seed >> 32) % UInt32)


## Specialized rand! kernels for common types

# Each Philox4x32 call produces 4 UInt32 outputs. Specialized kernels batch
# multiple values per work item to use all 4 outputs efficiently:
# - ≤4-byte types (including Float16): 4 values per call
# - 8-byte types (Int64/UInt64/Float64/Complex{Float32}): 2 values per call
# - 16-byte types (Int128/UInt128/Complex{Float64}):     1 value per call

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
for T in (UInt128, Int128)
    @eval @inline philox_to_vals(::Type{$T}, a1, a2, a3, a4) =
        ((UInt128(a1) | UInt128(a2) << 32 | UInt128(a3) << 64 | UInt128(a4) << 96) % $T,)
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
            philox_key(seed))...)
        for j in 1:N
            @inbounds A[idx - N + j] = vals[j]
        end
    elseif idx - N < len
        vals = philox_to_vals(T, philox4x32_10(
            (gid % UInt32, UInt32(0), counter, UInt32(0)),
            philox_key(seed))...)
        for j in 1:min(N, len - idx + N)
            @inbounds A[idx - N + j] = vals[j]
        end
    end
end

# Types with specialized batched kernels
const BatchedRandTypes = Union{
    Float16, Float32, Float64, Complex{Float32}, Complex{Float64},
    Bool, Int8, Int16, Int32, Int64, Int128,
    UInt8, UInt16, UInt32, UInt64, UInt128}

function Random.rand!(rng::RNG, A::AnyGPUArray{T}) where T <: BatchedRandTypes
    isempty(A) && return A
    N = vals_per_call(T)
    rand_batched_kernel!(get_backend(A))(rng.seed, rng.counter, A;
                                         ndrange=cld(length(A), N))
    advance_counter!(rng)
    A
end


## ElementRNG: a device-side AbstractRNG for generic rand/randn fallbacks

# For element types without a specialized batched kernel (BigFloat,
# FixedPointNumbers, user-defined types, ...), we create a per-work-item
# AbstractRNG and delegate to Random stdlib's `rand(rng, T)` / `randn(rng, T)`.
#
# The struct is immutable (mutable structs get heap-allocated on GPU); the
# per-work-item subcounter state lives in a stack-allocated Ref that the
# struct holds a pointer to.

struct ElementRNG <: AbstractRNG
    seed::UInt64
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
        philox_key(rng.seed))
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


## Generic rand! fallback via ElementRNG

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


## Specialized randn! kernel for common float types

# Float16/32/64 and their complex variants go through a batched kernel mirroring
# rand_batched_kernel!, rather than the generic ElementRNG path further down:
#
# 1. Random's `randn(rng, ::Float{16,32,64})` uses the Ziggurat algorithm with
#    global table lookups (`ki`, `wi`, `fi`), which aren't device-accessible
#    from a KernelAbstractions kernel.
#
# 2. The polar Box-Muller fallback reachable for other AbstractFloat types has
#    a rejection loop (`while true ... 0 < s < 1`), causing warp divergence
#    (~2x slower).
#
# 3. Direct (non-rejection) Box-Muller naturally produces value *pairs* from
#    sincospi. Batching these (4 Float32 / 2 Float64 per Philox call) keeps us
#    memory-bandwidth-bound; a 1-per-work-item kernel would discard half the
#    output.

# Convert Philox UInt32 outputs to N normally-distributed values of type T.
for T in (Float16, Float32)
    @eval @inline function philox_to_normals(::Type{$T}, a1, a2, a3, a4)
        n1, n2 = boxmuller($T, $T(u01(Float32, a1)), $T(u01(Float32, a2)))
        n3, n4 = boxmuller($T, $T(u01(Float32, a3)), $T(u01(Float32, a4)))
        (n1, n2, n3, n4)
    end
end
@inline function philox_to_normals(::Type{Float64}, a1, a2, a3, a4)
    boxmuller(Float64,
        u01(Float64, UInt64(a1) | UInt64(a2) << 32),
        u01(Float64, UInt64(a3) | UInt64(a4) << 32))
end
@inline function philox_to_normals(::Type{Complex{Float32}}, a1, a2, a3, a4)
    (boxmuller(Complex{Float32}, u01(Float32, a1), u01(Float32, a2)),
     boxmuller(Complex{Float32}, u01(Float32, a3), u01(Float32, a4)))
end
@inline function philox_to_normals(::Type{Complex{Float64}}, a1, a2, a3, a4)
    (boxmuller(Complex{Float64},
        u01(Float64, UInt64(a1) | UInt64(a2) << 32),
        u01(Float64, UInt64(a3) | UInt64(a4) << 32)),)
end

@kernel function randn_batched_kernel!(@Const(seed), @Const(counter), A::AbstractArray{T}) where T
    gid = @index(Global, Linear)
    N = vals_per_call(T)
    idx = N * gid
    len = length(A)
    if idx <= len
        vals = philox_to_normals(T, philox4x32_10(
            (gid % UInt32, UInt32(0), counter, UInt32(0)),
            philox_key(seed))...)
        for j in 1:N
            @inbounds A[idx - N + j] = vals[j]
        end
    elseif idx - N < len
        vals = philox_to_normals(T, philox4x32_10(
            (gid % UInt32, UInt32(0), counter, UInt32(0)),
            philox_key(seed))...)
        for j in 1:min(N, len - idx + N)
            @inbounds A[idx - N + j] = vals[j]
        end
    end
end

const BatchedRandnTypes = Union{Float16, Float32, Float64,
                                Complex{Float32}, Complex{Float64}}

function Random.randn!(rng::RNG, A::AnyGPUArray{T}) where T <: BatchedRandnTypes
    isempty(A) && return A
    N = vals_per_call(T)
    randn_batched_kernel!(get_backend(A))(rng.seed, rng.counter, A;
                                          ndrange=cld(length(A), N))
    advance_counter!(rng)
    A
end


## Generic randn! fallback via ElementRNG
#
# For AbstractFloats outside BatchedRandnTypes (BFloat16, user-defined float
# types, etc.) we route through Random's `randn(rng, T)`. The reachable methods
# from there are:
#
# - `randn(rng, ::BitFloatType)` (Float16/32/64) — ziggurat, uses global `wi`
#   /`ki`/`fi` tables that aren't device-accessible. Never reached here because
#   those types dispatch to the batched kernel above.
# - `randn(rng, ::Type{Complex{T}})` — recurses into `randn(rng, T)`.
# - `randn(rng, ::Type{T}) where T<:AbstractFloat` — Marsaglia polar Box-Muller
#   rejection loop. GPU-safe (only calls `rand(rng, T)`) but warp-divergent.

@kernel function randn_generic_kernel!(@Const(seed), @Const(counter), A::AbstractArray{T}) where T
    gid = @index(Global, Linear)
    if gid <= length(A)
        subctr = Ref{UInt32}(0)
        rng = ElementRNG(seed, counter, gid % UInt32,
                          Base.unsafe_convert(Ptr{UInt32}, subctr))
        @inbounds A[gid] = randn(rng, T)
    end
end

function Random.randn!(rng::RNG, A::AnyGPUArray{T}) where T <: Union{AbstractFloat,
                                                                     Complex{<:AbstractFloat}}
    isempty(A) && return A
    randn_generic_kernel!(get_backend(A))(rng.seed, rng.counter, A; ndrange=length(A))
    advance_counter!(rng)
    A
end


