# integration with Random stdlib

## Philox4x32-10 counter-based RNG

# Stateless: (counter, key) → 4 UInt32 outputs. Each unique counter gives independent
# random values with no shared memory or global state needed.
#
# Reference: Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3" (2011)
# Based on: PhiloxRNG.jl by Nathan Zimmerberg

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

@inline function philox4x32_10(ctr0::UInt64, ctr1::UInt64, key::UInt64)::NTuple{4, UInt32}
    philox4x32_10((ctr0%UInt32, (ctr0>>32)%UInt32, ctr1%UInt32, (ctr1>>32)%UInt32), (key%UInt32, (key>>32)%UInt32))
end

## Float conversions: unsigned integer → uniform float, strictly positive
## (can round up to exactly 1.0)

"""
    u01(F, u::Union{UInt32, UInt64})::F

Convert an unsigned integer to a float of type `F` uniformly distributed in (0, 1].

Ported from [Random123 uniform.hpp](https://github.com/DEShawResearch/random123/blob/v1.14.0/include/Random123/uniform.hpp#L175).
"""
@inline function u01(::Type{F}, u::UInt32)::F where F
    fma(F(u), F(2)^Int32(-32), F(2)^Int32(-33))
end

@inline function u01(::Type{F}, u::UInt64)::F where F
    fma(F(u), F(2)^Int32(-64), F(2)^Int32(-65))
end

## Fast sincospi for Box-Muller
#
# Base.sincospi(::Float32) widens to Float64 internally (via `sinpi_kernel_wide`,
# see base/special/trig.jl:744), which breaks Metal — it has no Float64 support
# at all — and wastes cycles on other backends that must emulate FP64.
#
# Vendored from PhiloxRNG.jl (MIT): a minimax polynomial that stays entirely
# in Float32 and consumes the Philox UInt32 output directly. Bottom 3 bits
# select one of 8 octants (swap/negate), upper 29 bits give the reduced
# argument (+0.5-biased so y ≠ 0).
#
# Vendored contents of:
# https://github.com/medyan-dev/PhiloxRNG.jl/blob/v1.1.1/src/fastsincospi.jl
# With minor style changes.

# ============================================================
# Fast sincospi for Box-Muller
#
# Computes (sin(θ), cos(θ)) from a uniform UInt32 (or UInt64),
# placing 2^N points uniformly around the unit circle with
# no point landing exactly on an axis.
#
# The bottom 3 bits of u select one of 8 octants (π/4 each).
# The upper bits give the reduced argument y ∈ (0, 0.25),
# with a +0.5 bias to avoid y = 0. The polynomials evaluate
# sin(πy) and cos(πy) — with π baked into the coefficients —
# then each octant bit directly controls one operation:
#   bit 0 → swap sin/cos
#   bit 1 → negate sin
#   bit 2 → negate cos
#
# The octants are not in geometric order, but the 2^N points
# are uniformly distributed around the unit circle regardless.
# ============================================================

# --- Float32 minimax coefficients for sin(πy)/y and cos(πy) in y² ---
#
# 4-term (degree 3 in y²) minimax via Remez algorithm on [0, 0.0625].
const SP_F32 = (3.1415927f0, -5.167708f0, 2.5497673f0, -0.58907866f0)
const CP_F32 = (1.0f0, -4.934788f0, 4.057578f0, -1.3061346f0)

@inline function fast_sincospi(::Type{Float32}, u::Union{UInt32, UInt64})
    oct = (u % Int32) & Int32(7)
    y = fma(Float32(u & ~oftype(u, 7)), Float32(2)^Int32(-(sizeof(u)*8+2)), Float32(2)^Int32(-(sizeof(u)*8)))

    sp = y * evalpoly(y * y, SP_F32)
    cp = evalpoly(y * y, CP_F32)

    swap    = !iszero(oct & Int32(1))
    sin_neg = !iszero(oct & Int32(2))
    cos_neg = !iszero(oct & Int32(4))

    s_raw = ifelse(swap, cp, sp)
    c_raw = ifelse(swap, sp, cp)
    sin_val = ifelse(sin_neg, -s_raw, s_raw)
    cos_val = ifelse(cos_neg, -c_raw, c_raw)
    (sin_val, cos_val)
end

# ============================================================
# Float64 / UInt64 version
#
# Same structure as Float32: bottom 3 bits → octant, upper
# 61 bits → reduced argument, +0.5 bias, direct bit mapping.
# ============================================================

const SP_F64 = (3.141592653589793, -5.167712780049954, 2.5501640398733785,
               -0.5992645289398095, 0.08214586918507949, -0.007370021659123395,
               0.0004615322405282014)
const CP_F64 = (1.0, -4.934802200544605, 4.0587121263978485,
               -1.3352627670374702, 0.23533054723811608, -0.025804938901032953,
               0.0019068114005246046)

@inline function fast_sincospi(::Type{Float64}, u::Union{UInt32, UInt64})
    oct = (u % Int32) & Int32(7)
    y = fma(Float64(u & ~oftype(u, 7)), Float64(2)^Int32(-(sizeof(u)*8+2)), Float64(2)^Int32(-(sizeof(u)*8)))

    sp = y * evalpoly(y * y, SP_F64)
    cp = evalpoly(y * y, CP_F64)

    swap    = !iszero(oct & Int32(1))
    sin_neg = !iszero(oct & Int32(2))
    cos_neg = !iszero(oct & Int32(4))

    s_raw = ifelse(swap, cp, sp)
    c_raw = ifelse(swap, sp, cp)
    sin_val = ifelse(sin_neg, -s_raw, s_raw)
    cos_val = ifelse(cos_neg, -c_raw, c_raw)
    (sin_val, cos_val)
end

# End of vendored https://github.com/medyan-dev/PhiloxRNG.jl/blob/v1.1.1/src/fastsincospi.jl

## Fast log for Box-Muller
#
# Base.log(::Float32) widens to Float64 internally (see base/special/log.jl:242
# `Float64(2f0*f)/(2.0+f)`), same Metal / FP64-emulation problem as sincospi.
#
# Vendored contents of:
# https://github.com/medyan-dev/PhiloxRNG.jl/blob/v1.1.1/src/fastlog.jl
# With minor style changes.

# Core log algorithm (polynomial coefficients, ln2 splitting, and reconstruction)
# adapted from fdlibm's e_log.c / e_logf.c (Sun Microsystems, 1993).
# See: https://github.com/JuliaMath/openlibm/blob/v0.8.7/src/e_log.c
#      https://github.com/JuliaMath/openlibm/blob/v0.8.7/src/e_logf.c

const SQRT_HALF_I32 = reinterpret(Int32, Float32(sqrt(0.5)))
const LOG_POLY_F32 = (0.6666666f0, 0.40000972f0, 0.28498787f0, 0.24279079f0)
const LN2_HI_F32 = 0.6931381f0
const LN2_LO_F32 = 9.058001f-6

@inline function fast_log(::Type{Float32}, u::Union{UInt32, UInt64})
    x = u01(Float32, u)

    # Goal: find k and f such that
    # x = 2^k * (1+f)
    # where sqrt(2)/2 ≤ (1+f) < sqrt(2)
    # if k is zero
    # we calculate f by -u01(Float32, ~u) which is more accurate for x near 1

    # Float32 has 23 fractional bits.
    # x is ordered by value in Int32 space.
    # Starting from x=1, k starts at 0, then ix becomes negative at x = prevfloat(sqrt(0.5f0))
    # making k = -1. For each power of 2 scale in x,
    # k changes by one, because we shift out the 23 fraction bits.
    ix = reinterpret(Int32, x) - SQRT_HALF_I32
    k = ix >> Int32(23)

    # `f_plus_one_std` will have the same fraction bits as `x`
    # because `- SQRT_HALF_I32` and `+ SQRT_HALF_I32` cancel out in the low 23 bits.
    # `& Int32(0x007fffff)` clears the exponent and sign fields.
    # `f_plus_one_std` must either have an exponent of -1 or 0.
    # If x's fractional bits are less than the fractional bits of SQRT_HALF_I32
    # the `- SQRT_HALF_I32` borrows a 2^23 from the exponent field of x,
    # which then shows up as an extra 2^23 in the low 23 bits after masking.
    # When adding SQRT_HALF_I32 this extra 2^23 propagates up and
    # bumps the exponent from -1 to 0.
    f_plus_one_std = reinterpret(Float32, (ix & Int32(0x007fffff)) + SQRT_HALF_I32)
    f_std = f_plus_one_std - 1.0f0

    f_comp = -u01(Float32, ~u)
    f = ifelse(k == Int32(0), f_comp, f_std)

    # Goal: get log(1+f) via a polynomial approx.
    # Let s = f/(2+f), z = s², and log_poly(z) ≈ evalpoly(z, LOG_POLY_F32)
    # log(1+f) = 2s + s³*log_poly(s²)
    # R = s²*log_poly(s²)
    # log(1+f) = f - f²/2 + s*(f²/2 + R)
    s = f / (2.0f0 + f)
    z = s * s
    R = z * evalpoly(z, LOG_POLY_F32)
    hfsq = 0.5f0 * f * f

    # log(x) = k*log(2) + log(1+f)
    k_f32 = Float32(k)
    # Simpler version, but fails the mean test by 2E-9
    # fma(k_f32, 0.6931472f0 #= log(2) =#, fma(s, R-f, f))
    # log(2) = LN2_HI_F32 + LN2_LO_F32
    fma(k_f32, LN2_HI_F32,
        f - (hfsq - fma(s, (hfsq + R), k_f32 * LN2_LO_F32))
    )
end

const SQRT_HALF_I64 = reinterpret(Int64, sqrt(0.5))
const LOG_POLY_F64 = (
    6.666666666666735130e-01,
    3.999999999940941908e-01,
    2.857142874366239149e-01,
    2.222219843214978396e-01,
    1.818357216161805012e-01,
    1.531383769920937332e-01,
    1.479819860511658591e-01,
)
const LN2_HI_F64 = 6.93147180369123816490e-01
const LN2_LO_F64 = 1.90821492927058770002e-10

@inline function fast_log(::Type{Float64}, u::Union{UInt32, UInt64})
    # See Float32 version for commentary
    x = u01(Float64, u)

    ix = reinterpret(Int64, x) - SQRT_HALF_I64
    k = ix >> Int64(52)
    f_std = reinterpret(Float64, (ix & Int64(0x000fffffffffffff)) + SQRT_HALF_I64) - 1.0

    f_comp = -u01(Float64, ~u)
    f = ifelse(k == Int64(0), f_comp, f_std)

    s = f / (2.0 + f)
    z = s * s
    R = z * evalpoly(z, LOG_POLY_F64)
    hfsq = 0.5 * f * f

    # log(x) = k*ln2 + log(1+f)
    k_f64 = Float64(k)
    fma(k_f64, LN2_HI_F64,
        f - (hfsq - fma(s, (hfsq + R), k_f64 * LN2_LO_F64))
    )
end

# End of vendored https://github.com/medyan-dev/PhiloxRNG.jl/blob/v1.1.1/src/fastlog.jl


## Box-Muller transform

# ≤32-bit float output: both log and sincospi go through the Float32
# polynomials above.
# Using Base.sqrt_llvm to avoid the DomainError check.
@inline function boxmuller(::Type{F}, u1::UInt32, u2::UInt32) where F <: Union{Float16,Float32}
    r = Base.sqrt_llvm(-2f0 * fast_log(Float32, u2))
    s, c = fast_sincospi(Float32, u1)
    (F(r * s), F(r * c))
end

@inline function boxmuller(::Type{Float64}, u1::UInt64, u2::UInt64)
    r = Base.sqrt_llvm(-2.0 * fast_log(Float64, u2))
    s, c = fast_sincospi(Float64, u1)
    (r * s, r * c)
end

# For complex normals each component has variance 1/2, so the radius is
# sqrt(-log(U)) rather than sqrt(-2·log(U)).
@inline function boxmuller(::Type{Complex{F}}, u1::UInt32, u2::UInt32) where F <: Union{Float16,Float32}
    r = Base.sqrt_llvm(-fast_log(Float32, u2))
    s, c = fast_sincospi(Float32, u1)
    complex(F(r * s), F(r * c))
end
@inline function boxmuller(::Type{Complex{Float64}}, u1::UInt64, u2::UInt64)
    r = Base.sqrt_llvm(-fast_log(Float64, u2))
    s, c = fast_sincospi(Float64, u1)
    complex(r * s, r * c)
end


## RNG type
#
# Parameterized on the GPU array type `AT` (e.g. `CuArray`, `ROCArray`, `JLArray`).
# `AT` is only consulted by the *out-of-place* `rand`/`randn` constructors and by
# the CPU-array fill fallback — the in-place `rand!`/`randn!` GPU paths dispatch
# on the destination's KernelAbstractions backend and ignore `AT`.

mutable struct RNG{AT} <: AbstractRNG
    seed::UInt64
    counter::UInt64
end

RNG{AT}() where {AT} = RNG{AT}(rand(Random.RandomDevice(), UInt64), rand(Random.RandomDevice(), UInt64))
RNG{AT}(seed::Integer) where {AT} = RNG{AT}(seed % UInt64, UInt64(0))

Random.seed!(rng::RNG) = (rng.seed = rand(Random.RandomDevice(), UInt64); rng.counter = rand(Random.RandomDevice(), UInt64); rng)
Random.seed!(rng::RNG, seed::Integer) = (rng.seed = seed % UInt64; rng.counter = UInt64(0); rng)

function advance_counter!(rng::RNG)
    rng.counter += one(UInt64)
    rng
end


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
@kernel function rand_batched_kernel!(seed::UInt64, counter::UInt64, A::AbstractArray{T}) where T
    gid = @index(Global, Linear)
    N = vals_per_call(T)
    idx = N * gid
    len = length(A)
    if idx <= len
        vals = philox_to_vals(T, philox4x32_10(gid % UInt64, counter, seed)...)
        for j in 1:N
            @inbounds A[idx - N + j] = vals[j]
        end
    elseif idx - N < len
        vals = philox_to_vals(T, philox4x32_10(gid % UInt64, counter, seed)...)
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
    counter::UInt64
    nthreads::UInt64
    ctr0_ptr::Ptr{UInt64}
end

@inline Random.rng_native_52(::ElementRNG) = UInt64

@inline function Random.rand(rng::ElementRNG, ::Random.SamplerType{UInt64})
    sc = unsafe_load(rng.ctr0_ptr) + rng.nthreads
    unsafe_store!(rng.ctr0_ptr, sc)
    a1, a2, _, _ = philox4x32_10(sc, rng.counter, rng.seed)
    UInt64(a1) | UInt64(a2) << 32
end

@inline function Random.rand(rng::ElementRNG, ::Random.SamplerType{UInt128})
    UInt128(rand(rng, Random.SamplerType{UInt64}())) |
    UInt128(rand(rng, Random.SamplerType{UInt64}())) << 64
end

@inline Random.rand(rng::ElementRNG, ::Random.SamplerType{T}) where T <: Union{Bool,Base.BitInteger} =
    rand(rng, Random.SamplerType{UInt64}()) % T


## Generic rand! fallback via ElementRNG

@kernel function rand_generic_kernel!(seed::UInt64, counter::UInt64, A::AbstractArray{T}) where T
    gid = @index(Global, Linear)
    len_A = length(A)
    if gid <= len_A
        subctr = Ref{UInt64}(gid%UInt64)
        GC.@preserve subctr begin
            rng = ElementRNG(seed, counter, len_A % UInt64,
                            Base.unsafe_convert(Ptr{UInt64}, subctr))
            @inbounds A[gid] = rand(rng, T)
        end
    end
end

function Random.rand!(rng::RNG, A::AnyGPUArray{T}) where T
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
# ≤32-bit float targets pass UInt32s to boxmuller directly. 64-bit
# targets use UInt64s for finer sampling.
for T in (Float16, Float32)
    @eval @inline function philox_to_normals(::Type{$T}, a1, a2, a3, a4)
        n1, n2 = boxmuller($T, a1, a2)
        n3, n4 = boxmuller($T, a3, a4)
        (n1, n2, n3, n4)
    end
end
@inline function philox_to_normals(::Type{Float64}, a1, a2, a3, a4)
    boxmuller(Float64,
        UInt64(a1) | UInt64(a2) << 32,
        UInt64(a3) | UInt64(a4) << 32)
end
@inline philox_to_normals(::Type{Complex{Float32}}, a1, a2, a3, a4) =
    (boxmuller(Complex{Float32}, a1, a2),
     boxmuller(Complex{Float32}, a3, a4))
@inline philox_to_normals(::Type{Complex{Float64}}, a1, a2, a3, a4) =
    (boxmuller(Complex{Float64},
        UInt64(a1) | UInt64(a2) << 32,
        UInt64(a3) | UInt64(a4) << 32),)

@kernel function randn_batched_kernel!(seed::UInt64, counter::UInt64, A::AbstractArray{T}) where T
    gid = @index(Global, Linear)
    N = vals_per_call(T)
    idx = N * gid
    len = length(A)
    if idx <= len
        vals = philox_to_normals(T, philox4x32_10(gid % UInt64, counter, seed)...)
        for j in 1:N
            @inbounds A[idx - N + j] = vals[j]
        end
    elseif idx - N < len
        vals = philox_to_normals(T, philox4x32_10(gid % UInt64, counter, seed)...)
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
#   /`ki`/`fi` tables that aren't device-accessible. Overridden below to use
#   our Box-Muller directly. The direct dispatch for these types goes through
#   the batched kernel above, but `randn(rng, Complex{Float16})` recurses into
#   `randn(rng, Float16)` which hits this path.
# - `randn(rng, ::Type{Complex{T}})` — recurses into `randn(rng, T)`.
# - `randn(rng, ::Type{T}) where T<:AbstractFloat` — Marsaglia polar Box-Muller
#   rejection loop. GPU-safe (only calls `rand(rng, T)`) but warp-divergent.

# Bypass Base's ziggurat-based randn(rng, Float{16,32,64}) — its `wi`/`ki`/`fi`
# tables aren't device-accessible, and on Metal the Float64 tables can't even
# be loaded. Reached via Base's Complex recursion when the element type is
# e.g. Complex{Float16}.
@inline Random.randn(rng::ElementRNG, ::Type{Float16}) =
    first(boxmuller(Float16, rand(rng, UInt32), rand(rng, UInt32)))
@inline Random.randn(rng::ElementRNG, ::Type{Float32}) =
    first(boxmuller(Float32, rand(rng, UInt32), rand(rng, UInt32)))
@inline Random.randn(rng::ElementRNG, ::Type{Float64}) =
    first(boxmuller(Float64, rand(rng, UInt64), rand(rng, UInt64)))

@kernel function randn_generic_kernel!(seed::UInt64, counter::UInt64, A::AbstractArray{T}) where T
    gid = @index(Global, Linear)
    len_A = length(A)
    if gid <= len_A
        subctr = Ref{UInt64}(gid%UInt64)
        GC.@preserve subctr begin
            rng = ElementRNG(seed, counter, len_A % UInt64,
                            Base.unsafe_convert(Ptr{UInt64}, subctr))
            @inbounds A[gid] = randn(rng, T)
        end
    end
end

function Random.randn!(rng::RNG, A::AnyGPUArray{T}) where T <: Union{AbstractFloat,
                                                                     Complex{<:AbstractFloat}}
    isempty(A) && return A
    randn_generic_kernel!(get_backend(A))(rng.seed, rng.counter, A; ndrange=length(A))
    advance_counter!(rng)
    A
end


## Non-GPU array fallback: generate on AT, copyto! the destination.
#
# Without this, `rand!(rng::RNG{CuArray}, ::Vector)` would hit Random's stdlib
# scalar path and silently iterate the GPU rng one element at a time.

function Random.rand!(rng::RNG{AT}, A::AbstractArray{T}) where {AT, T}
    isempty(A) && return A
    B = similar(AT{T}, size(A))
    Random.rand!(rng, B)
    copyto!(A, B)
end
function Random.randn!(rng::RNG{AT}, A::AbstractArray{T}) where {AT, T<:Union{AbstractFloat,
                                                                             Complex{<:AbstractFloat}}}
    isempty(A) && return A
    B = similar(AT{T}, size(A))
    Random.randn!(rng, B)
    copyto!(A, B)
end


## CPU RNG → GPU array: generate on CPU, copyto! the destination.
#
# Without this, `rand!(::Random.TaskLocalRNG, ::CuArray)` hits Random's stdlib
# scalar path and iterates the GPU array one element at a time.

Random.rand!(rng::AbstractRNG, A::AnyGPUArray) =
    copyto!(A, rand(rng, eltype(A), size(A)...))
Random.randn!(rng::AbstractRNG, A::AnyGPUArray{T}) where {T<:Union{AbstractFloat,
                                                                   Complex{<:AbstractFloat}}} =
    copyto!(A, randn(rng, eltype(A), size(A)...))


## Out-of-place rand / randn — construct an AT array and fill it.

Random.rand(rng::RNG{AT}, ::Type{T}, dims::Dims) where {AT, T} =
    Random.rand!(rng, similar(AT{T}, dims))
Random.randn(rng::RNG{AT}, ::Type{T}, dims::Dims) where {AT, T<:Union{AbstractFloat,
                                                                       Complex{<:AbstractFloat}}} =
    Random.randn!(rng, similar(AT{T}, dims))

# untyped: default to Float32 (matches CUDA convention; better fit than Float64
# on consumer GPUs)
Random.rand(rng::RNG{AT}, dims::Dims) where {AT} = Random.rand(rng, Float32, dims)
Random.randn(rng::RNG{AT}, dims::Dims) where {AT} = Random.randn(rng, Float32, dims)

# variadic dim spellings
Random.rand(rng::RNG, dim1::Integer, dims::Integer...) =
    Random.rand(rng, Dims((dim1, dims...)))
Random.randn(rng::RNG, dim1::Integer, dims::Integer...) =
    Random.randn(rng, Dims((dim1, dims...)))
Random.rand(rng::RNG, ::Type{T}, dim1::Integer, dims::Integer...) where T =
    Random.rand(rng, T, Dims((dim1, dims...)))
Random.randn(rng::RNG, ::Type{T}, dim1::Integer, dims::Integer...) where T =
    Random.randn(rng, T, Dims((dim1, dims...)))
