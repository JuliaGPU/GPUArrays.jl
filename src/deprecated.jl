# RNG: the old RNG took a GPU state array and Vector{UInt32} seeds.
# The new stateless Philox4x32 RNG doesn't need either.

function RNG(state::AbstractGPUArray)
    AT = Base.typename(typeof(state)).wrapper
    Base.depwarn("RNG(state::AbstractGPUArray) is deprecated, use RNG{$AT}() instead", :RNG)
    RNG{AT}()
end

function Random.seed!(rng::RNG, seed::Vector{UInt32})
    Base.depwarn("seed!(rng::RNG, seed::Vector{UInt32}) is deprecated, use seed!(rng, seed::Integer) instead", :seed!)
    Random.seed!(rng, isempty(seed) ? rand(Random.RandomDevice(), UInt64) : first(seed))
end

# Stub kept so downstream packages that still extend `GPUArrays.default_rng`
# (Metal.jl, etc.) continue to load. The interface itself is gone — `RNG{AT}()`
# is now constructed directly — so any extension of this is dead code.
function default_rng end

# The per-format abstract types are replaced by the concrete, storage-parametric types.
# The aliases keep `isa` checks and dispatch working, but cannot be subtyped.
Base.@deprecate_binding AbstractGPUSparseMatrixCSR GPUSparseMatrixCSR false
Base.@deprecate_binding AbstractGPUSparseMatrixCSC GPUSparseMatrixCSC false
Base.@deprecate_binding AbstractGPUSparseMatrixCOO GPUSparseMatrixCOO false
