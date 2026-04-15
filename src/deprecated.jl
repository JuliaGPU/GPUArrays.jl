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
