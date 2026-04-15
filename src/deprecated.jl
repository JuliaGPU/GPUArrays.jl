# RNG: the old RNG took a GPU state array and Vector{UInt32} seeds.
# The new stateless Philox4x32 RNG doesn't need either.

function RNG(state::AbstractGPUArray)
    Base.depwarn("RNG(state::AbstractGPUArray) is deprecated, use RNG() instead", :RNG)
    RNG()
end

function Random.seed!(rng::RNG, seed::Vector{UInt32})
    Base.depwarn("seed!(rng::RNG, seed::Vector{UInt32}) is deprecated, use seed!(rng, seed::Integer) instead", :seed!)
    Random.seed!(rng, isempty(seed) ? rand(Random.RandomDevice(), UInt64) : first(seed))
end
