# reverse

# JLArrays are backed by a regular Array, so reverse defers to Base on that storage. This keeps the
# reference backend independent of AcceleratedKernels' GPU kernels. `reverse!` goes through the
# out-of-place `reverse` so every `dims` form (integer, tuple, `:`) is handled by Base directly.
# These mirror the GPUArrays generic signatures one-for-one so JLArrays wins by specificity.

Base.reverse!(data::AnyJLArray; dims=:) =
    (copyto!(typed_data(data), reverse(typed_data(data); dims)); data)

Base.reverse(input::AnyJLArray; dims=:) = JLArray(reverse(typed_data(input); dims))

Base.@propagate_inbounds function Base.reverse!(data::AnyJLVector, start::Integer,
                                                stop::Integer=lastindex(data))
    reverse!(typed_data(data), start, stop)
    return data
end

Base.reverse!(data::AnyJLVector) = (reverse!(typed_data(data)); data)

Base.@propagate_inbounds function Base.reverse(input::AnyJLVector, start::Integer,
                                               stop::Integer=lastindex(input))
    JLArray(reverse(typed_data(input), start, stop))
end

Base.reverse(input::AnyJLVector) = JLArray(reverse(typed_data(input)))
