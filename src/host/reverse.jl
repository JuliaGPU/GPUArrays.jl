# reversing

# GPUArrays' `reverse` / `reverse!` forward to AcceleratedKernels, the same way the other array
# operations delegate to a single vendor-agnostic implementation: AK provides one KernelAbstractions
# reverse (whole-array and per-`dims`) that every backend shares, so there is no per-backend kernel to
# maintain here. A contiguous sub-range of a vector is reversed by handing AK a view of that range.


# n-dimensional API

function Base.reverse!(data::AnyGPUArray; dims=:)
    AK.reverse!(data; dims)
    return data
end

Base.reverse(input::AnyGPUArray; dims=:) = AK.reverse(input; dims)


# 1-dimensional API (in-place)

Base.@propagate_inbounds function Base.reverse!(data::AnyGPUVector, start::Integer,
                                                stop::Integer=lastindex(data))
    AK.reverse!(view(data, start:stop))
    return data
end

Base.reverse!(data::AnyGPUVector) = (AK.reverse!(data); data)


# 1-dimensional API (out-of-place)

Base.@propagate_inbounds function Base.reverse(input::AnyGPUVector, start::Integer,
                                               stop::Integer=lastindex(input))
    output = copy(input)
    AK.reverse!(view(output, start:stop))
    return output
end

Base.reverse(input::AnyGPUVector) = AK.reverse(input)
