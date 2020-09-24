# Base mathematical operations

function Base.clamp!(A::AnyGPUArray, low, high)
    gpu_call(A, low, high) do ctx, A, low, high
        I = @cartesianidx A
        A[I] = clamp(A[I], low, high)
        return
    end
    return A
end
