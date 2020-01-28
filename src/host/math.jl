import Base.clamp!

function Base.clamp!(A::AbstractGPUArray, low, high)
    function kernel(state, A, low, high)
        I = @cartesianidx A state
        A[I...] = clamp(A[I...], low, high)
        return
    end
    gpu_call(kernel, A, low, high)
    return A
end
