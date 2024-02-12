# Base mathematical operations

function Base.clamp!(A::AnyGPUArray, low, high)
    @kernel function clamp_kernel!(A::AnyGPUArray, low, high)
        I = @index(Global, Cartesian)
        A[I] = clamp(A[I], low, high)
    end
    clamp_kernel!(backend(A))(A, low, high, ndrange = size(A))
    return A
end
