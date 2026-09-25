# Base algorithms that other back-ends implement on the device, and that GPUArrays' generic
# sparse code uses. JLArrays runs them on the host.

Base.sortperm(x::JLVector; kwargs...) = JLArray(sortperm(Array(x); kwargs...))

function Base.accumulate!(op, B::JLArray, A::JLArray; kwargs...)
    copyto!(B, accumulate(op, Array(A); kwargs...))
    return B
end

Base.findall(bools::JLArray{Bool}) = JLArray(findall(Array(bools)))
Base.findall(f::Function, A::JLArray) = JLArray(findall(f, Array(A)))
Base.findall(f::Base.Fix2{typeof(in)}, A::JLArray) = JLArray(findall(f, Array(A)))
