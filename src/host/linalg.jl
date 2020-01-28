# integration with LinearAlgebra stdlib

function LinearAlgebra.transpose!(At::AbstractGPUArray{T, 2}, A::AbstractGPUArray{T, 2}) where T
    gpu_call(At, A) do ctx, At, A
        idx = @cartesianidx A ctx
        @inbounds At[idx[2], idx[1]] = A[idx[1], idx[2]]
        return
    end
    At
end

function genperm(I::NTuple{N}, perm::NTuple{N}) where N
    ntuple(d-> (@inbounds return I[perm[d]]), Val(N))
end

function LinearAlgebra.permutedims!(dest::AbstractGPUArray, src::AbstractGPUArray, perm) where N
    perm isa Tuple || (perm = Tuple(perm))
    gpu_call(dest, src, perm) do ctx, dest, src, perm
        I = @cartesianidx src ctx
        @inbounds dest[genperm(I, perm)...] = src[I...]
        return
    end
    return dest
end

function Base.copyto!(A::AbstractArray, B::Adjoint{<:Any, <:AbstractGPUArray})
    copyto!(A, Adjoint(Array(parent(B))))
end
function Base.copyto!(A::AbstractArray, B::Transpose{<:Any, <:AbstractGPUArray})
    copyto!(A, Transpose(Array(parent(B))))
end
function Base.copyto!(A::AbstractArray, B::UpperTriangular{<:Any, <:AbstractGPUArray})
    copyto!(A, UpperTriangular(Array(parent(B))))
end
function Base.copyto!(A::AbstractArray, B::LowerTriangular{<:Any, <:AbstractGPUArray})
    copyto!(A, LowerTriangular(Array(parent(B))))
end

function Base.copyto!(A::AbstractGPUArray, B::Adjoint{T, <: AbstractGPUArray}) where T
    transpose!(A, B.parent)
end
