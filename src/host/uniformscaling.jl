import Base: +, -

const genericwrappers = (
    :LowerTriangular,
    :UpperTriangular,
    :Hermitian,
    :Symmetric
)

const unittriangularwrappers = (
    (:UnitUpperTriangular, :UpperTriangular), 
    (:UnitLowerTriangular, :LowerTriangular)
)

@kernel function kernel_generic(B, J)
    lin_idx = @index(Global, Linear)
    @inbounds diag_idx = diagind(B)[lin_idx]
    @inbounds B[diag_idx] += J
end

@kernel function kernel_unittriangular(B, J, diagonal_val)
    lin_idx = @index(Global, Linear)
    @inbounds diag_idx = diagind(B)[lin_idx]
    @inbounds B[diag_idx] = diagonal_val + J
end

for (t1, t2) in unittriangularwrappers
    @eval begin
        function (+)(A::$t1{T, <:AbstractGPUMatrix}, J::UniformScaling) where T
            B = similar(parent(A), typeof(oneunit(T) + J))
            copyto!(B, parent(A))
            min_size = minimum(size(B))
            kernel_unittriangular(get_backend(B))(B, J, one(eltype(B)); ndrange=min_size)
            return $t2(B)
        end

        function (-)(J::UniformScaling, A::$t1{T, <:AbstractGPUMatrix}) where T
            B = similar(parent(A), typeof(J - oneunit(T)))
            B .= .- parent(A)
            min_size = minimum(size(B))
            kernel_unittriangular(get_backend(B))(B, J, -one(eltype(B)); ndrange=min_size)
            return $t2(B)
        end
    end
end

for t in genericwrappers
    @eval begin
        function (+)(A::$t{T, <:AbstractGPUMatrix}, J::UniformScaling) where T
            B = similar(parent(A), typeof(oneunit(T) + J))
            copyto!(B, parent(A))
            min_size = minimum(size(B))
            kernel_generic(get_backend(B))(B, J; ndrange=min_size)
            return $t(B)
        end

        function (-)(J::UniformScaling, A::$t{T, <:AbstractGPUMatrix}) where T
            B = similar(parent(A), typeof(J - oneunit(T)))
            B .= .- parent(A)
            min_size = minimum(size(B))
            kernel_generic(get_backend(B))(B, J; ndrange=min_size)
            return $t(B)
        end
    end
end

# Breaking Hermiticity when adding a complex value to the diagonal
function (+)(A::Hermitian{T,<:AbstractGPUMatrix}, J::UniformScaling{<:Complex}) where T
    B = similar(parent(A), typeof(oneunit(T) + J))
    copyto!(B, parent(A))
    min_size = minimum(size(B))
    kernel_generic(get_backend(B))(B, J; ndrange=min_size)
    return B
end

function (-)(J::UniformScaling{<:Complex}, A::Hermitian{T,<:AbstractGPUMatrix}) where T
    B = similar(parent(A), typeof(J - oneunit(T)))
    B .= .-parent(A)
    min_size = minimum(size(B))
    kernel_generic(get_backend(B))(B, J; ndrange=min_size)
    return B
end

# Finally the generic matrix version
function (+)(A::AbstractGPUMatrix{T}, J::UniformScaling) where T
    B = similar(A, typeof(oneunit(T) + J))
    copyto!(B, A)
    min_size = minimum(size(B))
    kernel_generic(get_backend(B))(B, J; ndrange=min_size)
    return B
end

function (-)(J::UniformScaling, A::AbstractGPUMatrix{T}) where T
    B = similar(A, typeof(J - oneunit(T)))
    B .= .-A
    min_size = minimum(size(B))
    kernel_generic(get_backend(B))(B, J; ndrange=min_size)
    return B
end
