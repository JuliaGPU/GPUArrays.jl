using Test, JLArrays, LinearAlgebra

@testset "JLArray LU" begin
    A = jl([2.0 1.0; 1.0 3.0])
    F = lu(A; check = false)

    @test F isa LU{Float64, Matrix{Float64}, Vector{Int}}

    b = jl([1.0, 2.0])
    ldiv!(F, b)
    @test Array(b) ≈ [0.2, 0.6]

    y = similar(b)
    ldiv!(y, F, jl([1.0, 2.0]))
    @test Array(y) ≈ [0.2, 0.6]
end
