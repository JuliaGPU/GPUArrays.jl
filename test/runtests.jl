using GPUArrays
using Base.Test

@testset "CUDAnative backend" begin
    include("cuda.jl")
end

@testset "Threaded Julia backend" begin
    include("jlbackend.jl")
end
