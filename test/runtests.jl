using GPUArrays
using Base.Test

function jltest(a, b)
    x = sqrt(sin(a) * b) / 10
    y = 33x + cos(b)
    y*10
end

@testset "CUDAnative backend" begin
    include("cuda.jl")
end

@testset "Threaded Julia backend" begin
    include("jlbackend.jl")
end
