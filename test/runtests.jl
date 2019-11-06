using GPUArrays, Test

include("testsuite.jl")

@testset "JLArray" begin
    test(JLArray)
end
