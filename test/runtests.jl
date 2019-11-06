using GPUArrays, Test

include("testsuite.jl")

@testset "JLArray" begin
    TestSuite.test(JLArray)
end
