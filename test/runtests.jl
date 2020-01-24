using GPUArrays, Test

include("testsuite.jl")

@testset "JLArray" begin
    using GPUArrays.JLArrays
    TestSuite.test(JLArray)
end
