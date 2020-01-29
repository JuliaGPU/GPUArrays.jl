using GPUArrays, Test

include("testsuite.jl")

@testset "JLArray" begin
    using GPUArrays.JLArrays
    JLArrays.allowscalar(false)
    TestSuite.test(JLArray)
end
