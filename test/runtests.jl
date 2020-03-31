using GPUArrays, Test

include("testsuite.jl")

@testset "JLArray" begin
    using GPUArrays.JLArrays

    jl([1])

    JLArrays.allowscalar(false)
    TestSuite.test(JLArray)
end
