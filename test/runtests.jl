using GPUArrays, Test

include("testsuite.jl")

@testset "JLArray" begin
    include("../lib/JLArrays/src/JLArrays.jl")  # get the latest file directly, ignore the registry
    using .JLArrays

    jl([1])

    TestSuite.test(JLArray)
end

@testset "Array" begin
    TestSuite.test(Array)
end
