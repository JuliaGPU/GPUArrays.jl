using GPUArrays, Test, Pkg

include("testsuite.jl")

@testset "JLArray" begin
    using JLArrays

    jl([1])

    TestSuite.test(JLArray)
end

@testset "Array" begin
    TestSuite.test(Array)
end
