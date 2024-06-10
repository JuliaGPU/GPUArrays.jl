using GPUArrays, Test, Pkg

@testset "GPUArraysCore" begin
    include("gpuarrayscore.jl")
end

include("testsuite.jl")

@testset "JLArray" begin
    using JLArrays

    jl([1])

    TestSuite.test(JLArray)
end

@testset "Array" begin
    TestSuite.test(Array)
end
