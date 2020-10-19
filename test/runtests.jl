using GPUArrays, Test

include("testsuite.jl")

@testset "JLArray" begin
    include("jlarray.jl")
    using .JLArrays

    jl([1])

    JLArrays.allowscalar(false)
    TestSuite.test(JLArray)
end

@testset "Array" begin
    TestSuite.test(Array)
end
