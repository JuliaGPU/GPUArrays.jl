using GPUArrays

try
    using XUnit
catch
    using Test
end

@testset "GPUArrays" begin

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

end
