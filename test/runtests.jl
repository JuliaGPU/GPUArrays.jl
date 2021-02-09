using GPUArrays

using Pkg
Pkg.add(url="https://github.com/maleadt/XUnit.jl", rev="tb/for_loop")

using XUnit

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
