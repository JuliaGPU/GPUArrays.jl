using GPUArrays, Test
using GPUArrays.TestSuite


@testset "Julia reference implementation:" begin
    TestSuite.run_tests(JLArray)
end
