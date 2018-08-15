using GPUArrays, Test
using GPUArrays.TestSuite


@testset "Julia reference implementation:" begin
    GPUArrays.test(JLArray)
end
