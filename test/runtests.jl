using GPUArrays, Test

@testset "JLArray" begin
    GPUArrays.test(JLArray)
end
