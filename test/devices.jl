using GPUArrays

@testset "context creation" begin
    ctx = GPUArrays.init(:opencl)
    ctx2 = GPUArrays.init(:opencl)
    @test ctx === ctx2

    ctx = GPUArrays.init(:cudanative)
    ctx2 = GPUArrays.init(:cudanative)
    @test ctx === ctx2

    ctx = GPUArrays.init(:julia)
    ctx2 = GPUArrays.init(:julia)
    @test ctx === ctx2
end
