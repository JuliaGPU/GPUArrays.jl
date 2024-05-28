using Test, GPUArraysCore, EnzymeCore

@testset "EnzymeCore" begin
    @test nothing == EnzymeCore.EnzymeRules.inactive_noinl(GPUArraysCore.assertscalar)

    @test nothing == EnzymeCore.EnzymeRules.inactive_noinl(GPUArraysCore.default_scalar_indexing)

    @test nothing == EnzymeCore.EnzymeRules.inactive_noinl(GPUArraysCore.allowscalar, identity)
end
