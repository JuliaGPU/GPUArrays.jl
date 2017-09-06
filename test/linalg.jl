@testset "transpose" begin
    A = rand(Float32, 32, 32)
    Agpu = GPUArray(A)
    @test Array(Agpu') == A'
end
@testset "PermuteDims" begin
  testf(x -> permutedims(x, (2, 1)), rand(2, 3))
  testf(x -> permutedims(x, (2, 1, 3)), rand(4, 5, 6))
end
