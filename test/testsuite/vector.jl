@testsuite "vectors" AT->begin
    a = Float32[]
    x = AT(a)
    @test length(x) == 0
    push!(x, 12f0)
    @test length(x) == 1
    @test x[1] == 12f0

    a = Float32[0]
    x = AT(a)
    @test length(x) == 1
    @test length(GPUArrays.buffer(x)) == 1
    push!(x, 12)
    @test length(GPUArrays.buffer(x)) == GPUArrays.grow_dimensions(0, 1, 1)
    resize!(x, 5)
    @test length(x) == 5
    @test length(GPUArrays.buffer(x)) == 5

    resize!(x, 3)
    @test length(x) == 3
    # we don't shrink buffers yet... TODO shrink them... or should we?
    @test length(GPUArrays.buffer(x)) == 5

    x = AT(Array{Float32}(undef, 16))
    reshape!(x, (2, 2, 2, 2))
    @test size(x) == (2, 2, 2, 2)
    x = AT(Array{Float32}(undef, 16))
    y = AT(rand(Float32, 32))
    update!(x, y)
    @test size(x) == (32,)
end
