@testsuite "vectors" (AT, eltypes)->begin
    ## push! not defined for most GPU arrays,
    ## uncomment once it is.
    # a = Float32[]
    # x = AT(a)
    # @test length(x) == 0
    # @test push!(x, 12)
    # @allowscalar begin
    #     @test x[1] == 12
    # end

    a = Float32[0]
    x = AT(a)
    resize!(x, 3)
    @test length(x) == 3

    a = Float32[0, 1, 2]
    x = AT(a)
    resize!(x, 2)
    @test length(x) == 2
    @allowscalar begin
        @test x[1] == 0
        @test x[2] == 1
    end

    a = Float32[0, 1, 2]
    x = AT(a)
    append!(x, [3, 4])
    @test length(x) == 5
    @allowscalar begin
        @test x[4] == 3
        @test x[5] == 4
    end

    a = Float32[0, 1, 2]
    x = AT(a)
    append!(x, (3, 4))
    @test length(x) == 5
    @allowscalar begin
        @test x[4] == 3
        @test x[5] == 4
    end
end
