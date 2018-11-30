function test_mapreduce(AT)
    @testset "mapreduce" begin
        for ET in supported_eltypes()
            T = AT{ET}
            @testset "$ET" begin
                range = ET <: Integer ? (ET(-2):ET(2)) : ET
                @testset "mapreducedim" begin
                    for N in (2, 10)
                        @test compare(x -> sum(x, dims=2),      AT, rand(range, N, N))
                        @test compare(x -> sum(x, dims=1),      AT, rand(range, N, N))
                        @test compare(x -> sum(x, dims=(1, 2)), AT, rand(range, N, N))

                        @test compare(x -> sum(x, dims=2),      AT, rand(range, N, 10))
                        @test compare(x -> sum(x, dims=1),      AT, rand(range, N, 10))

                        @test compare(x -> sum(x, dims=2),      AT, rand(range, 10, N))
                        @test compare(x -> sum(x, dims=1),      AT, rand(range, 10, N))

                        _zero = zero(ET)
                        _addone(z) = z + one(z)
                        @test compare(x->mapreduce(_addone, +, x; dims = 2),
                                      AT, rand(range, N, N))
                        @test compare(x->mapreduce(_addone, +, x; dims = 2, init = _zero),
                                      AT, rand(range, N, N))
                    end
                end
                @testset "sum maximum minimum prod" begin
                    for dims in ((4048,), (1024,1024), (77,), (1923,209))
                        @test compare(sum,  AT, rand(range, dims))
                        @test compare(prod, AT, rand(range, dims))
                        ET <: Complex || @test compare(maximum, AT,rand(range, dims))
                        ET <: Complex || @test compare(minimum, AT,rand(range, dims))
                    end
                end
            end
        end
        @testset "any all ==" begin
            for Ac in ([false, false], [false, true], [true, true])
                A = AT(Ac)
                @test typeof(A) == AT{Bool,1}
                @test any(A) == any(Ac)
                @test all(A) == all(Ac)
                @test A == copy(A)
                @test A !== copy(A)
                @test A == deepcopy(A)
                @test A !== deepcopy(A)
            end
        end

        @testset "isapprox" begin
            for ET in supported_eltypes()
                ET <: Complex && continue
                A = fill(AT{ET}, ET(0), (100,))
                B = ones(AT{ET}, 100)
                @test !(A ≈ B)
                @test !(A ≈ Array(B))
                @test !(Array(A) ≈ B)


                ca = AT(randn(ComplexF64,3,3))
                cb = copy(ca)
                cb[1:1, 1:1] .+= 1e-7im
                @test isapprox(ca, cb, atol=1e-5)
                @test !isapprox(ca, cb, atol=1e-9)
            end
        end
    end
end
