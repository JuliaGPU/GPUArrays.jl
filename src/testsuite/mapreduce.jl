function test_mapreduce(Typ)
    @testset "mapreduce" begin
        for ET in supported_eltypes()
            T = Typ{ET}
            @testset "$ET" begin
                range = ET <: Integer ? (ET(-2):ET(2)) : ET
                @testset "mapreducedim" begin
                    for N in (2, 10)
                        y = rand(range, N, N)
                        x = T(y)
                        @test sum(y, dims = 2) ≈ Array(sum(x, dims = 2))
                        @test sum(y, dims = 1) ≈ Array(sum(x, dims = 1))
                        @test sum(y, dims = (1, 2)) ≈ Array(sum(x, dims = (1, 2)))

                        y = rand(range, N, 10)
                        x = T(y)
                        @test sum(y, dims = 2) ≈ Array(sum(x, dims = 2))
                        @test sum(y, dims = 1) ≈ Array(sum(x, dims = 1))

                        y = rand(range, 10, N)
                        x = T(y)
                        @test sum(y, dims = 2) ≈ Array(sum(x, dims = 2))
                        @test sum(y, dims = 1) ≈ Array(sum(x, dims = 1))
                    end
                end
                @testset "sum maximum minimum prod" begin
                    for dims in ((4048,), (1024,1024), (77,), (1923,209))
                        Ac = rand(range, dims)
                        A = T(Ac)
                        @test sum(A) ≈ sum(Ac)
                        ET <: Complex || @test maximum(A) ≈ maximum(Ac)
                        ET <: Complex || @test minimum(A) ≈ minimum(Ac)
                        @test prod(A) ≈ prod(Ac)
                    end
                end
            end
        end
        @testset "any all ==" begin
            for Ac in ([false, false], [false, true], [true, true])
                A = Typ(Ac)
                @test typeof(A) == Typ{Bool,1}
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
                A = fill(Typ{ET}, ET(0), (100,))
                B = ones(Typ{ET}, 100)
                @test !(A ≈ B)
                @test !(A ≈ Array(B))
                @test !(Array(A) ≈ B)
            end
        end
    end
end
