@testsuite "indexing scalar" (AT, eltypes)->begin
    AT <: AbstractGPUArray && @testset "errors and warnings" begin
        x = AT([0])

        @test_throws ErrorException x[]

        @allowscalar begin
            x[] = 1
            @test x[] == 1
        end

        @test_throws ErrorException x[]

        allowscalar() do
            x[] = 2
            @test x[] == 2
        end

        @test_throws ErrorException x[]

        @allowscalar y = 42
        @test y == 42
    end

    @allowscalar @testset "getindex with $T" for T in eltypes
        x = rand(T, 32)
        src = AT(x)
        for (i, xi) in enumerate(x)
            @test src[i] == xi
        end
        @test Array(src[1:3]) == x[1:3]
        @test Array(src[3:end]) == x[3:end]
    end

    @allowscalar @testset "setindex! with $T" for T in eltypes
        x = fill(zero(T), 7)
        src = AT(x)
        for i = 1:7
            src[i] = i
        end
        @test Array(src) == T[1:7;]
        src[1:3] = T[77, 22, 11]
        @test Array(src[1:3]) == T[77, 22, 11]
        src[1] = T(0)
    end

    @allowscalar @testset "issue #42 with $T" for T in eltypes
        Ac = rand(Float32, 2, 2)
        A = AT(Ac)
        @test A[1] == Ac[1]
        @test A[end] == Ac[end]
        @test A[1, 1] == Ac[1, 1]
    end

    @allowscalar @testset "get/setindex!" begin
        # literal calls to get/setindex! have different return types
        @test compare(x->getindex(x,1), AT, zeros(Int, 2))
        @test compare(x->setindex!(x,1,1), AT, zeros(Int, 2))

        # issue #319
        @test compare(x->setindex!(x,1,1,1), AT, zeros(Float32, 2, 2))
    end
end

@testsuite "indexing multidimensional" (AT, eltypes)->begin
    @testset "sliced setindex" for T in eltypes
        x = AT(zeros(T, (10, 10, 10, 10)))
        y = AT(rand(T, (5, 5, 10, 10)))
        x[2:6, 2:6, :, :] = y
        @test Array(x[2:6, 2:6, :, :]) == Array(y)
    end

    @testset "sliced setindex, CPU source" for T in eltypes
        x = AT(zeros(T, (2,3,4)))
        y = AT(rand(T, (2,3)))
        x[:, :, 2] = y
        @test Array(x[:, :, 2]) == Array(y)
    end

    @allowscalar @testset "empty array" begin
        @testset "1D" begin
            Ac = zeros(Float32, 10)
            A = AT(Ac)
            @test typeof(A[[]]) == typeof(AT(Ac[[]]))
            @test size(A[[]]) == size(Ac[[]])
        end

        @testset "2D with other index $other" for other in (Colon(), 1:5, 5)
            Ac = zeros(Float32, 10, 10)
            A = AT(Ac)

            @test typeof(A[[], other]) == typeof(AT(Ac[[], other]))
            @test size(A[[], other]) == size(Ac[[], other])

            @test typeof(A[other, []]) == typeof(AT(Ac[other, []]))
            @test size(A[other, []]) == size(Ac[other, []])
        end

        @test compare(AT, rand(Float32, 2,2)) do a
            a[:, 2:end-2] = AT{Float32}(undef,2,0)
        end
    end

    @testset "GPU source" begin
        a = rand(Float32, 3)
        i = rand(1:3, 2)
        @test compare(getindex, AT, a, i)
        @test compare(getindex, AT, a, i')
        @test compare(getindex, AT, a, view(i, 2:2))
    end

    @testset "CPU source" begin
        # JuliaGPU/CUDA.jl#345
        a = rand(Float32, 3, 4)
        i = rand(1:3,2,2)
        @test compare(a->a[i,:], AT, a)
        @test compare(a->a[i',:], AT, a)
        @test compare(a->a[view(i,1,:),:], AT, a)
    end

    @testset "JuliaGPU/CUDA.jl#461: sliced setindex" begin
        @test compare((X,Y)->(X[1,:] = Y), AT, zeros(Float32, 2,2), ones(Float32, 2))
    end
end

@testsuite "indexing find" (AT, eltypes)->begin
    @testset "findfirst" begin
        # 1D
        @test compare(findfirst, AT, rand(Bool, 100))
        @test compare(x->findfirst(>(0.5f0), x), AT, rand(Float32, 100))
        let x = fill(false, 10)
            @test findfirst(x) == findfirst(AT(x))
        end

        # ND
        let x = rand(Bool, 10, 10)
            @test findfirst(x) == findfirst(AT(x))
        end
        let x = rand(Float32, 10, 10)
            @test findfirst(>(0.5f0), x) == findfirst(>(0.5f0), AT(x))
        end
    end

    @testset "findmax & findmin" begin
        let x = rand(Float32, 100)
            @test findmax(x) == findmax(AT(x))
            @test findmax(x; dims=1) == Array.(findmax(AT(x); dims=1))

            x[32] = x[33] = x[55] = x[66] = NaN32
            @test isequal(findmax(x), findmax(AT(x)))
            @test isequal(findmax(x; dims=1), Array.(findmax(AT(x); dims=1)))
        end
        let x = rand(Float32, 10, 10)
            @test findmax(x) == findmax(AT(x))
            @test findmax(x; dims=1) == Array.(findmax(AT(x); dims=1))
            @test findmax(x; dims=2) == Array.(findmax(AT(x); dims=2))

            x[rand(CartesianIndices((10, 10)), 10)] .= NaN
            @test isequal(findmax(x), findmax(AT(x)))
            @test isequal(findmax(x; dims=1), Array.(findmax(AT(x); dims=1)))
        end
        let x = rand(Float32, 10, 10, 10)
            @test findmax(x) == findmax(AT(x))
            @test findmax(x; dims=1) == Array.(findmax(AT(x); dims=1))
            @test findmax(x; dims=2) == Array.(findmax(AT(x); dims=2))
            @test findmax(x; dims=3) == Array.(findmax(AT(x); dims=3))

            x[rand(CartesianIndices((10, 10, 10)), 20)] .= NaN
            @test isequal(findmax(x), findmax(AT(x)))
            @test isequal(findmax(x; dims=1), Array.(findmax(AT(x); dims=1)))
            @test isequal(findmax(x; dims=2), Array.(findmax(AT(x); dims=2)))
            @test isequal(findmax(x; dims=3), Array.(findmax(AT(x); dims=3)))
        end

        let x = rand(Float32, 100)
            @test findmin(x) == findmin(AT(x))
            @test findmin(x; dims=1) == Array.(findmin(AT(x); dims=1))

            x[32] = x[33] = x[55] = x[66] = NaN32
            @test isequal(findmin(x), findmin(AT(x)))
            @test isequal(findmin(x; dims=1), Array.(findmin(AT(x); dims=1)))
        end
        let x = rand(Float32, 10, 10)
            @test findmin(x) == findmin(AT(x))
            @test findmin(x; dims=1) == Array.(findmin(AT(x); dims=1))
            @test findmin(x; dims=2) == Array.(findmin(AT(x); dims=2))

            x[rand(CartesianIndices((10, 10)), 10)] .= NaN
            @test isequal(findmin(x), findmin(AT(x)))
            @test isequal(findmin(x; dims=1), Array.(findmin(AT(x); dims=1)))
            @test isequal(findmin(x; dims=2), Array.(findmin(AT(x); dims=2)))
            @test isequal(findmin(x; dims=3), Array.(findmin(AT(x); dims=3)))
        end
        let x = rand(Float32, 10, 10, 10)
            @test findmin(x) == findmin(AT(x))
            @test findmin(x; dims=1) == Array.(findmin(AT(x); dims=1))
            @test findmin(x; dims=2) == Array.(findmin(AT(x); dims=2))
            @test findmin(x; dims=3) == Array.(findmin(AT(x); dims=3))

            x[rand(CartesianIndices((10, 10, 10)), 20)] .= NaN
            @test isequal(findmin(x), findmin(AT(x)))
            @test isequal(findmin(x; dims=1), Array.(findmin(AT(x); dims=1)))
            @test isequal(findmin(x; dims=2), Array.(findmin(AT(x); dims=2)))
            @test isequal(findmin(x; dims=3), Array.(findmin(AT(x); dims=3)))
        end
    end

    @testset "argmax & argmin" begin
        @test compare(argmax, AT, rand(Int, 10))
        @test compare(argmax, AT, -rand(Int, 10))

        @test compare(argmin, AT, rand(Int, 10))
        @test compare(argmin, AT, -rand(Int, 10))
    end
end
