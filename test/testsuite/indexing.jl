@testsuite "indexing scalar" AT->begin
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
    end

    @allowscalar @testset "getindex with $T" for T in supported_eltypes()
        x = rand(T, 32)
        src = AT(x)
        for (i, xi) in enumerate(x)
            @test src[i] == xi
        end
        @test Array(src[1:3]) == x[1:3]
        @test Array(src[3:end]) == x[3:end]
    end

    @allowscalar @testset "setindex! with $T" for T in supported_eltypes()
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

    @allowscalar @testset "issue #42 with $T" for T in supported_eltypes()
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
        @test compare(x->setindex!(x,1,1,1), AT, zeros(Float64, 2, 2))
    end
end

@testsuite "indexing multidimensional" AT->begin
    @testset "sliced setindex" for T in supported_eltypes()
        x = AT(zeros(T, (10, 10, 10, 10)))
        y = AT(rand(T, (5, 5, 10, 10)))
        x[2:6, 2:6, :, :] = y
        @test Array(x[2:6, 2:6, :, :]) == Array(y)
    end

    @testset "sliced setindex, CPU source" for T in supported_eltypes()
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
        a = rand(3)
        i = rand(1:3, 2)
        @test compare(getindex, AT, a, i)
        @test compare(getindex, AT, a, i')
        @test compare(getindex, AT, a, view(i, 2:2))
    end

    @testset "CPU source" begin
        # JuliaGPU/CUDA.jl#345
        a = rand(3,4)
        i = rand(1:3,2,2)
        @test compare(a->a[i,:], AT, a)
        @test compare(a->a[i',:], AT, a)
        @test compare(a->a[view(i,1,:),:], AT, a)
    end

    @testset "JuliaGPU/CUDA.jl#461: sliced setindex" begin
        @test compare((X,Y)->(X[1,:] = Y), AT, zeros(2,2), ones(2))
    end
end
