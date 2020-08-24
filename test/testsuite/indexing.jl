@testsuite "indexing" AT->begin
    @allowscalar @testset "errors and warnings" begin
        x = AT([0])

        allowscalar(true, false)
        x[1] = 1
        @test x[1] == 1

        @disallowscalar begin
            @test_throws ErrorException x[1]
            @test_throws ErrorException x[1] = 1
        end

        x[1] = 2
        @test x[1] == 2

        allowscalar(false)
        @test_throws ErrorException x[1]
        @test_throws ErrorException x[1] = 1

        @allowscalar begin
            x[1] = 3
            @test x[1] == 3
        end

        allowscalar() do
            x[1] = 4
            @test x[1] == 4
        end

        @test_throws ErrorException x[1]
        @test_throws ErrorException x[1] = 1

        allowscalar(true, false)
        x[1]

        allowscalar(true, true)
        @test_logs (:warn, r"Performing scalar operations on GPU arrays: .*") x[1]
        @test_logs x[1]

        # NOTE: this inner testset _needs_ to be wrapped with allowscalar
        #       to make sure its original value is restored.
    end

    @allowscalar for T in (Float32, Int32)
        @testset "Indexing with $T" begin
            x = rand(T, 32)
            src = AT(x)
            for (i, xi) in enumerate(x)
                @test src[i] == xi
            end
            @test Array(src[1:3]) == x[1:3]
            @test Array(src[3:end]) == x[3:end]
        end
        @testset "multi dim, sliced setindex" begin
            x = fill(AT{T}, T(0), (10, 10, 10, 10))
            y = AT{T}(undef, 5, 5, 10, 10)
            rand!(y)
            x[2:6, 2:6, :, :] = y
            @test Array(x[2:6, 2:6, :, :]) == Array(y)
        end
        @testset "multi dim, sliced setindex, CPU source" begin
            x = fill(AT{T}, T(0), (2,3,4))
            y = Array{T}(undef, 2,3)
            rand!(y)
            x[:, :, 2] = y
            @test x[:, :, 2] == y
        end
    end

    @allowscalar for T in (Float32, Int32)
        @testset "Indexing with $T" begin
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
    end

    @allowscalar for T in (Float32, Int32)
        @testset "issue #42 with $T" begin
            Ac = rand(Float32, 2, 2)
            A = AT(Ac)
            @test A[1] == Ac[1]
            @test A[end] == Ac[end]
            @test A[1, 1] == Ac[1, 1]
        end
    end

    @allowscalar @testset "get/setindex!" begin
        # literal calls to get/setindex! have different return types
        @test compare(x->getindex(x,1), AT, zeros(Int, 2))
        @test compare(x->setindex!(x,1,1), AT, zeros(Int, 2))
    end

    @allowscalar @testset "Index with empty array" begin
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

    @testset "indexing with indirect CPU source" begin
        # JuliaGPU/CUDA.jl#345
        a = rand(3,4)
        i = rand(1:3,2,2)
        @test compare(a->a[i,:], AT, a)
        @test compare(a->a[i',:], AT, a)
        @test compare(a->a[view(i,1,:),:], AT, a)
    end
end
