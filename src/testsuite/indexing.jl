

function run_indexing(Typ)
    @testset "indexing" begin
        for T in (Float32, Int32#=, SVector{3, Float32}=#)
            @testset "Indexing with $T" begin
                x = rand(T, 32)
                src = Typ(x)
                GPUArrays.allowslow(true)
                for (i, xi) in enumerate(x)
                    @test src[i] == xi
                end
                GPUArrays.allowslow(false)
                @test Array(src[1:3]) == x[1:3]
                @test Array(src[3:end]) == x[3:end]
            end
            @testset "multi dim, sliced setindex" begin
                x = zeros(Typ{T}, 10, 10, 10, 10)
                y = rand(Typ{T}, 5, 5, 10, 10)
                x[2:6, 2:6, :, :] = y
                x[2:6, 2:6, :, :] == y
           end

        end

        for T in (Float32, Int32)
            @testset "Indexing with $T" begin
                x = zeros(T, 7)
                src = Typ(x)
                GPUArrays.allowslow(true)
                for i = 1:7
                    src[i] = i
                end
                @test Array(src) == T[1:7;]
                src[1:3] = T[77, 22, 11]
                @test Array(src[1:3]) == T[77, 22, 11]
                src[1] = T(0)
                src[2:end] = 77
                GPUArrays.allowslow(false)
                @test Array(src) == T[0, 77, 77, 77, 77, 77, 77]
            end
        end

        for T in (Float32, Int32)
            @testset "issue #42 with $T" begin
                Ac = rand(Float32, 2, 2)
                A = Typ(Ac)
                GPUArrays.allowslow(true)
                @test A[1] == Ac[1]
                @test A[end] == Ac[end]
                @test A[1, 1] == Ac[1, 1]
                GPUArrays.allowslow(false)
            end
        end
        for T in (Float32, Int32)
            @testset "Colon() $T" begin
                Ac = rand(T, 10)
                A = Typ(Ac)
                GPUArrays.allowslow(false)
                A[:] = T(1)
                @test all(x-> x == 1, A)
                A[:] = Typ(Ac)
                @test Array(A) == Ac
            end
        end
    end
end
