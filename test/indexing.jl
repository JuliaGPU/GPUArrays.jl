using GPUArrays, StaticArrays
using Base.Test

for T in (Float32, Int32, SVector{3, Float32})
    @allbackends "Indexing with $T" backend begin
        x = rand(T, 32)
        src = GPUArray(x)
        for (i, xi) in enumerate(x)
            @test src[i] == xi
        end
        @test src[1:3] == x[1:3]
        @test src[3:end] == x[3:end]
    end
end

for T in (Float32, Int32)
    @allbackends "Indexing with $T" backend begin
        x = zeros(T, 7)
        src = GPUArray(x)
        for i = 1:7
            src[i] = i
        end
        @test Array(src) == T[1:7;]
        src[1:3] = T[77, 22, 11]
        @test src[1:3] == T[77, 22, 11]
        # src[2:end] = 77
        # src[1] = T(0)
        # @test Array(src) == T[0, 77, 77, 77, 77, 77, 77]
    end
end

for T in (Float32, Int32)
    @allbackends "Indexing with $T" backend begin
        x = zeros(T, 7)
        src = GPUArray(x)
        for i = 1:7
            src[i] = i
        end
        @test Array(src) == T[1:7;]
        src[1:3] = T[77, 22, 11]
        @test src[1:3] == T[77, 22, 11]
        # src[2:end] = 77
        # src[1] = T(0)
        # @test Array(src) == T[0, 77, 77, 77, 77, 77, 77]
    end
end

using GPUArrays
CLBackend.init()

for T in (Float32, Int32)
    @allbackends "issue #42 with $T" backend begin
        Ac = rand(Float32, 2, 2)
        A = GPUArray(Ac)
        @test A[1] == Ac[1]
        @test A[end] == Ac[end]
        @test A[1, 1] == Ac[1, 1]
    end
end
