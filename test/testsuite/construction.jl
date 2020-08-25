@testsuite "constructors" AT->begin
    @testset "constructors + similar" begin
        for T in supported_eltypes()
            B = AT{T}(undef, 10)
            @test B isa AT{T,1}
            @test size(B) == (10,)
            @test eltype(B) == T

            B = AT{T}(undef, 10, 10)
            @test B isa AT{T,2}
            @test size(B) == (10, 10)
            @test eltype(B) == T

            B = AT{T}(undef, (10, 10))
            @test B isa AT{T,2}
            @test size(B) == (10, 10)
            @test eltype(B) == T

            B = similar(B, Int32, 11, 15)
            @test B isa AT{Int32,2}
            @test size(B) == (11, 15)
            @test eltype(B) == Int32

            B = similar(B, T)
            @test B isa AT{T,2}
            @test size(B) == (11, 15)
            @test eltype(B) == T

            B = similar(B, (5,))
            @test B isa AT{T,1}
            @test size(B) == (5,)
            @test eltype(B) == T

            B = similar(B, 7)
            @test B isa AT{T,1}
            @test size(B) == (7,)
            @test eltype(B) == T

            B = similar(AT{Int32}, (11, 15))
            @test B isa AT{Int32,2}
            @test size(B) == (11, 15)
            @test eltype(B) == Int32

            B = similar(AT{T}, (5,))
            @test B isa AT{T,1}
            @test size(B) == (5,)
            @test eltype(B) == T

            B = similar(AT{T}, 7)
            @test B isa AT{T,1}
            @test size(B) == (7,)
            @test eltype(B) == T

            B = similar(Broadcast.Broadcasted(*, (B, B)), Int32, (11, 15))
            @test B isa AT{Int32,2}
            @test size(B) == (11, 15)
            @test eltype(B) == Int32

            B = similar(Broadcast.Broadcasted(*, (B, B)), T)
            @test B isa AT{T,2}
            @test size(B) == (11, 15)
            @test eltype(B) == T
        end
    end

    @testset "comparison against Array" begin
        for typs in [(), (Int,), (Int,1), (Int,2), (Float32,), (Float32,1), (Float32,2)],
            args in [(), (1,), (1,2), ((1,),), ((1,2),),
                     (undef,), (undef, 1,), (undef, 1,2), (undef, (1,),), (undef, (1,2),),
                     (Int,), (Int, 1,), (Int, 1,2), (Int, (1,),), (Int, (1,2),),
                     ([1,2],), ([1 2],)]
            cpu = try
                Array{typs...}(args...)
            catch ex
                isa(ex, MethodError) || rethrow()
                nothing
            end

            gpu = try
                AT{typs...}(args...)
            catch ex
                isa(ex, MethodError) || rethrow()
                cpu == nothing || rethrow()
                nothing
            end

            if cpu == nothing
                @test gpu == nothing
            else
                @test typeof(cpu) == typeof(convert(Array, gpu))
            end
        end
    end
end

@testsuite "conversions" AT->begin
    for T in supported_eltypes()
        Bc = round.(rand(10, 10) .* 10.0)
        B = AT{T}(Bc)
        @test size(B) == (10, 10)
        @test eltype(B) == T
        @test Array(B) ≈ Bc

        Bc = rand(T, 10)
        B = AT(Bc)
        @test size(B) == (10,)
        @test eltype(B) == T
        @test Array(B) ≈ Bc

        Bc = rand(T, 10, 10)
        B = AT{T, 2}(Bc)
        @test size(B) == (10, 10)
        @test eltype(B) == T
        @test Array(B) ≈ Bc

        Bc = rand(Int32, 3, 3, 3)
        B = convert(AT{T, 3}, Bc)
        @test size(B) == (3, 3, 3)
        @test eltype(B) == T
        @test Array(B) ≈ Bc
    end
end

@testsuite "value constructors" AT->begin
    for T in supported_eltypes()
        @test compare((a,b)->fill!(a, b), AT, rand(T, 3), rand(T))

        x = Matrix{T}(I, 4, 2)

        x1 = AT{T, 2}(I, 4, 2)
        x2 = AT{T}(I, (4, 2))
        x3 = AT{T, 2}(I, (4, 2))

        @test Array(x1) ≈ x
        @test Array(x2) ≈ x
        @test Array(x3) ≈ x

        x = Matrix(T(3) * I, 2, 4)
        x1 = AT(T(3) * I, 2, 4)
        @test eltype(x1) == T
        @test Array(x1) ≈ x
    end
end

@testsuite "iterator constructors" AT->begin
    for T in supported_eltypes()
        @test AT(Fill(T(0), (10,))) == fill!(similar(AT{T}, (10,)), T(0))
        @test AT(Fill(T(0), (10, 10))) == fill!(similar(AT{T}, (10, 10)), T(0))
        if T <: Real
            x = AT{Float32}(Fill(T(0), (10, 10)))
            @test eltype(x) == Float32
            @test AT(Eye{T}((10))) == AT{T}(I, 10, 10)
            x = AT{Float32}(Eye{T}(10))
            @test eltype(x) == Float32
        end
    end
end
