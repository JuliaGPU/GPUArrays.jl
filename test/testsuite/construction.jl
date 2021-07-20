@testsuite "construct/direct" AT->begin
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
    end

    # compare against Array
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

@testsuite "construct/similar" AT->begin
    for T in supported_eltypes()
        B = AT{T}(undef, 10)

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

        B = similar(Broadcast.Broadcasted(*, (B, B)), T)
        @test B isa AT{T,1}
        @test size(B) == (7,)
        @test eltype(B) == T

        B = similar(Broadcast.Broadcasted(*, (B, B)), Int32, (11, 15))
        @test B isa AT{Int32,2}
        @test size(B) == (11, 15)
        @test eltype(B) == Int32
    end
end

@testsuite "construct/convenience" AT->begin
    for T in supported_eltypes()
        A = AT(rand(T, 3))
        b = rand(T)
        fill!(A, b)
        @test A isa AT{T,1}
        @test Array(A) == fill(b, 3)

        A = zero(AT(rand(T, 2)))
        @test A isa AT{T,1}
        @test Array(A) == zero(rand(T, 2))

        A = zero(AT(rand(T, 2, 2)))
        @test A isa AT{T,2}
        @test Array(A) == zero(rand(T, 2, 2))

        A = zero(AT(rand(T, 2, 2, 2)))
        @test A isa AT{T,3}
        @test Array(A) == zero(rand(T, 2, 2, 2))

        A = one(AT(rand(T, 2, 2)))
        @test A isa AT{T,2}
        @test Array(A) == one(rand(T, 2, 2))

        A = oneunit(AT(rand(T, 2, 2)))
        @test A isa AT{T,2}
        @test Array(A) == oneunit(rand(T, 2, 2))
    end
end

@testsuite "construct/conversions" AT->begin
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

@testsuite "construct/uniformscaling" AT->begin
    for T in supported_eltypes()
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
