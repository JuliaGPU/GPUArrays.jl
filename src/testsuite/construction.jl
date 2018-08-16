function test_construction(AT)
    @testset "construction" begin
        constructors(AT)
        conversion(AT)
        value_constructor(AT)
        iterator_constructors(AT)
    end
end

function constructors(AT)
    @testset "constructors + similar" begin
        for T in supported_eltypes()
            B = AT{T}(10)
            @test B isa AT{T,1}
            @test size(B) == (10,)
            @test eltype(B) == T

            B = AT{T}(10, 10)
            @test B isa AT{T,2}
            @test size(B) == (10, 10)
            @test eltype(B) == T

            B = AT{T}((10, 10))
            @test B isa AT{T,2}
            @test size(B) == (10, 10)
            @test eltype(B) == T

            B = similar(B, Int32, (11, 15))
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

            B = similar(AT{Int32, 2}, T, (11, 15))
            @test B isa AT{T,2}
            @test size(B) == (11, 15)
            @test eltype(B) == T

            B = similar(AT{T}, (5,))
            @test B isa AT{T,1}
            @test size(B) == (5,)
            @test eltype(B) == T

            B = similar(AT{T}, 7)
            @test B isa AT{T,1}
            @test size(B) == (7,)
            @test eltype(B) == T
        end
    end
end

function conversion(AT)
    @testset "conversion" begin
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
end

function value_constructor(AT)
    @testset "value constructors" begin
        for T in supported_eltypes()
            x = fill(zero(T), (2, 2))
            x1 = fill(AT{T}, T(0), (2, 2))
            x2 = fill(AT{T}, T(0), (2, 2))
            x3 = fill(AT{T, 2}, T(0), (2, 2))
            @test Array(x1) ≈ x
            @test Array(x2) ≈ x
            @test Array(x3) ≈ x

            x = fill(T(1), (2, 2))
            x1 = fill(AT{T}, T(1), (2, 2))
            x2 = fill(AT{T}, T(1), (2, 2))
            x3 = fill(AT{T, 2}, T(1), (2, 2))
            @test Array(x1) ≈ x
            @test Array(x2) ≈ x
            @test Array(x3) ≈ x

            fill!(x, 0)

            x1 = fill(AT{T}, 0f0, (2, 2))
            x2 = fill(AT{T}, T(0), (2, 2))
            x3 = fill(AT{T, 2}, T(0), (2, 2))
            @test Array(x1) ≈ x
            @test Array(x2) ≈ x
            @test Array(x3) ≈ x

            fill!(x1, 2f0)
            x2 = fill!(AT{Int32}((4, 4, 4)), 77f0)
            @test all(x-> x == 2f0, Array(x1))
            @test all(x-> x == Int32(77), Array(x2))

            x = Matrix{T}(I, 2, 2)

            x1 = AT{T, 2}(I, 2, 2)
            x2 = AT{T}(I, (2, 2))
            x3 = AT{T, 2}(I, (2, 2))

            @test Array(x1) ≈ x
            @test Array(x2) ≈ x
            @test Array(x3) ≈ x
        end
    end
end
function iterator_constructors(AT)
    @testset "iterator constructors" begin
        for T in supported_eltypes()
            @test AT(Fill(T(0), (10,))) == fill(AT{T}, T(0), (10,))
            @test AT(Fill(T(0), (10, 10))) == fill(AT{T}, T(0), (10, 10))
            if T <: Real
                x = AT{Float32}(Fill(T(0), (10, 10)))
                @test eltype(x) == Float32
                @test AT(Eye{T}((10, 10))) == AT{T}(I, 10, 10)
                x = AT{Float32}(Eye{T}((10, 10)))
                @test eltype(x) == Float32
            end
        end
    end
end
