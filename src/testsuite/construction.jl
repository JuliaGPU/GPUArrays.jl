using FillArrays

function run_construction(Typ)
    @testset "Construction" begin
        constructors(Typ)
        conversion(Typ)
        value_constructor(Typ)
        iterator_constructors(Typ)
    end
end

function constructors(Typ)
    @testset "similar + constructor" begin
        for T in supported_eltypes()
            B = Typ{T}(10)
            @test size(B) == (10,)
            @test eltype(B) == T

            B = Typ{T}(10, 10)
            @test size(B) == (10, 10)
            @test eltype(B) == T

            B = Typ{T}((10, 10))
            @test size(B) == (10, 10)
            @test eltype(B) == T

            B = similar(B, Int32, (11, 15))
            @test size(B) == (11, 15)
            @test eltype(B) == Int32

            B = similar(B, T)
            @test size(B) == (11, 15)
            @test eltype(B) == T

            B = similar(B, (5,))
            @test size(B) == (5,)
            @test eltype(B) == T

            B = similar(B, 7)
            @test size(B) == (7,)
            @test eltype(B) == T


            B = similar(Typ{Int32}, (11, 15))
            @test size(B) == (11, 15)
            @test eltype(B) == Int32

            B = similar(Typ{Int32, 2}, T, (11, 15))
            @test size(B) == (11, 15)
            @test eltype(B) == T

            B = similar(Typ{T}, (5,))
            @test size(B) == (5,)
            @test eltype(B) == T

            B = similar(Typ{T}, 7)
            @test size(B) == (7,)
            @test eltype(B) == T
        end
    end
end

function conversion(Typ)
    @testset "conversion" begin
        for T in supported_eltypes()
            Bc = round.(rand(10, 10) .* 10.0)
            B = Typ{T}(Bc)
            @test size(B) == (10, 10)
            @test eltype(B) == T
            @test Array(B) ≈ Bc

            Bc = rand(T, 10)
            B = Typ(Bc)
            @test size(B) == (10,)
            @test eltype(B) == T
            @test Array(B) ≈ Bc

            Bc = rand(T, 10, 10)
            B = Typ{T, 2}(Bc)
            @test size(B) == (10, 10)
            @test eltype(B) == T
            @test Array(B) ≈ Bc

            Bc = rand(Int32, 3, 3, 3)
            B = convert(Typ{T, 3}, Bc)
            @test size(B) == (3, 3, 3)
            @test eltype(B) == T
            @test Array(B) ≈ Bc
        end
    end
end

function value_constructor(Typ)
    @testset "value constructor" begin
        for T in supported_eltypes()
            x = fill(zero(T), (2, 2))
            x1 = fill(Typ{T}, T(0), (2, 2))
            x2 = fill(Typ{T}, T(0), (2, 2))
            x3 = fill(Typ{T, 2}, T(0), (2, 2))
            @test Array(x1) ≈ x
            @test Array(x2) ≈ x
            @test Array(x3) ≈ x

            x = fill(T(1), (2, 2))
            x1 = fill(Typ{T}, T(1), (2, 2))
            x2 = fill(Typ{T}, T(1), (2, 2))
            x3 = fill(Typ{T, 2}, T(1), (2, 2))
            @test Array(x1) ≈ x
            @test Array(x2) ≈ x
            @test Array(x3) ≈ x

            fill!(x, 0)

            x1 = fill(Typ{T}, 0f0, (2, 2))
            x2 = fill(Typ{T}, T(0), (2, 2))
            x3 = fill(Typ{T, 2}, T(0), (2, 2))
            @test Array(x1) ≈ x
            @test Array(x2) ≈ x
            @test Array(x3) ≈ x

            fill!(x1, 2f0)
            x2 = fill!(Typ{Int32}((4, 4, 4)), 77f0)
            @test all(x-> x == 2f0, Array(x1))
            @test all(x-> x == Int32(77), Array(x2))

            x = Matrix{T}(I, 2, 2)

            x1 = Typ{T, 2}(I, 2, 2)
            x2 = Typ{T}(I, (2, 2))
            x3 = Typ{T, 2}(I, (2, 2))

            @test Array(x1) ≈ x
            @test Array(x2) ≈ x
            @test Array(x3) ≈ x
        end
    end
end
function iterator_constructors(Typ)
    @testset "iterator constructors" begin
        for T in supported_eltypes()
            @test Typ(Fill(T(0), (10,))) == fill(Typ{T}, T(0), (10,))
            @test Typ(Fill(T(0), (10, 10))) == fill(Typ{T}, T(0), (10, 10))
            if T <: Real
                x = Typ{Float32}(Fill(T(0), (10, 10)))
                @test eltype(x) == Float32
                @test Typ(Eye{T}((10, 10))) == Typ{T}(I, 10, 10)
                x = Typ{Float32}(Eye{T}((10, 10)))
                @test eltype(x) == Float32
            end
        end
    end
end
