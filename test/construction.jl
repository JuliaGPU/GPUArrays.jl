module Construction

using GPUArrays
using Base.Test, GPUArrays.TestSuite

function main(Typ)
    @testset "Construction" begin
        construction(Typ)
        conversion(Typ)
        value_constructor(Typ)
    end
end

function construction(Typ)
    @testset "similar + constructor" begin
        B = Typ{Float32}(10)
        @test size(B) == (10,)
        @test eltype(B) == Float32

        B = Typ{Float32}(10, 10)
        @test size(B) == (10, 10)
        @test eltype(B) == Float32

        B = Typ{Float32}((10, 10))
        @test size(B) == (10, 10)
        @test eltype(B) == Float32

        B = similar(B, Int32, (11, 15))
        @test size(B) == (11, 15)
        @test eltype(B) == Int32

        B = similar(B, Float32)
        @test size(B) == (11, 15)
        @test eltype(B) == Float32

        B = similar(B, (5,))
        @test size(B) == (5,)
        @test eltype(B) == Float32

        B = similar(B, 7)
        @test size(B) == (7,)
        @test eltype(B) == Float32


        B = similar(Typ{Int32}, (11, 15))
        @test size(B) == (11, 15)
        @test eltype(B) == Int32

        B = similar(Typ{Int32, 2}, Float32, (11, 15))
        @test size(B) == (11, 15)
        @test eltype(B) == Float32

        B = similar(Typ{Float32}, (5,))
        @test size(B) == (5,)
        @test eltype(B) == Float32

        B = similar(Typ{Float32}, 7)
        @test size(B) == (7,)
        @test eltype(B) == Float32
    end
end

function conversion(Typ)
    @testset "conversion" begin
        Bc = rand(10, 10)
        B = Typ{Float32}(Bc)
        @test size(B) == (10, 10)
        @test eltype(B) == Float32
        @test Array(B) ≈ Bc

        Bc = rand(Float32, 10)
        B = Typ(Bc)
        @test size(B) == (10,)
        @test eltype(B) == Float32
        @test Array(B) ≈ Bc

        Bc = rand(Int32, 10, 10)
        B = Typ{Int32, 2}(Bc)
        @test size(B) == (10, 10)
        @test eltype(B) == Int32
        @test Array(B) ≈ Bc

        Bc = rand(Int32, 3, 3, 3)
        B = convert(Typ{Float32, 3}, Bc)
        @test size(B) == (3, 3, 3)
        @test eltype(B) == Float32
        @test Array(B) ≈ Bc
    end
end

function value_constructor(Typ)
    @testset "value constructor" begin
        x = zeros(Float32, 2, 2)
        x1 = zeros(Typ{Float32}, 2, 2)
        x2 = zeros(Typ{Int32}, (2, 2))
        x3 = zeros(Typ{Int32, 2}, (2, 2))
        @test Array(x1) ≈ x
        @test Array(x2) ≈ x
        @test Array(x3) ≈ x

        x1 = fill(Typ{Float32}, 0f0, 2, 2)
        x2 = fill(Typ{Int32}, 0f0, (2, 2))
        x3 = fill(Typ{Float32, 2}, 0f0, (2, 2))
        @test Array(x1) ≈ x
        @test Array(x2) ≈ x
        @test Array(x3) ≈ x

        fill!(x1, 2f0)
        x2 = fill!(Typ{Int32}((4, 4, 4)), 77f0)
        @test all(x-> x == 2f0, Array(x1))
        @test all(x-> x == Int32(77), Array(x2))

        x = eye(Float32, 2, 2)
        x1 = eye(Typ{Int32, 2}, 2, 2)
        x2 = eye(Typ{Float32}, (2, 2))
        x3 = eye(Typ{Float32, 2}, (2, 2))

        @test Array(x1) ≈ x
        @test Array(x2) ≈ x
        @test Array(x3) ≈ x

    end
end

end
