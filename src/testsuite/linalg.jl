
function run_linalg(Typ)
    T = Typ{Float32}
    @testset "Linalg" begin
        @testset "transpose" begin
            against_base(ctranspose, T, (32, 32))
        end
        @testset "PermuteDims" begin
            against_base(x -> permutedims(x, (2, 1)), T, (2, 3))
            against_base(x -> permutedims(x, (2, 1, 3)), T, (4, 5, 6))
            against_base(x -> permutedims(x, (3, 1, 2)), T, (4, 5, 6))
        end
    end
end
