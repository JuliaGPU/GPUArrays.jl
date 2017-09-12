module Broadcasting

using GPUArrays
using Base.Test, GPUArrays.TestSuite

function main(Typ)
    @testset "broadcast" begin
        test_vec3(Typ)
        test_broadcast(Typ)
    end
end

test_idx(idx, A) = A[idx] * 2f0

function testv3_1(a, b)
    x = a .* b
    y =  x .+ b
    @inbounds return y[1]
end

function testv3_2(a, b)
    x = a .* b
    y =  x .+ b
    return y
end

# more complex function for broadcast
function test_kernel{T}(a::T, b)
    c = a
    for i=1:5
        c += T(b)
        c = a * T(2)
    end
    return c
end


function test_broadcast(Typ)
    @testset "broadcast" begin
        N = 10
        T = Typ{Float32}

        idx = Typ(rand(Cuint(1):Cuint(N), 2*N))
        y = Base.RefValue(Typ(rand(Float32, 2*N)))
        result = Typ(zeros(Float32, 2*N))

        result .= test_idx.(idx, y)
        res1 = Array(result)
        res2 = similar(res1)
        res2 .= test_idx.(Array(idx), Base.RefValue(Array(y[])))
        @test res2 == res1

        ############
        # issue #27
        against_base((a, b)-> a .+ b, T, (4, 5, 3), (1, 5, 3))
        against_base((a, b)-> a .+ b, T, (4, 5, 3), (1, 5, 1))

        ############
        # issue #22
        dim = (32, 32)
        against_base(T, dim, dim, dim) do tmp, a1, a2
            tmp .=  a1 .+ a2 .* 2f0
        end

        ############
        # issue #21
        against_base((a1, a2)-> muladd.(2f0, a1, a2), T, dim, dim)

        ###########
        # issue #20
        against_base(a-> abs.(a), T, dim)

        #########
        # issue #41
        # The first issue is likely https://github.com/JuliaLang/julia/issues/22255
        # since GPUArrays adds some arguments to the function, it becomes longer longer, hitting the 12
        # so this wont fix for now
        against_base(T, dim, dim, dim, dim, dim, dim) do a1, a2, a3, a4, a5, a6
            @. a1 = a2 + 1.2f0*(1.3f0*a3 + 1.4f0*a4 + 1.5f0*a5 + 1.6f0*a6)
        end

        against_base(T, dim, dim, dim, dim) do u, uprev, duprev, ku
            fract = Float32(1//2)
            dt = 1.4f0
            @. u = uprev + dt*duprev + dt^2*(fract*ku)
        end

        against_base((x) -> fill!(x, 1), T, (3,3))
        against_base((x, y) -> map(+, x, y), T, (2, 3), (2, 3))
        against_base((x) -> sin.(x), T, (2, 3))
        against_base((x) -> 2x, T, (2, 3))
        against_base((x, y) -> x .+ y, T, (2, 3), (1, 3))
        against_base((z, x, y) -> z .= x .+ y, T, (2, 3), (2, 3), (2,))

        for ET in (Float32, Complex64)
            T = Typ{ET}
            against_base(A -> A .= identity.(10f0), T, (40, 40))
            against_base(A -> test_kernel.(A, 10f0), T, (40, 40))
            against_base(A -> A .* 10f0, T, (40, 40))
            against_base((A, B) -> A .* B, T, (40, 40), (40, 40))
            against_base((A, B) -> A .* B .+ 10f0, T, (40, 40), (40, 40))
        end
    end
end

function test_vec3(Typ)
    @testset "vec 3" begin
        N = 20

        xc = map(x-> ntuple(i-> rand(Float32), Val{3}), 1:N)
        yc = map(x-> ntuple(i-> rand(Float32), Val{3}), 1:N)

        x = Typ(xc)
        y = Typ(yc)

        res1c = zeros(Float32, N)
        res2c = similar(xc)

        res1 = Typ(res1c)
        res2 = Typ(res2c)

        res1 .= testv3_1.(x, y)
        res1c .= testv3_1.(xc, yc)
        @test Array(res1) ≈ res1c

        res2 .= testv3_2.(x, y)
        res2c .= testv3_2.(xc, yc)
        @test all(map((a,b)-> all((1,2,3) .≈ (1,2,3)), Array(res2), res2c))
    end
end



end

using Broadcasting
Broadcasting.main(CLArray)
