

function run_broadcasting(Typ)
    @testset "broadcast" begin
        test_broadcast(Typ)
        test_vec3(Typ)
    end
end

test_idx(idx, A::AbstractArray{T}) where T = A[idx] * T(2)

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
function test_kernel(a::T, b) where T
    c = a
    for i=1:5
        c += T(b)
        c = a * T(2)
    end
    return c
end


function test_broadcast(Typ)
    for ET in supported_eltypes()
        N = 10
        T = Typ{ET}
        @testset "broadcast $ET" begin
            @testset "RefValue" begin
                cidx = rand(UInt32(1):UInt32(N), 2*N)
                gidx = Typ(cidx)
                cy = TestSuite.toarray(ET, (2*N,))
                gy = Typ(cy)
                cres = zeros(ET, size(cidx))
                gres = Typ(cres)
                gres .= test_idx.(gidx, Base.RefValue(gy))
                cres .= test_idx.(cidx, Base.RefValue(cy))
                @test Array(gres) == cres
            end
            @testset "Tuple" begin
                against_base(T, (3, N), (3, N), (N,), (N,), (N,)) do out, arr, a, b, c
                    res2 = broadcast!(out, arr, (a, b, c)) do xx, yy
                        xx + sum(yy)
                    end
                end
            end
            ############
            # issue #27
            against_base((a, b)-> a .+ b, T, (4, 5, 3), (1, 5, 3))
            against_base((a, b)-> a .+ b, T, (4, 5, 3), (1, 5, 1))

            ############
            # issue #22
            dim = (32, 32)
            against_base(T, dim, dim, dim) do tmp, a1, a2
                tmp .=  a1 .+ a2 .* ET(2)
            end

            ############
            # issue #21
            if ET in (Float32, Float64)
                against_base((a1, a2)-> muladd.(ET(2), a1, a2), T, dim, dim)
                #########
                # issue #41
                # The first issue is likely https://github.com/JuliaLang/julia/issues/22255
                # since GPUArrays adds some arguments to the function, it becomes longer longer, hitting the 12
                # so this wont fix for now
                against_base(T, dim, dim, dim, dim, dim, dim) do a1, a2, a3, a4, a5, a6
                    @. a1 = a2 + (1.2) *((1.3)*a3 + (1.4)*a4 + (1.5)*a5 + (1.6)*a6)
                end

                against_base(T, dim, dim, dim, dim) do u, uprev, duprev, ku
                    fract = ET(1//2)
                    dt = ET(1.4)
                    dt2 = dt^2
                    @. u = uprev + dt*duprev + dt2*(fract*ku)
                end
                against_base((x) -> (-).(x), T, (2, 3))

                against_base(T, dim, dim, dim, dim, dim, dim) do utilde, gA, k1, k2, k3, k4
                    btilde1 = ET(1)
                    btilde2 = ET(1)
                    btilde3 = ET(1)
                    btilde4 = ET(1)
                    dt = ET(1)
                    @. utilde = dt*(btilde1*k1 + btilde2*k2 + btilde3*k3 + btilde4*k4)
                end
            end

            against_base((x) -> fill!(x, 1), T, (3,3))
            against_base((x, y) -> map(+, x, y), T, (2, 3), (2, 3))

            against_base((x) -> 2x, T, (2, 3))
            against_base((x, y) -> x .+ y, T, (2, 3), (1, 3))
            against_base((z, x, y) -> z .= x .+ y, T, (2, 3), (2, 3), (2,))

            T = Typ{ET}
            against_base(A -> A .= identity.(ET(10)), T, (40, 40))
            against_base(A -> test_kernel.(A, ET(10)), T, (40, 40))
            against_base(A -> A .* ET(10), T, (40, 40))
            against_base((A, B) -> A .* B, T, (40, 40), (40, 40))
            against_base((A, B) -> A .* B .+ ET(10), T, (40, 40), (40, 40))
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
