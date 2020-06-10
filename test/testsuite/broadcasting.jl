@testsuite "broadcasting" AT->begin
    broadcasting(AT)
    vec3(AT)
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

function broadcasting(AT)
    for ET in supported_eltypes()
        N = 10
        @testset "broadcast $ET" begin
            @testset "RefValue" begin
                cidx = rand(1:Int(N), 2*N)
                gidx = AT(cidx)
                cy = rand(ET, 2*N)
                gy = AT(cy)
                cres = fill(zero(ET), size(cidx))
                gres = AT(cres)
                gres .= test_idx.(gidx, Base.RefValue(gy))
                cres .= test_idx.(cidx, Base.RefValue(cy))
                @test Array(gres) == cres
            end

            @testset "Tuple" begin
                @test compare(AT, rand(ET, 3, N), rand(ET, 3, N), rand(ET, N), rand(ET, N), rand(ET, N)) do out, arr, a, b, c
                    broadcast!(out, arr, (a, b, c)) do xx, yy
                        xx + first(yy)
                    end
                end
            end

            @testset "Adjoint and Transpose" begin
                A = AT(rand(ET, N))
                A' .= ET(2)
                @test all(isequal(ET(2)'), A)
                transpose(A) .= ET(1)
                @test all(isequal(ET(1)), A)
            end

            ############
            # issue #27
            @test compare((a, b)-> a .+ b, AT, rand(ET, 4, 5, 3), rand(ET, 1, 5, 3))
            @test compare((a, b)-> a .+ b, AT, rand(ET, 4, 5, 3), rand(ET, 1, 5, 1))

            ############
            # issue #22
            dim = (32, 32)
            @test compare(AT, rand(ET, dim), rand(ET, dim), rand(ET, dim)) do tmp, a1, a2
                tmp .=  a1 .+ a2 .* ET(2)
            end

            ############
            # issue #21
            if ET in (Float32, Float64)
                @test compare((a1, a2)-> muladd.(ET(2), a1, a2), AT, rand(ET, dim), rand(ET, dim))
                #########
                # issue #41
                # The first issue is likely https://github.com/JuliaLang/julia/issues/22255
                # since GPUArrays adds some arguments to the function, it becomes longer longer, hitting the 12
                # so this wont fix for now
                @test compare(AT, rand(ET, dim), rand(ET, dim), rand(ET, dim), rand(ET, dim), rand(ET, dim), rand(ET, dim)) do a1, a2, a3, a4, a5, a6
                    @. a1 = a2 + (1.2) *((1.3)*a3 + (1.4)*a4 + (1.5)*a5 + (1.6)*a6)
                end

                @test compare(AT, rand(ET, dim), rand(ET, dim), rand(ET, dim), rand(ET, dim)) do u, uprev, duprev, ku
                    fract = ET(1//2)
                    dt = ET(1.4)
                    dt2 = dt^2
                    @. u = uprev + dt*duprev + dt2*(fract*ku)
                end
                @test compare((x) -> (-).(x), AT, rand(ET, 2, 3))

                @test compare(AT, rand(ET, dim), rand(ET, dim), rand(ET, dim), rand(ET, dim), rand(ET, dim), rand(ET, dim)) do utilde, gA, k1, k2, k3, k4
                    btilde1 = ET(1)
                    btilde2 = ET(1)
                    btilde3 = ET(1)
                    btilde4 = ET(1)
                    dt = ET(1)
                    @. utilde = dt*(btilde1*k1 + btilde2*k2 + btilde3*k3 + btilde4*k4)
                end
            end

            @test compare((x) -> fill!(x, 1), AT, rand(ET, 3,3))
            @test compare((x, y) -> map(+, x, y), AT, rand(ET, 2, 3), rand(ET, 2, 3))

            @test compare((x) -> 2x, AT, rand(ET, 2, 3))
            @test compare((x, y) -> x .+ y, AT, rand(ET, 2, 3), rand(ET, 1, 3))
            @test compare((z, x, y) -> z .= x .+ y, AT, rand(ET, 2, 3), rand(ET, 2, 3), rand(ET, 2))

            @test compare(A -> A .= identity.(ET(10)), AT, rand(ET, 40, 40))
            @test compare(A -> test_kernel.(A, ET(10)), AT, rand(ET, 40, 40))
            @test compare(A -> A .* ET(10), AT, rand(ET, 40, 40))
            @test compare((A, B) -> A .* B, AT, rand(ET, 40, 40), rand(ET, 40, 40))
            @test compare((A, B) -> A .* B .+ ET(10), AT, rand(ET, 40, 40), rand(ET, 40, 40))
        end

        @testset "map! $ET" begin
            @test compare(AT, rand(2,2), rand(2,2)) do x,y
                map!(+, x, y)
            end
            @test compare(AT, rand(2), rand(2,2)) do x,y
                map!(+, x, y)
            end
            @test compare(AT, rand(2,2), rand(2)) do x,y
                map!(+, x, y)
            end
        end

        @testset "map $ET" begin
            @test compare(AT, rand(2,2), rand(2,2)) do x,y
                map(+, x, y)
            end
            @test compare(AT, rand(2), rand(2,2)) do x,y
                map(+, x, y)
            end
            @test compare(AT, rand(2,2), rand(2)) do x,y
                map(+, x, y)
            end
        end
    end

    @testset "0D" begin
        x = AT{Float64}(undef)
        x .= 1
        @test collect(x)[] == 1
        x /= 2
        @test collect(x)[] == 0.5
    end

    @testset "Ref" begin
        # as first arg, 0d broadcast
        @test compare(x->getindex.(Ref(x),1), AT, [0])

        void_setindex!(args...) = (setindex!(args...); return)
        @test compare(x->(void_setindex!.(Ref(x),1); x), AT, [0])

        # regular broadcast
        a = AT(rand(10))
        b = AT(rand(10))
        cpy(i,a,b) = (a[i] = b[i]; return)
        cpy.(1:10, Ref(a), Ref(b))
        @test Array(a) == Array(b)
    end

    @testset "stackoverflow in copy(::Broadcast)" begin
        copy(Base.broadcasted(identity, AT(Int[])))
    end
end

function vec3(AT)
    @testset "vec 3" begin
        N = 20

        xc = map(x-> ntuple(i-> rand(Float32), Val(3)), 1:N)
        yc = map(x-> ntuple(i-> rand(Float32), Val(3)), 1:N)

        x = AT(xc)
        y = AT(yc)

        res1c = fill(0f0, N)
        res2c = similar(xc)

        res1 = AT(res1c)
        res2 = AT(res2c)

        res1 .= testv3_1.(x, y)
        res1c .= testv3_1.(xc, yc)
        @test Array(res1) ≈ res1c

        res2 .= testv3_2.(x, y)
        res2c .= testv3_2.(xc, yc)
        @test all(map((a,b)-> all((1,2,3) .≈ (1,2,3)), Array(res2), res2c))
    end
end
