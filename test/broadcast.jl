using GPUArrays
using Base.Test

test(idx, A) = A[idx] * 2f0
ctx = opencl()

@allbackends "broadcast" ctx begin
    N = 10
    idx = GPUArray(rand(Cuint(1):Cuint(N), 2*N))
    y = Base.RefValue(GPUArray(rand(Float32, 2*N)))
    result = GPUArray(zeros(Float32, 2*N))

    result .= test.(idx, y)
    res1 = Array(result)
    res2 = similar(res1)
    result .= test.(idx, y)
    res2 .= test.(Array(idx), Base.RefValue(Array(y[])))
    @test res2 == res1

    ############
    # issue #27
    a1, a2 = rand(Float32, 4,5,3), rand(Float32, 1,5,3);
    g1 = GPUArray(a1);
    g2 = GPUArray(a2);
    @test Array(g1 .+ g2) ≈ a1 .+ a2
    a3 = rand(Float32, 1,5,1)
    g3 = GPUArray(a3)

    @test Array(g1 .+ g3) ≈ a1 .+ a3

    ############
    # issue #22

    u0 = GPUArray(rand(Float32, 32, 32))
    tmp = ones(u0)
    uprev = ones(u0)
    k1 = ones(u0)
    a = 2f0

    u0c = Array(u0)
    tmpc = Array(tmp)
    uprevc = Array(uprev)
    k1c = Array(k1)

    tmp .=  uprev .+ a .* k1
    tmpc .=  uprevc .+ a .* k1c
    @test Array(tmp) ≈ tmpc

    ############
    # issue #21

    k1 = GPUArray(rand(Float32, 32, 32))
    uprev = ones(k1)
    k1c = Array(k1)
    uprevc = Array(uprev)

    res = muladd.(2f0, k1, uprev)
    resc = muladd.(2f0, k1c, uprevc)
    @test Array(res) ≈ resc

    ###########
    # issue #20
    u0c = rand(Float32, 32, 32)
    u0 = GPUArray(u0c)
    @test Array(abs.(u0)) ≈ abs.(u0c)

    #########
    # issue #41
    Ac = rand(Float32, 32, 32)
    uprev = GPUArray(Ac)
    k1 = GPUArray(Ac)
    k2 = GPUArray(Ac)
    k3 = GPUArray(Ac)
    k4 = GPUArray(Ac)
    dt = 1.2f0
    b1 = 1.3f0
    b2 = 1.4f0
    b3 = 1.5f0
    b4 = 1.6f0
    # The first issue is likely https://github.com/JuliaLang/julia/issues/22255
    # since GPUArrays adds some arguments to the function, it becomes longer longer, hitting the 12
    # so this wont fix for now
    @. utilde = uprev + dt*(b1*k1 + b2*k2 + b3*k3 + b4*k4)

    duprev = GPUArray(Ac)
    ku = GPUArray(Ac)
    u = similar(duprev)
    uc = similar(Ac)
    fract = Float32(1//2)
    @. u = uprev + dt*duprev + dt^2*(fract*ku)
    @. uc = Ac + dt*Ac + dt^2*(fract*Ac)
    @test Array(u) ≈ uc

    testf((x)       -> fill!(x, 1),  rand(3,3))
    testf((x, y)    -> map(+, x, y), rand(2, 3), rand(2, 3))
    testf((x)       -> sin.(x),      rand(2, 3))
    testf((x)       -> 2x,      rand(2, 3))
    testf((x, y)    -> x .+ y,       rand(2, 3), rand(1, 3))
    testf((z, x, y) -> z .= x .+ y,  rand(2, 3), rand(2, 3), rand(2))
end

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

@allbackends "vec 3" ctx begin
    N = 20

    xc = map(x-> ntuple(i-> rand(Float32), Val{3}), 1:N)
    yc = map(x-> ntuple(i-> rand(Float32), Val{3}), 1:N)

    x = GPUArray(xc)
    y = GPUArray(yc)

    res1c = zeros(Float32, N)
    res2c = similar(xc)

    res1 = GPUArray(res1c)
    res2 = GPUArray(res2c)

    res1 .= testv3_1.(x, y)
    res1c .= testv3_1.(xc, yc)
    @test Array(res1) ≈ res1c

    res2 .= testv3_2.(x, y)
    res2c .= testv3_2.(xc, yc)
    @test all(map((a,b)->all((1,2,3) .≈ (1,2,3)),Array(res2), res2c))
end

# more complex function for broadcast
function test{T}(a::T, b)
    x = sqrt(sin(a) * b) / T(10.0)
    y = T(33.0)x + cos(b)
    y * T(10.0)
end

@testset "broadcast Float32" begin
    A = GPUArray(rand(Float32, 40, 40))

    A .= identity.(10f0)
    @test all(x-> x == 10f0, Array(A))

    A .= identity.(0.5f0)
    B = test.(A, 10f0)
    @test all(x-> x ≈ test(0.5f0, 10f0), Array(B))
    A .= identity.(2f0)
    C = (*).(A, 10f0)
    @test all(x-> x == 20f0, Array(C))
    D = (*).(A, B)
    @test all(x-> x ≈ test(0.5f0, 10f0) * 2, Array(D))
    D .= (+).((*).(A, B), 10f0)
    @test all(x-> x ≈ test(0.5f0, 10f0) * 2 + 10f0, Array(D))
    free(D); free(C); free(A); free(B)
end

@testset "broadcast Complex64" begin
    A = GPUArray(fill(10f0*im, 40, 40))
    A .= identity.(10f0*im)
    @test all(x-> x == 10f0*im, Array(A))

    B = angle.(A)
    @test all(x-> x ≈ angle(10f0*im), Array(B))
    A .= identity.(2f0*im)
    C = (*).(A, (2f0*im))
    @test all(x-> x ≈ 2f0*im * 2f0*im, Array(C))
    D = (*).(A, B)
    @test all(x-> x ≈ angle(10f0*im) * 2f0*im, Array(D))
    D .= (+).((*).(A, B), (0.5f0*im))
    @test all(x-> x ≈ (2f0*im * angle(10f0*im) + (0.5f0*im)), Array(D))
    free(D); free(C); free(A); free(B)
end
