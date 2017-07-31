using GPUArrays
using Base.Test

test(idx, A) = A[idx] * 2f0

@allbackends "broadcast" backend begin
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


@allbackends "vec 3" backend begin
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

Sugar.isintrinsic(Transpiler.CLMethod(Complex64))
