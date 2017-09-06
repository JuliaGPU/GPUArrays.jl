import SpecialFunctions: erfc
using GPUArrays: gpu_sub2ind

function blackscholes(sptprice, strike, rate, volatility, time)
    logterm = log( sptprice / strike)
    powterm = .5f0 * volatility * volatility
    den = volatility * sqrt(time)
    d1 = (((rate + powterm) * time) + logterm) / den
    d2 = d1 - den
    NofXd1 = cndf2(d1)
    NofXd2 = cndf2(d2)
    futureValue = strike * exp(- rate * time)
    c1 = futureValue * NofXd2
    call_ = sptprice * NofXd1 - c1
    put  = call_ - futureValue + sptprice
    return put
end

function cndf2(x)
    0.5f0 + 0.5f0 * erfc(0.707106781f0 * x)
end

# jeeez -.- So CUDAnative still doesn't recognize e.g. sqrt in the LLVM-IR,
#since it's implemented within a C-library.... Should be fixed soon!
function cu_blackscholes(sptprice, strike, rate, volatility, time)
    logterm = cu.log(sptprice / strike)
    powterm = .5f0 * volatility * volatility
    den = volatility * cu.sqrt(time)
    d1 = (((rate + powterm) * time) + logterm) / den
    d2 = d1 - den
    NofXd1 = cu_cndf2(d1)
    NofXd2 = cu_cndf2(d2)
    futureValue = strike * cu.exp(- rate * time)
    c1 = futureValue * NofXd2
    call_ = sptprice * NofXd1 - c1
    put  = call_ - futureValue + sptprice
    return put
end

function cu_cndf2(x)
    0.5f0 + 0.5f0 * cu.erfc(0.707106781f0 * x)
end

for T in (Float32,)
    N = 60
    sptprice   = T[42.0 for i = 1:N]
    initStrike = T[40.0 + (i / N) for i = 1:N]
    rate       = T[0.5 for i = 1:N]
    volatility = T[0.2 for i = 1:N]
    time       = T[0.5 for i = 1:N]
    result     = similar(time)
    comparison = blackscholes.(sptprice, initStrike, rate, volatility, time)
    @allbackends "Blackscholes with $T" backend begin
        _sptprice = GPUArray(sptprice)
        _initStrike = GPUArray(initStrike)
        _rate = GPUArray(rate)
        _volatility = GPUArray(volatility)
        _time = GPUArray(time)
        _result = GPUArray(result)
        blackschole_f = if is_cudanative(backend)
            cu_blackscholes
        else
            blackscholes
        end
        _result .= blackschole_f.(_sptprice, _initStrike, _rate, _volatility, _time)
        @test Array(_result) ≈ comparison
    end
end

@allbackends "mapidx" backend begin
    a = rand(Complex64, 77)
    b = rand(Complex64, 77)
    A = GPUArray(a)
    B = GPUArray(b)
    off = Cuint(1)
    mapidx(A, (B, off, Cuint(length(A)))) do i, a, b, off, len
        x = b[i]
        x2 = b[min(i+off, len)]
        a[i] = x * x2
    end
    foreach(1:length(a)) do i
        x = b[i]
        x2 = b[min(i+off, length(a))]
        a[i] = x * x2
    end
    @test Array(A) ≈ a
end


@allbackends "muladd & abs" backend begin
    a = rand(Float32, 32) - 0.5f0
    A = GPUArray(a)
    x = abs.(A)
    @test Array(x) == abs.(a)
    y = muladd.(A, 2f0, x)
    @test Array(y) == muladd(a, 2f0, abs.(a))
end


@allbackends "copy!" backend begin
    x = zeros(Float32, 10, 10)
    y = rand(Float32, 20, 10)
    a = GPUArray(x)
    b = GPUArray(y)
    r1 = CartesianRange(CartesianIndex(1, 3), CartesianIndex(7, 8))
    r2 = CartesianRange(CartesianIndex(4, 3), CartesianIndex(10, 8))
    copy!(x, r1, y, r2)
    copy!(a, r1, b, r2)
    @test x == Array(a)

    x2 = zeros(Float32, 10, 10)
    copy!(x2, r1, b, r2)
    @test x2 == x

    fill!(a, 0f0)
    copy!(a, r1, y, r2)
    @test Array(a) == x
end

@allbackends "vcat + hcat" backend begin
    x = zeros(Float32, 10, 10)
    y = rand(Float32, 20, 10)
    a = GPUArray(x)
    b = GPUArray(y)
    @test vcat(x, y) == Array(vcat(a, b))
    z = rand(Float32, 10, 10)
    c = GPUArray(z)
    @test hcat(x, z) == Array(hcat(a, c))
    
    testf(hcat, rand(3, 3), rand(3, 3))
    testf(vcat, rand(3, 3), rand(3, 3))
end

@allbackends "reinterpret" backend begin
    a = rand(Complex64, 22)
    A = GPUArray(a)
    af0 = reinterpret(Float32, a)
    Af0 = reinterpret(Float32, A)
    @test Array(Af0) == af0

    a = rand(Complex64, 10 * 10)
    A = GPUArray(a)
    af0 = reinterpret(Float32, a, (20, 10))
    Af0 = reinterpret(Float32, A, (20, 10))
    @test Array(Af0) == af0
end

function ntuple_test(state, result, ::Val{N}) where N
    result[1] = ntuple(Val{N}) do i
        Float32(i) * 77f0
    end
    return
end

function ntuple_closure(state, result, ::Val{N}, testval) where N
    result[1] = ntuple(Val{N}) do i
        Float32(i) * testval
    end
    return
end


@allbackends "ntuple test" backend begin
    result = GPUArray(Vector{NTuple{3, Float32}}(1))
    gpu_call(ntuple_test, result, (result, Val{3}()))
    @test result[1] == (77, 2*77, 3*77)
    x = 88f0
    gpu_call(ntuple_closure, result, (result, Val{3}(), x))
    @test result[1] == (x, 2*x, 3*x)
end

function cartesian_iter(state, A, res, Asize)
    for i in CartesianRange(CartesianIndex(Asize))
        idx = gpu_sub2ind(Asize, i.I)
        res[idx] = A[idx]
    end
    return
end

@allbackends "cartesian iteration" backend begin
    Ac = rand(Float32, 32, 32)
    A = GPUArray(Ac)
    result = zeros(A)
    gpu_call(cartesian_iter, result, (A, result, Cint.(size(A))))
    Array(result) == Ac
end




function clmap!(state, f, out, b)
    i = linear_index(out, state) # get the kernel index it gets scheduled on
    out[i] = f(b[i])
    return
end
@testset "Custom kernel from Julia function" begin
    x = GPUArray(rand(Float32, 100))
    y = GPUArray(rand(Float32, 100))
    gpu_call(clmap!, x, (sin, x, y))
    # same here, x is just passed to supply a kernel size!
    jy = Array(y)
    @test map!(sin, jy, jy) ≈ Array(x)
end
