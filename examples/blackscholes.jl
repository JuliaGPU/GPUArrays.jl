using JTensors
import JTensors: JLBackend, CLBackend
import JTensors.JLBackend: JLArray

function blackscholes(
        sptprice,
        strike,
        rate,
        volatility,
        time
    )
    logterm = log10( sptprice / strike)
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

@inline function cndf2(x)
    0.5f0 + 0.5f0 * erf(0.707106781f0 * x)
end
using OpenCL: cl
function test(result, a, b, c, d, e)
    event = (result .= blackscholes.(a, b, c, d, e))
    cl.wait(event)
end
function test2(result, a, b, c, d, e)
    result .= blackscholes.(a, b, c, d, e)
end

N = 10^7
sptprice   = Float32[ 42.0 for i = 1:N ];
initStrike = Float32[ 40.0 + (i / N) for i = 1:N ];
rate       = Float32[ 0.5 for i = 1:N ];
volatility = Float32[ 0.2 for i = 1:N ];
time       = Float32[ 0.5 for i = 1:N ];
result = similar(time)
result .= blackscholes.(sptprice, initStrike, rate, volatility, time)

ctx = CLBackend.init()
sptprice_gpu = GPUArray(sptprice, flag = :r)
initStrike_gpu = GPUArray(initStrike, flag = :r)
rate_gpu = GPUArray(rate, flag = :r)
volatility_gpu = GPUArray(volatility, flag = :r)
time_gpu = GPUArray(time, flag = :r)
result_gpu = GPUArray(result, flag = :w)

using BenchmarkTools
@time test(
    result_gpu,
    sptprice_gpu,
    initStrike_gpu,
    rate_gpu,
    volatility_gpu,
    time_gpu
)
@time test2(
    result,
    sptprice,
    initStrike,
    rate,
    volatility,
    time
)

0.1 / 0.000652
0.99 / 0.002
