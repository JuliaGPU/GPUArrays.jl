import SpecialFunctions: erfc
using CLArrays

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

using BenchmarkTools

T = Float32
N = 10^7
sptprice   = T[42.0 for i = 1:N]
initStrike = T[40.0 + (i / N) for i = 1:N]
rate       = T[0.5 for i = 1:N]
volatility = T[0.2 for i = 1:N]
time       = T[0.5 for i = 1:N]
result     = similar(time)
comparison = blackscholes.(sptprice, initStrike, rate, volatility, time)
_sptprice = CLArray(sptprice)
_initStrike = CLArray(initStrike)
_rate = CLArray(rate)
_volatility = CLArray(volatility)
_time = CLArray(time)
_result = CLArray(result)

@time begin
    _result .= blackscholes.(_sptprice, _initStrike, _rate, _volatility, _time)
    (GPUArrays.synchronize)(_result)
end
