# you might need to add SpecialFunctions with Pkg.add("SpecialFunctions")
using GPUArrays, SpecialFunctions
using GPUArrays: perbackend, synchronize, free

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


@noinline function test(result, a, b, c, d, e)
    result .= blackscholes.(a, b, c, d, e)
    GPUArrays.synchronize(result)
end


const cu = CUDAnative

# jeeez -.- So CUDAnative still doesn't recognize e.g. sqrt in the LLVM-IR,
#since it's implemented within a C-library.... Should be fixed soon!
function cu_blackscholes(sptprice, strike, rate, volatility, time)
    logterm = cu.log10( sptprice / strike)
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
    0.5f0 + 0.5f0 * cu.erf(0.707106781f0 * x)
end
function runbench(f, out, a, b, c, d, e)
    out .= f.(a, b, c, d, e)
    synchronize(out)
end



# add BenchmarkTools with Pkg.add("BenchmarkTools")
using BenchmarkTools
benchmarks = Dict()
for n in 1:7
    N = 10^n
    sptprice   = Float32[42.0 for i = 1:N]
    initStrike = Float32[40.0 + (i / N) for i = 1:N]
    rate       = Float32[0.5 for i = 1:N]
    volatility = Float32[0.2 for i = 1:N]
    spttime    = Float32[0.5 for i = 1:N]
    result     = similar(spttime)
    comparison = blackscholes.(sptprice, initStrike, rate, volatility, spttime)
    perbackend() do backend
        _sptprice = GPUArray(sptprice)
        _initStrike = GPUArray(initStrike)
        _rate = GPUArray(rate)
        _volatility = GPUArray(volatility)
        _time = GPUArray(spttime)
        _result = GPUArray(result)
        f = backend == :cudanative ? cu_blackscholes : blackscholes
        b = @benchmark runbench($f, $_result, $_sptprice, $_initStrike, $_rate, $_volatility, $_time)
        benches = get!(benchmarks, backend, [])
        push!(benches, b)
        @assert Array(_result) ≈ comparison
        # this is optional, but needed in a loop like this, which allocates a lot of GPUArrays
        # for the future, we need a way to tell the Julia gc about GPU memory
        free(_sptprice);free(_initStrike);free(_rate);free(_volatility);free(_time);free(_result);
    end
end
# Plot results:
# Pkg.add("Plots")
using Plots
benchmarks = benchy
labels = String.(keys(benchmarks))
times = map(values(benchmarks)) do v
    map(x-> minimum(x).time, v)
end

p2 = plot(
   times,
   m = (5, 0.8, :circle, stroke(0)),
   line = 1.5,
   labels = reshape(labels, (1, length(label))),
   title = "blackscholes",
   xaxis = ("10^N"),
   yaxis = ("Time in Seconds")
)

println("| Backend | Time in Seconds N = 10^7 |")
println("| ---- | ---- |")
for (l, nums) in zip(labels, times)
    println("| ", l, " | ", last(nums), " |")
end
