using GPUArrays
import GPUArrays: JLBackend, CLBackend
import GPUArrays.JLBackend: JLArray

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

# Somehow benchmark tools doesn't like this!
# so we create a super stupid benchmark macro ourselves
macro benchmark(expr)
    quote
        mintime = Inf
        for i=1:10
            tic()
            $(esc(expr))
            t = toq()
            mintime = min(t, mintime)
        end
        mintime
    end
end
cltimes = Float64[]
jltimes = Float64[]
threadtimes = Float64[]
for n in linspace(100, 10^7, 8)
    N = round(Int, n)
    sptprice   = Float32[ 42.0 for i = 1:N ];
    initStrike = Float32[ 40.0 + (i / N) for i = 1:N ];
    rate       = Float32[ 0.5 for i = 1:N ];
    volatility = Float32[ 0.2 for i = 1:N ];
    time       = Float32[ 0.5 for i = 1:N ];
    result = similar(time)
    #
    # ctx = CLBackend.init()
    # sptprice_gpu = GPUArray(sptprice)
    # initStrike_gpu = GPUArray(initStrike)
    # rate_gpu = GPUArray(rate)
    # volatility_gpu = GPUArray(volatility)
    # time_gpu = GPUArray(time)
    # result_gpu = GPUArray(result)

    ctx = JLBackend.init()
    sptprice_cpu = GPUArray(sptprice)
    initStrike_cpu = GPUArray(initStrike)
    rate_cpu = GPUArray(rate)
    volatility_cpu = GPUArray(volatility)
    time_cpu = GPUArray(time)
    result_cpu = GPUArray(result)
    if is_backend_supported(:opencl)
        bench_cl = @benchmark test(
            result_gpu,
            sptprice_gpu,
            initStrike_gpu,
            rate_gpu,
            volatility_gpu,
            time_gpu
        )
        push!(cltimes, bench_cl)
    end
    bench_thread = @benchmark test(
        result_cpu,
        sptprice_cpu,
        initStrike_cpu,
        rate_cpu,
        volatility_cpu,
        time_cpu
    )
    push!(threadtimes, bench_thread)
    # baseline
    bench_jl = @benchmark test(
        result,
        sptprice,
        initStrike,
        rate,
        volatility,
        time
    )
    push!(jltimes, bench_jl)

end
