# you might need to add SpecialFunctions with Pkg.add("SpecialFunctions")

if Pkg.installed("BenchmarkTools") == nothing ||
   Pkg.installed("Query") == nothing ||
   Pkg.installed("CUDAnative") == nothing ||
   Pkg.installed("DataFrames") == nothing

   error("Please install BenchmarkTools, Query, CUDAnative and DataFrames")
end

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


using BenchmarkTools
import BenchmarkTools: Trial
using DataFrames
using Query

Nmax = 7

benchmarks = DataFrame([Symbol, Int64, Trial, Float64], [:Backend, :N, :Trial, :minT], 0)

NT = Base.Threads.nthreads()
info("Running benchmarks number of threads: $NT")

for n in 1:Nmax
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
        push!(benchmarks, (backend, N, b, minimum(b).time))

        @assert Array(_result) ≈ comparison
        # this is optional, but needed in a loop like this, which allocates a lot of GPUArrays
        # for the future, we need a way to tell the Julia gc about GPU memory
        free(_sptprice);free(_initStrike);free(_rate);free(_volatility);free(_time);free(_result);
    end
end

results = @from b in benchmarks begin
   @select {b.Backend, b.N, b.minT}
   @collect DataFrame
end

writetable("benchmark_results_$(NT).csv", results)

function filterResults(df, n)
   dfR = @from r in df begin
      @where r.N == 10^n
      @select {r.Backend, r.minT}
      @collect DataFrame
   end

   return dfR
end

io = IOBuffer()
for n in 1:Nmax
   df = filterResults(results, n)
   write(io, "| Backend | Time in Seconds N = 10^$Nmax |", "\n")
   write(io, "| ---- | ---- |", "\n")
   for row in eachrow(df)
      b = row[:Backend]
      t = row[:minT]
      write(io, "| ", b, " | ", t, " |", "\n")
   end
   display(Markdown.parse(io))
end

# Plot results:
@static if VERSION < v"0.6.0-dev" &&
           Pkg.installed("Plots") != nothing
   using Plots

   df7 = filterResults(results, 7)
   labels = df7[:Backend]
   times = df7[:minT]

   p2 = plot(
      times,
      m = (5, 0.8, :circle, stroke(0)),
      line = 1.5,
      labels = reshape(labels, (1, length(label))),
      title = "blackscholes",
      xaxis = ("10^N"),
      yaxis = ("Time in Seconds")
   )
end
