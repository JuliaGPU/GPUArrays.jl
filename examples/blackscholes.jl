# you might need to add SpecialFunctions with Pkg.add("SpecialFunctions")

if Pkg.installed("BenchmarkTools") == nothing ||
   Pkg.installed("Query") == nothing ||
  # Pkg.installed("CUDAnative") == nothing ||
   Pkg.installed("DataFrames") == nothing

   error("Please install BenchmarkTools, Query, CUDAnative and DataFrames")
   # Pkg.add("Query")
   # Pkg.add("DataFrames")
end
using GPUArrays, SpecialFunctions
using GPUArrays: perbackend, synchronize, free
using SpecialFunctions: erf

function blackscholes(
        sptprice,
        strike,
        rate,
        volatility,
        time
    )
    logterm = log( sptprice / strike)
    powterm = .5f0 * volatility * volatility
    den = volatility * sqrt(time)
    d1 = (((rate + powterm) * time) + logterm) / den
    d2 = d1 - den
    NofXd1 = cndf2(d1)
    NofXd2 = cndf2(d2)
    futureValue = strike * exp(-rate * time)
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

if Pkg.installed("CUDAnative") != nothing
    using CUDAnative
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
end
function runbench(f, out, a, b, c, d, e)
    out .= f.(a, b, c, d, e)
    synchronize(out)
end


using BenchmarkTools
import BenchmarkTools: Trial
using DataFrames
using Query

# on 0.6:
# Pkg.clone("https://github.com/davidanthoff/IterableTables.jl.git")
# Pkg.checkout("NamedTuples")
# Pkg.checkout("Query")
# Pkg.checkout("SimpleTraits")

Nmax = 7

benchmarks = DataFrame([String, Int64, Trial, Float64], [:Backend, :N, :Trial, :minT], 0)

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
    perbackend() do backend # nice to go through all backends, but for benchmarks we might want to have this more explicit!
        if true #backend == :julia
            _sptprice = GPUArray(sptprice)
            _initStrike = GPUArray(initStrike)
            _rate = GPUArray(rate)
            _volatility = GPUArray(volatility)
            _time = GPUArray(spttime)
            _result = GPUArray(result)
            f = backend == :cudanative ? cu_blackscholes : blackscholes
            b = @benchmark runbench($f, $_result, $_sptprice, $_initStrike, $_rate, $_volatility, $_time)
            ctx_str = sprint() do io
                show(io, GPUArrays.current_context())
            end
            push!(benchmarks, (ctx_str, N, b, minimum(b).time))

            @assert Array(_result) ≈ comparison
            # this is optional, but needed in a loop like this, which allocates a lot of GPUArrays
            # for the future, we need a way to tell the Julia gc about GPU memory
            free(_sptprice);free(_initStrike);free(_rate);free(_volatility);free(_time);free(_result);
        end
    end
end

results = @from b in benchmarks begin
   @select {b.Backend, b.N, b.minT}
   @collect DataFrame
end

results = cd(dirname(@__FILE__)) do
    file = "blackscholes_results.csv"
    # merge
    merged = if isfile(file)
        merged = readtable(file)
        backends = unique(merged[:Backend])
        benched_backends = unique(results[:Backend])
        to_add = setdiff(benched_backends, backends)
        for i in 1:size(results, 1)
            row = results[i, :]
            if row[:Backend][1] in to_add
                append!(merged, row)
            end
        end
        merged
    else
        results
    end
    writetable(file, merged)
    merged
end

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
   println(io, "| Backend | Time (μs) for N = 10^$n |")
   println(io, "| ---- | ---- |")
   for row in eachrow(df)
      b = row[:Backend]
      t = row[:minT]
      @printf(io, "| %s | %6.2f μs|\n", b, t / 10^9)
   end
   display(Markdown.parse(io))
   seekstart(io)
   println(String(take!(io)))
end
