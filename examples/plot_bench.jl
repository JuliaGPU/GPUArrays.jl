if Pkg.installed("BenchmarkTools") == nothing ||
   Pkg.installed("Query") == nothing ||
  # Pkg.installed("CUDAnative") == nothing ||
   Pkg.installed("DataFrames") == nothing

   error("Please install BenchmarkTools, Query, CUDAnative and DataFrames")
   # Pkg.add("Query")
   # Pkg.add("DataFrames")
end

using DataFrames, Query, Plots

labels = String[]
times = Vector{Float64}[]
cd(dirname(@__FILE__))
name = "blackscholes"
results = readtable("$(name)_results.csv")
for backend in unique(results[:Backend])
    push!(labels, string(backend))
    time = @from r in results begin
      @where r.Backend == backend
      @select r.minT
      @collect
   end
   push!(times, get.(time) ./ 10^9)
end

p2 = plot(
    times,
    m = (5, 0.8, :circle, stroke(0)),
    line = 1.5,
    labels = reshape(labels, (1, length(labels))),
    title = name,
    xaxis = "10^N",
    yaxis = "Time in s"
)
savefig("$name.png")
