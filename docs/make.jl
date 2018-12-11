using Documenter, GPUArrays

makedocs(
    modules = [GPUArrays],
    format = Documenter.HTML(),
    sitename = "GPUArrays.jl",
    pages = [
        "Home" => "index.md",
    ],
    doctest = true
)

deploydocs(
    repo   = "github.com/JuliaGPU/GPUArrays.jl.git"
)
