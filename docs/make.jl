using Documenter, GPUArrays

makedocs()

deploydocs(
    repo   = "github.com/JuliaGPU/GPUArrays.jl.git",
    julia  = "0.7",
    osname = "linux"
    # no need to build anything here, re-use output of `makedocs`
    deps   = nothing,
    make   = nothing,
)
