using Documenter, GPUArrays

makedocs()

deploydocs(
    deps   = Deps.pip("mkdocs", "python-markdown-math", "mkdocs-cinder"),
    repo   = "github.com/JuliaGPU/GPUArrays.jl.git",
    julia  = "0.6",
    osname = "linux"
)
