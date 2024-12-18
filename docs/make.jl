using Documenter, GPUArrays

function main()
    makedocs(
        modules = [GPUArrays],
        format = Documenter.HTML(
            # Use clean URLs on CI
            prettyurls = get(ENV, "CI", nothing) == "true",
            assets = ["assets/favicon.ico"],
            analytics = "UA-154489943-6",
        ),
        sitename = "GPUArrays.jl",
        pages = [
            "Home"          => "index.md",
            "Interface"     => "interface.md",
            "Functionality" => [
                "functionality/host.md",
                "functionality/device.md",
            ],
            "Test suite"    => "testsuite.md",
        ],
        doctest = true,
        warnonly=[:missing_docs],
    )

    deploydocs(
        repo = "github.com/JuliaGPU/GPUArrays.jl.git"
    )
end

isinteractive() || main()
