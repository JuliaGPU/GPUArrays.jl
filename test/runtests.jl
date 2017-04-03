#TODO register these packages!
for pkg in ("Sugar", "Transpiler")
    installed = try
        Pkg.installed(pkg) != nothing
    catch e
        false
    end
    installed || Pkg.clone("https://github.com/SimonDanisch/$(pkg).jl.git")
end

using GPUArrays
using Base.Test
srand(42) # set random seed for reproducability
function jltest(a, b)
    x = sqrt(sin(a) * b) / 10
    y = 33x + cos(b)
    y*10
end

macro allbackends(title, block)
    quote
        for backend in supported_backends()
            @testset "$($(esc(title))) $backend" begin
                ctx = GPUArrays.init(backend)
                $(esc(block))
            end
        end
    end
end

# Only test supported backends!
for backend in supported_backends()
    @testset "$backend" begin
        include("$(backend).jl")
    end
end
@testset "BLAS" begin
    include("blas.jl")
end

@testset "Shared" begin
    include("shared.jl")
end
