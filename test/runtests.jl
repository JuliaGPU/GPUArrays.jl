

if get(ENV, "TRAVIS", "") == "true" ||
        get(ENV, "APPVEYOR", "") == "true" ||
        get(ENV, "CI", "") == "true"

    cd(()-> run(`git checkout sd/applyvarargstack`), Pkg.dir("Sugar"))
    cd(()-> run(`git checkout sd/for`), Pkg.dir("Transpiler"))

end




using GPUArrays
using Base.Test
srand(42) # set random seed for reproducability
function jltest(a, b)
    x = sqrt(sin(a) * b) / 10f0
    y = (33f0)x + cos(b)
    y*10f0
end

macro allbackends(title, backendname::Symbol, block)
    quote
        for backend in supported_backends()
            if backend in (:opencl, :cudanative)
                @testset "$($(esc(title))) $backend" begin
                    ctx = GPUArrays.init(backend)
                    $(esc(backendname)) = backend
                    $(esc(block))
                end
            end
        end
    end
end

# Only test supported backends!
for backend in supported_backends()
    if backend in (:opencl, :cudanative)
        @testset "$backend" begin
            include("$(backend).jl")
        end
    end
end
@testset "BLAS" begin
    include("blas.jl")
end

@testset "Shared" begin
    include("shared.jl")
end

@testset "Array/Vector Operations" begin
    include("indexing.jl")
    include("vector.jl")
end
