#TODO remove before merging
Pkg.checkout("Transpiler")
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
            if backend in (:opencl, :cudanative, :julia)
                @testset "$($(esc(title))) $backend" begin
                    ctx = GPUArrays.init(backend)
                    $(esc(backendname)) = backend
                    $(esc(block))
                end
                if backend == :cudanative
                    println("GPUMem: ", CUDAdrv.Mem.used() / 10^6)
                    gc()
                    println("    gc: ", CUDAdrv.Mem.used() / 10^6)
                end
            end
        end
    end
end

@testset "Broadcast" begin
    include("broadcast.jl")
end

# Only test supported backends!
for backend in supported_backends()
    if backend in (:opencl, :julia, :cudanative)
        @testset "$backend" begin
            include("$(backend).jl")
        end
        gc()
    end
end

@testset "BLAS" begin
    include("blas.jl")
end
gc()

@testset "Shared" begin
    include("shared.jl")
end
gc()
@testset "Array/Vector Operations" begin
    include("indexing.jl")
    include("vector.jl")
end
gc()
@testset "FFT" begin
    include("fft.jl")
end
gc()
