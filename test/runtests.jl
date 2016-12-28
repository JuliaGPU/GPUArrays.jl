using GPUArrays
using Base.Test

function jltest(a, b)
    x = sqrt(sin(a) * b) / 10
    y = 33x + cos(b)
    y*10
end

if is_backend_supported(:cudanative)
    @testset "CUDAnative backend" begin
        include("cuda.jl")
    end
end
if is_backend_supported(:julia)
    @testset "Threaded Julia backend" begin
        include("jlbackend.jl")
    end
end
