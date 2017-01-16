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
else
    info("not testing cudanative backend")
end
if is_backend_supported(:julia)
    @testset "Threaded Julia backend" begin
        include("jlbackend.jl")
    end
else
    info("not testing julia backend")
end
if is_backend_supported(:opengl)
    @testset "OpenGL backend" begin
        include("opengl.jl")
    end
else
    info("not testing opengl backend")
end
