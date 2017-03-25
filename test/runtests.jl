using JTensors
using Base.Test

function jltest(a, b)
    x = sqrt(sin(a) * b) / 10
    y = 33x + cos(b)
    y*10
end

# Only test supported backends!
for backend in supported_backends()
    @testset "$backend" begin
        include("$(backend).jl")
    end
end
