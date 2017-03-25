using JTensors, CUDAnative, BenchmarkTools
import JTensors: CUBackend, JLBackend
import CUBackend: CUArray
import JLBackend: JLArray
cuctx = CUBackend.init()
jlctx = JLBackend.init()
const cu = CUDAnative

function jltest(a, b)
    x = sqrt(sin(a) * b) / 10
    y = 33x + cos(b)
    y*10
end
function cutest(a, b)
    x = cu.sqrt(cu.sin(a) * b) / 10
    y = 33x + cu.cos(b)
    y*10
end

function jlbench(out, A, B)
    out .= A .* jltest.(B, A) .+ 77f0
end

function cubench(out, A, B)
    out .= A .* cutest.(B, A) .+ 77f0
end

for n = 20:50:300
    for T in (Float32, Float64)
        A1 = rand(T, n, n)
        B1 = rand(T, n, n)
        out1 = similar(A1)

        A2 = JLArray(A1)
        B2 = JLArray(B1)
        out2 = similar(A2)

        A3 = CUArray(A1)
        B3 = CUArray(B1)
        out3 = similar(A3)

        b1 = @benchmark jlbench(out1, A1, B1)
        b2 = @benchmark jlbench(out2, A2, B2)
        b3 = @benchmark cubench(out3, A3, B3)
    end
end
