using GPUArrays
using Base.Test
import GPUArrays: GLBackend
import GLBackend: GLArray
using ModernGL
GLBackend.init()

function test(b)
    x = sqrt(sin(b*2.0) * b) / 10.0
    y = 33.0*x + cos(b)
    if y == 77.0
        y = 0.0
    end
    y*10.0
end
function test2(b)
    x = sqrt(sin(b*2.0) * b) / 10.0
    y = 33.0*x + cos(b)
    y*87.0
end
A = JLArray(rand(Float32, 512, 512));
out = JLArray(rand(Float32, 512, 512));
A = GLArray(rand(Float32, 512, 512));
out = GLArray(rand(Float32, 512, 512));
# TODO make already compiled functions persistent in modules
out .= (+).(A, 2.0)
out .= test2.(A)

_A = Array(A)
_out = zeros(Float32, 512, 512)
_out = (+).(_A, 2.0)
all(isapprox.(Array(out), _out))

function bench(A, out)
    out .= test2.(A)
end
using BenchmarkTools
a = @benchmark bench($A, $out)
b = @benchmark bench($_A, $_out)
judge(minimum(b), minimum(a))
minimum(a)
