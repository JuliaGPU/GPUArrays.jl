using GPUArrays
using Base.Test
import GPUArrays: GLBackend
import GLBackend: GLArray
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
A = GLArray(rand(Float32, 512, 512));
out = GLArray(rand(Float32, 512, 512));
out .= (+).(A, 2.0)
