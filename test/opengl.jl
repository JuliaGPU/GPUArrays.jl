using GPUArrays, ModernGL
import GLAbstraction
import GPUArrays: GLBackend
import GLBackend: GLArrayTex, GLArrayBuff
GLBackend.init()
a = GPUArrays.GPUArray(rand(Float32, 10, 10))
tex = convert(GLArrayTex{Float32, 2}, a)
Array(a) == Array(tex)
