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
# TODO make already compiled functions persistent in modules
# TODO reuse those modules
out .= (+).(A, 2.0)

ComputeProgram(broadcast_kernel, (A, f, args...))
typs = GLBackend.to_glsl_types((A, +, out, 2.0))
typs


glslio, funcargs = GLBackend.transpile(
    GLBackend.broadcast_kernel, typs
)
local_size = (16, 16, 1)
# add compute program dependant infos
io = IOBuffer()
print(io,
    "#version 430\n", # hardcode version for now #TODO don't hardcode :P
    "layout (local_size_x = 16, local_size_y = 16) in;\n", # same here
)
declare_global(io, funcargs)
varnames = map(x-> string(global_identifier, x[2][1]), funcargs)
print(io, "void main(){\n    ")
show_name(io, f)
print(io, "(", join(varnames, ", "), ");\n}")
shader = Shader(Symbol(f), takebuf_array(io), GL_COMPUTE_SHADER)
program = GLAbstraction.compile_program([shader], [])
ComputeProgram{T}(program, local_size)
