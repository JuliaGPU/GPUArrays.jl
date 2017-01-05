using GLWindow, GLAbstraction, ModernGL, Reactive, GLFW, GeometryTypes
using FileIO
using Reactive, GeometryTypes
if !isdefined(:window)
const window = create_glcontext("Compute Shader", resolution = (512, 512))
end





prg = LazyShader(load(dir*"/test.comp"))
outtex = Texture(zeros(Float32, 512, 512))
glBindImageTexture(2, outtex.id, 0, GL_FALSE, 0, GL_WRITE_ONLY, outtex.internalformat)
jl_in = rand(Float32, 512, 512)
intex = Texture(jl_in)




function test(b)
    x = sqrt(sin(b*2.0) * b) / 10.0
    y = 33.0x + cos(b)
    if y == 77.0
        y = 0.0
    end
    y*10.0
end


ro.postrenderfunction = ()-> glDispatchCompute(div(512,16), div(512,16), 1) # 512^2 threads in blocks of 16^2
ro.vertexarray.program
GLAbstraction.render(ro)
glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
b = gpu_data(outtex)
a = test.(jl_in)
all(isapprox.(a, b))
