module GLBackend

using ..GPUArrays

import GPUArrays: buffer, create_buffer, acc_broadcast!
import GPUArrays: Context, GPUArray, context, broadcast_index

import GLAbstraction, GLWindow, GLFW
using ModernGL, Compat
using Transpiler, Sugar
import Transpiler: gli, GLMethod, to_glsl_types, glsl_gensym, GLIO

const gl = GLAbstraction

immutable GLContext <: Context
    window::GLFW.Window
end
Base.show(io::IO, ctx::GLContext) = print(io, "GLContext")

@compat const GLBuffer{T, N} = GPUArray{T, N, gl.GLBuffer{T}, GLContext}
@compat const GLSampler{T, N} = GPUArray{T, N, gl.Texture{T, N}, GLContext}
@compat const GLArray{T, N} = Union{GLBuffer{T, N}, GLSampler{T, N}}


function any_context()
    window = GLWindow.create_glcontext(major = 4, minor = 3, visible = false)
    GLContext(window)
end

global all_contexts, current_context, init
let contexts = GLContext[]
    all_contexts() = copy(contexts)::Vector{GLContext}
    current_context() = last(contexts)::GLContext
    function init(; ctx = any_context())
        init(ctx)
    end
    init(ctx::GLWindow.Screen) = init(GLContext(GLWindow.nativewindow(ctx)))
    function init(ctx::GLContext)
        GPUArrays.make_current(ctx)
        push!(contexts, ctx)
        ctx
    end
end

function create_buffer{T, N}(
        ctx::GLContext, ::Type{T}, sz::NTuple{N, Int};
        usage = GL_STATIC_READ, kw_args...
    )
    gl.GLBuffer(T, prod(sz); kw_args...)
end

function glTexSubImage{N}(tex, offset::NTuple{N, Int}, width::NTuple{N, Int}, data)
    glfun = N == 1 ? glTexSubImage1D : N == 2 ? glTexSubImage2D : N == 3 ? glTexSubImage3D : error("Dim $N not supported")
    glfun(tex.texturetype, 0, offset..., width..., tex.format, tex.pixeltype, data)
end

function Base.convert{ET, ND}(
        ::Type{GLSampler{ET, ND}},
        A::GLBuffer{ET, ND}
    )
    texB = GPUArray(gl.Texture(ET, size(A)), size(A), context(A))
    copy!(texB, A)
    texB
end


################################################################################
# Broadcast



function bindlocation{T, N}(A::GLSampler{T, N}, i)
    t = buffer(A)
    glBindImageTexture(i, t.id, 0, GL_FALSE, 0, GL_READ_WRITE, t.internalformat)
end

# TODO integrate buffers
# function bind(t::GLBuffer, i)
#     glBindImageTexture(i, t.id, 0, GL_FALSE, 0, GL_READ_WRITE, t.internalformat)
# end

function bindlocation(t, i)
    gluniform(i, t)
end
#Functions will be declared in the shader as a constant, so we don't need to bind them
function bindlocation(t::Function, i)
end

#
# immutable TransformFeedbackProgram{T, VBO, UNIFORMS, SAMPLERS}
#     program::GLuint
#     uniforms::UniformBuffer{UNIFORMS}
# end
#
# function (tfp::TransformFeedbackProgram{VBO, UNIFORMS, SAMPLERS}){VBO, UNIFORMS, SAMPLERS}(
#         out::GLBuffer{T}, vbo::VertexArray{VBO}, uniforms::UNIFORMS, samplers
#     )
#     tfp.uniforms[] = uniforms # update args
#     glEnable(GL_RASTERIZER_DISCARD)
#     glUseProgram(tfp.program)
#     glBindVertexArray(vbo.id)
#     glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, out.id)
#     glBeginTransformFeedback(GL_POINTS)
#     glDrawArrays(GL_POINTS, 0, length(out))
#     glEndTransformFeedback()
#     glBindVertexArray(0)
# end


function broadcast_shader{T}(kernel, out::Type{T}, args)
    accessors = []
    uniforms = []
    samplers = []
    vbos = []
    argtypes = []
    names = []
    # figure out how to access the args
    for (i, elem) in enumerate(args)
        name = "arg$i"
        globalsym = glsl_gensym(string("local", name))
        push!(names, globalsym)
        accessor = globalsym
        argtype = elem
        wasref = false
        if isa(elem, Ref)
            elem = elem[] # unpack
            wasref = true
        end
        if elem <: GLBuffer
            push!(vbos, globalsym => eltype(elem))
            if !wasref # already set otherwise
                argtype = eltype(elem)
            end
        elseif isa(elem, GLSampler)
            error("Sampler not implemented right now!")
            # push!(samplers, elem)
            # push!(accessors, globalsym)
        else
            isbits(elem) || error("Only Ref, isbits Scalar, GLSampler and GLBuffer are supported for opengl broadcast. Found: $(typeof(elem))")
            push!(uniforms, globalsym => elem)
        end
        push!(accessors, accessor)
        push!(argtypes, argtype)
    end

    m = GLMethod((kernel, Tuple{argtypes...,}))

    io = GLIO(IOBuffer(), m)

    println(io, "#version 330")
    Transpiler.print_dependencies(io, m)

    # # Vertex in block
    println(io, "// vertex input:")
    idx = 0
    for (name, T) in vbos
        print(io, "layout (location = $idx) in ")
        Transpiler.show_type(io, T)
        print(io, ' ')
        Transpiler.show_name(io, name)
        println(io, ';')
        idx += 1
    end
    uniform_types = last.(uniforms)
    Transpiler.show_uniforms(io, first.(uniforms), uniform_types)
    # print out value
    outsym = glsl_gensym("output")
    println(io)
    print(io, "out ")
    Transpiler.show_type(io, eltype(T))
    println(io, " $outsym;")
    println(io)
    if !Sugar.isintrinsic(m) # broadcastet function might be intrinsic
        println(io, Sugar.getsource!(m))
    end
    println(io)
    println(io, "// vertex main function:")
    println(io, "void main(){")
        print(io, "    $outsym = ")
        Transpiler.show_function(io, m.signature...)
        print(io, '(')
        print(io, join(accessors, ", "))
        println(io, ");")
    println(io, '}')
    vsource = take!(io.io)
    vshader = GLAbstraction.compile_shader(vsource, GL_VERTEX_SHADER, Symbol(kernel))
    program = glCreateProgram()
    glAttachShader(program, vshader.id)
    outvar = ["$outsym"]
    glTransformFeedbackVaryings(program, 1, outvar, GL_INTERLEAVED_ATTRIBS)
    glLinkProgram(program)
    if !GLAbstraction.islinked(program)
        println("ERROR IN SHADER: ")
        write(STDOUT, vsource)
        println(getinfolog(program))
    else
        println("SUCCESSS")
    end
    buffer = UniformBuffer(Tuple{uniform_types...}, 1)
    TransformFeedbackProgram(program, buffer)
end


function acc_broadcast!(kernel, A::GLBuffer, args)
    inputAttrib = glGetAttribLocation(program, "inValue")
    glEnableVertexAttribArray(inputAttrib)
    glVertexAttribPointer(inputAttrib, 1, GL_FLOAT, GL_FALSE, 0, C_NULL)
end



# implement BLAS backend


end

using .GLBackend
export GLBackend

# using ModernGL, GLAbstraction, Visualize
# w = GLFWWindow()
#
# vertex_src = """
# #version 130
# layout(location = 0) in float inValue;
# out float outValue;
#
# void main()
# {
#     outValue = sqrt(inValue);
# }
# """
# shader = GLAbstraction.compile_shader()
# glShaderSource(shader, 1, Vector{UInt8}[vertex_src], C_NULL)
# glCompileShader(shader)
# GLAbstraction.iscompiled(shader)
#
#
# program = glCreateProgram()
# glAttachShader(program, shader)
# feedbackVaryings = ["outValue"]
# glTransformFeedbackVaryings(program, 1, feedbackVaryings, GL_INTERLEAVED_ATTRIBS)
#
# glLinkProgram(program)
# GLAbstraction.islinked(program)
#
# glUseProgram(program)
# vao = Ref{GLuint}()
# glGenVertexArrays(1, vao)
# glBindVertexArray(vao[])
#
# data = rand(Float32, 1000)
# vbo = Ref{GLuint}()
# glGenBuffers(1, vbo);
# glBindBuffer(GL_ARRAY_BUFFER, vbo[])
# glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW)
#
# inputAttrib = glGetAttribLocation(program, "inValue")
# glEnableVertexAttribArray(inputAttrib)
# glVertexAttribPointer(inputAttrib, 1, GL_FLOAT, GL_FALSE, 0, C_NULL)
#
#
# dest = GLBuffer(Float32, 1000, usage = GL_STATIC_READ)
