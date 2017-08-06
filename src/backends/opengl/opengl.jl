module GLBackend

using ..GPUArrays

import GPUArrays: buffer, create_buffer, acc_broadcast!, synchronize, free
import GPUArrays: Context, GPUArray, context, broadcast_index, default_buffer_type

import GLAbstraction, GLWindow, GLFW
using ModernGL, StaticArrays
using Transpiler, Sugar
import Transpiler: gli, GLMethod, to_gl_types, glsl_gensym, GLIO
import Base: copy!, convert

const gl = GLAbstraction


immutable GLContext <: Context
    # There are sadly many types of contexts from different packages.
    # We can't add those packages as a dependency, just to type this field
    window
    program_cache::Dict{Any, Any}
end
GLContext(window) = GLContext(window, Dict{Any, Any}())
Base.show(io::IO, ctx::GLContext) = print(io, "GLContext")

const GLBuffer{T, N} = GPUArray{T, N, gl.GLBuffer{T}, GLContext}
const GLSampler{T, N} = GPUArray{T, N, gl.Texture{T, N}, GLContext}
const GLArray{T, N, Buffer} = GPUArray{T, N, Buffer, GLContext}




global all_contexts, current_context, init, any_context
let contexts = GLContext[]
    function any_context()
        if isempty(contexts)
            window = GLWindow.create_glcontext(major = 3, minor = 3, visible = false)
            GLContext(window)
        else
            last(contexts) # TODO check if is open (all context creating library need to implement isopen first)
        end
    end
    all_contexts() = copy(contexts)::Vector{GLContext}
    current_context() = last(contexts)::GLContext
    function init(; ctx = any_context())
        init(ctx)
    end
    init(ctx) = init(GLContext(ctx))
    function init(ctx::GLContext)
        GPUArrays.make_current(ctx)
        push!(contexts, ctx)
        ctx
    end
end



#synchronize
function synchronize(x::GLArray)
    # TODO figure out more fine grained solutions
    glFinish()
end

function free(x::GLArray)
    synchronize(x)
    gl.free(buffer(x))
end

function default_buffer_type{T, N}(
        ::Type, ::Type{Tuple{T, N}}, ::GLContext
    )
    gl.GLBuffer{T}
end
function default_buffer_type{T, N}(
        ::Type{<: GPUArray{FT, FN, gl.Texture{FT, FN}} where {FT, FN}},
        ::Type{Tuple{T, N}}, ::GLContext
    )
    gl.Texture{T, N}
end
# default_buffer_type{T, N}(::Tuple{T, 2}, ::GLContext) = gl.GLBuffer{T, 1}
# default_buffer_type{T, N}(::Tuple{T, 3}, ::GLContext) = gl.GLBuffer{T, 1}


function (AT::Type{GLArray{T, N, Buffer}}){T, N, Buffer <: gl.GLBuffer}(
        size::NTuple{N, Int};
        context = current_context(),
        usage = GL_STATIC_READ, kw_args...
    )
    buff = gl.GLBuffer(T, prod(size); usage = usage, kw_args...)
    AT(buff, size, context)
end

function (AT::Type{GLArray{T, N, Buffer}}){T, N, Buffer <: gl.Texture}(
        size::NTuple{N, Int};
        context = current_context(), kw_args...
    )
    tex = gl.Texture(T, size; kw_args...)
    AT(tex, size, context)
end

function Base.convert{ET, ND}(
        ::Type{GLSampler{ET, ND}},
        A::GLBuffer{ET, ND}
    )
    texB = GLSampler{ET, ND}(size(A), context = context(A))
    copy!(buffer(texB), buffer(A))
    texB
end
function copy!{T, N}(
        dest::GLSampler{T, N}, dest_range::CartesianRange{CartesianIndex{N}},
        src::Array{T, N}, src_range::CartesianRange{CartesianIndex{N}},
    )
    copy!(buffer(dest), dest_range, src, src_range)
end

################################################################################
# Broadcast

include("glutils.jl")

immutable TransformFeedbackProgram{T, VBO, UNIFORMS, SAMPLERS, N}
    program::GLuint
    uniforms::Nullable{UniformBuffer{UNIFORMS, N}}
end


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


function (tfp::TransformFeedbackProgram{T}){T}(
        A::GLBuffer{T}, args
    )
    buffer_loc = 0
    uniforms = []
    glUseProgram(tfp.program)
    vao = Ref{GLuint}()
    glGenVertexArrays(1, vao)
    glBindVertexArray(vao[])
    for elem in args
        if isa(elem, GLBuffer)
            buff = buffer(elem)
            GLAbstraction.bind(buff)
            glEnableVertexAttribArray(buffer_loc)
            glVertexAttribPointer(
                buffer_loc,
                GLAbstraction.cardinality(buff),
                GLAbstraction.julia2glenum(eltype(buff)),
                GL_FALSE, 0, C_NULL
            )
            buffer_loc += 1
        else
            push!(uniforms, elem)
        end
    end
    if !isnull(tfp.uniforms)
        get(tfp.uniforms)[] = (uniforms...,)
    end
    glEnable(GL_RASTERIZER_DISCARD)
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, buffer(A).id)
    glBeginTransformFeedback(GL_POINTS)
    glDrawArrays(GL_POINTS, 0, length(A))
    glEndTransformFeedback()
    glBindVertexArray(0)
    nothing
end


function TransformFeedbackProgram{T}(context, kernel, out::Type{T}, args)
    key = (Symbol(kernel), (out, args...))
    get!(context.program_cache, key) do
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
            if elem <: Ref
                elem = eltype(elem) # unpack
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
        for (name, t) in vbos
            print(io, "layout (location = $idx) in ")
            Transpiler.show_type(io, t)
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
        slotstart = length(Sugar.slottypes(m))
        println(io, "void main(){")
            print(io, "    $outsym = ")
            fname = Symbol(Sugar.functionname(io, m.signature...))
            func_prec = Base.operator_precedence(fname)
            if func_prec > 0
                print(io, join(accessors, " $fname "))
                println(io, ";")
            else
                print(io, fname)
                print(io, '(')
                print(io, join(accessors, ", "))
                println(io, ");")
            end
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
            println(GLAbstraction.getinfolog(program))
        end

        buffer = if isempty(uniform_types)
            et = Tuple{}; n = 0
            Nullable{UniformBuffer{et, n}}()
        else
            et = Tuple{uniform_types...}; n = 1
            Nullable(UniformBuffer(Tuple{uniform_types...}, 1))
        end
        TransformFeedbackProgram{T, Tuple{last.(vbos)...}, et, Void, n}(program, buffer)
    end
end


function acc_broadcast!(kernel, A::GLBuffer, args)
    typs = map(typeof, args)
    tfp = TransformFeedbackProgram(context(A), kernel, eltype(A), typs)
    tfp(A, args)
end



# implement BLAS backend

end

using .GLBackend
export GLBackend
#
# using ModernGL, GLAbstraction, GLWindow
# w = create_glcontext()
#
# vertex_src = """
# #version 330
# // dependant type declarations
# // dependant function declarations
# // vertex input:
# layout (location = 0) in float _gensymed_localarg1;
# layout (location = 1) in float _gensymed_localarg2;
# // uniform inputs:
#
# out float _gensymed_output;
#
# float test(float a, float b)
# {
#     float y;
#     float x;
#     x = sqrt(sin(a) * b) / float(10.0);
#     y = float(33.0) * x + cos(b);
#     return y * float(10.0);
# }
#
# // vertex main function:
# void main(){
#     _gensymed_output = test(_gensymed_localarg1, _gensymed_localarg2);
# }
# """
# shader = GLAbstraction.compile_shader(Vector{UInt8}(vertex_src), GL_VERTEX_SHADER, :test).id
# GLAbstraction.iscompiled(shader)
#
#
# program = glCreateProgram()
# glAttachShader(program, shader)
# feedbackVaryings = ["_gensymed_output"]
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
# glEnableVertexAttribArray(0)
# glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, C_NULL)
#
# data2 = rand(Float32, 1000)
# vbo2 = Ref{GLuint}()
# glGenBuffers(1, vbo2);
# glBindBuffer(GL_ARRAY_BUFFER, vbo2[])
# glBufferData(GL_ARRAY_BUFFER, sizeof(data2), data2, GL_STATIC_DRAW)
# glEnableVertexAttribArray(1)
# glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, C_NULL)
#
# dest = GLBuffer(Float32, 1000, usage = GL_STATIC_READ);
#
# glEnable(GL_RASTERIZER_DISCARD)
# glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, dest.id)
# glBeginTransformFeedback(GL_POINTS)
# glDrawArrays(GL_POINTS, 0, length(dest))
# glEndTransformFeedback()
# glBindVertexArray(0)
# glFinish()
# x = GLAbstraction.gpu_data(dest)
#
# test.(data, data2) ≈ x
