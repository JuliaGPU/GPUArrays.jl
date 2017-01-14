using Sugar, GeometryTypes
using GLWindow, GLAbstraction, ModernGL, Reactive, GLFW, GeometryTypes
using FileIO
include("gl_codegen.jl")

immutable ComputeProgram{Args <: Tuple}
    program::GLProgram
    local_size::NTuple{3, Int}
end

const compiled_functions = Dict{Any, ComputeProgram}()

function ComputeProgram{T}(f::Function, args::T)
    gltypes = to_glsl_types(args)
    get!(compiled_functions, (f, gltypes)) do # TODO make this faster
        f_string = transpile(f, gltypes)
        local_size = (16, 16, 1)
        # add compute program dependant infos
        f_string = string(
            "#version 430\n", # hardcode version for now #TODO don't hardcode :P
            "layout (local_size_x = 16, local_size_y = 16) in;\n", # same here
            f_string
        )
        declare_global(glslio, funcargs)
        varnames = map(x-> string(global_identifier, x[2][1]), funcargs)
        print(glslio, "void main(){\n    ")
        show_name(glslio, f)
        print(glslio, "(", join(varnames, ", "), ");\n}")
        shader = Shader(Symbol(f), Vector{UInt8}(f_string), GL_COMPUTE_SHADER)
        program = GLAbstraction.compile_program([shader], [])
        ComputeProgram{T}(program, local_size)
    end::ComputeProgram{T}
end

useprogram(p::ComputeProgram) = glUseProgram(p.program.id)


function (p::ComputeProgram{Args}){Args}(args::Args, size::NTuple{3})
    useprogram(p)
    for i = 1:length(args)
        bindlocation(args[i], i-1)
    end
    size = ntuple(Val{3}) do i
        div(size[i], p.local_size[i])
    end
    glDispatchCompute(size[1], size[2], size[3])
    return
end
