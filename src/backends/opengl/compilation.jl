using Sugar, GeometryTypes
using GLWindow, GLAbstraction, ModernGL, Reactive, GLFW, GeometryTypes
using FileIO
include("gl_codegen.jl")

immutable ComputeProgram{Args <: Tuple}
    program::GLProgram
    local_size::NTuple{3, Int}
end

const compiled_functions = Dict{Any, ComputeProgram}()
function add_deps!(io, deps, visited = Set())
    for dep in deps
        if !(dep in visited)
            str, dependencies = _module_cache[dep]
            push!(visited, dep)
            add_deps!(io, dependencies, visited)
            println(io, str)
        end
    end
end
function ComputeProgram{T}(f::Function, args::T)
    gltypes = to_glsl_types(args)
    get!(compiled_functions, (f, gltypes)) do # TODO make this faster
        io, funcargs, f_string = transpile(f, gltypes)
        local_size = (16, 16, 1)
        # add compute program dependant infos
        close(io.io)
        io.io = IOBuffer()
        print(io,
            "#version 430\n", # hardcode version for now #TODO don't hardcode :P
            "layout (local_size_x = 16, local_size_y = 16) in;\n", # same here
        )
        cache = get_module_cache()
        for typ in io.types
            println(io, typ)
        end
        visited = Set()
        add_deps!(io, io.dependencies)
        println(io)
        println(io, f_string)
        declare_global(io, funcargs)
        varnames = map(x-> string(global_identifier, x[2][1]), funcargs)
        print(io, "void main(){\n    ")
        show_name(io, f)
        print(io, "(", join(varnames, ", "), ");\n}")
        shader = Shader(Symbol(f), takebuf_array(io.io), GL_COMPUTE_SHADER)
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
        ceil(Int, size[i] / p.local_size[i])
    end
    glDispatchCompute(size[1], size[2], size[3])
    return
end
