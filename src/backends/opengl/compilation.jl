using Sugar, GeometryTypes
using GLWindow, GLAbstraction, ModernGL, Reactive, GLFW, GeometryTypes
using FileIO

include("rewrite_ast.jl")

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
function ComputeProgram{T}(f::Function, args::T; local_size = (16, 16, 1))
    gltypes = to_glsl_types(args)
    get!(compiled_functions, (f, gltypes)) do # TODO make this faster
        decl = Decl((f, gltypes), Transpiler())

        # add compute program dependant infos
        io = GLSLIO(IOBuffer())
        print(io,
            "#version 430\n", # hardcode version for now #TODO don't hardcode :P
            "layout (local_size_x = 16, local_size_y = 16) in;\n", # same here
        )
        dependant_types = filter(istype, decl.dependencies)
        dependant_funcs = filter(isfunction, decl.dependencies)
        println(io, "// type declarations")
        for typ in dependant_types
            println(io, getsource!(typ))
        end
        println(io, "// function declarations")
        for func in dependant_funcs
            println(io, getfuncsource!(func))
        end
        println(io, "// Main inner function")
        println(io, getfuncsource!(decl))
        funcargs = getfuncargs(decl)
        declare_global(io, getfuncargs(decl))

        varnames = map(x-> string(global_identifier, x.args[1]), funcargs.args)
        print(io, "void main(){\n    ")
        show_name(io, f)
        print(io, "(", join(varnames, ", "), ");\n}")
        shader = Shader(Symbol(f), String(take!(io.io)), GL_COMPUTE_SHADER)
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
