using Sugar, GeometryTypes
using GLWindow, GLAbstraction, ModernGL, Reactive, GLFW, GeometryTypes
using FileIO

include("rewrite_ast.jl")

immutable ComputeProgram{Args <: Tuple}
    program::GLProgram
    local_size::NTuple{3, Int}
end

const compiled_functions = Dict{Any, ComputeProgram}()
function resolve_dependencies!(dep::LazyMethod, visited = LazyMethod(Void, dep.transpiler), indent = 0)
    println("    "^indent, dep.signature)
    if dep in visited.dependencies
        # when already in deps we need to move it up!
        delete!(visited.dependencies, dep)
        push!(visited.dependencies, dep)
    else
        push!(visited, dep)
        if last(visited.dependencies) == dep
            resolve_dependencies!(dependencies!(dep), visited)
        end
    end
    visited.dependencies
end
function resolve_dependencies!(deps, visited, indent = 0)
    for dep in copy(deps)
        resolve_dependencies!(dep, visited, indent + 1)
    end
    visited.dependencies
end
function ComputeProgram{T}(f::Function, args::T; local_size = (16, 16, 1))
    gltypes = to_glsl_types(args)
    get!(compiled_functions, (f, gltypes)) do # TODO make this faster
        decl = LazyMethod((f, gltypes), Transpiler())
        funcsource = getfuncsource!(decl)
        # add compute program dependant infos
        io = GLSLIO(IOBuffer())
        print(io,
            "#version 430\n", # hardcode version for now #TODO don't hardcode :P
            "layout (local_size_x = 16, local_size_y = 16) in;\n", # same here
        )

        deps = reverse(collect(resolve_dependencies!(decl)))
        types = filter(istype, deps)
        funcs = filter(isfunction, deps)
        println(io, "// dependant type declarations")
        foreach(dep-> println(io, getsource!(dep)), types)
        println(io, "// dependant function declarations")
        foreach(dep-> println(io, getfuncsource!(dep)), funcs)

        println(io, "// Main inner function")
        println(io, funcsource)
        funcargs = getfuncargs(decl)
        declare_global(io, getfuncargs(decl))
        varnames = map(x-> string(global_identifier, x.args[1]), funcargs)
        print(io, "void main(){\n    ")
        show_name(io, f)
        print(io, "(", join(varnames, ", "), ");\n}")
        shader = GLAbstraction.compile_shader(take!(io.io), GL_COMPUTE_SHADER, Symbol(f))
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
