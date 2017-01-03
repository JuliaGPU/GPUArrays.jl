using Sugar, GeometryTypes
include("../src/backends/opengl/gl_codegen.jl")

const prescripts = Dict(
    Float32 => "",
    Float64 => "",
    Int => "i",
    Int32 => "i",
    UInt => "ui",
)
function glsl_hygiene(sym)
    x = string(sym)
    x = replace(x, "#", "_")
end

function show_name(io::GLSLIO, x::Function)
    # TODO
    print(io, glsl_hygiene(Symbol(x)))
end
function show_name{N, T}(io::GLSLIO, x::Type{Vec{N, T}})
    print(io, prescripts[T])
    print(io, "vec", N)
end
function show_name{T, N}(io::GLSLIO, x::Type{GLArray{T, N}})
    if !(N in (1, 2, 3))
        # TODO, fake ND arrays with 1D array
        error("GPUArray can't have more than 3 dimensions for now")
    end
    print(io, "image$(N)D /* $T \\*")
end
function show_name(io::GLSLIO, x::Type{Void})
    print(io, "void")
end
function show_name(io::GLSLIO, x::Type{Float64})
    print(io, "float")
end
function show_name(io::GLSLIO, x::DataType)
    xname = typename(x)
    if !get(io.vardecl, x, false)
        push!(io.dependencies, sprint() do io
            print(io, "struct ", xname, '{')
            fnames = fieldnames(x)
            if isempty(fnames) # structs can't be empty
                # we use bool as a short placeholder type.
                # TODO, are there corner cases where bool is no good?
                print(io, "bool empty;")
            else
                for name in fieldnames(x)
                    T = fieldtype(x, name)
                    print(io, typename(T))
                    print(io, ' ')
                    print(io, name)
                    println(io, ';')
                end
            end
            println(io, "};")
        end)
        io.vardecl[x] = true
    end
    print(io, xname)
end
function show_name(io::GLSLIO, x::Union{AbstractString, Symbol})
    print(io, x)
end
function materialize_io(x::GLSLIO)
    result_str = ""
    for str in x.dependencies
        result_str *= str * "\n"
    end
    string(result_str, '\n', takebuf_string(x.io))
end

function transpile(f, typs, parentio = nothing)
    ast = Sugar.sugared(f, typs, code_typed)
    li = Sugar.get_lambda(code_typed, f, typs)
    slotnames = Base.lambdainfo_slotnames(li)
    ret_type = Sugar.return_type(f, typs)
    glslio = GLSLIO(IOBuffer(), li, slotnames)

    show_name(glslio, ret_type)
    print(glslio, ' ')
    show_name(glslio, f)
    print(glslio, '(')
    vars = Sugar.slot_vector(li)
    for (i, (slot, (name, T))) in enumerate(vars[2:li.nargs])
        glslio.vardecl[slot] = true
        show_name(glslio, T)
        print(glslio, ' ')
        show_name(glslio, name)
        i != (li.nargs - 1) && print(glslio, ", ")
    end
    print(glslio, ')')
    slots = filter(vars[(li.nargs+1):end]) do decl
        var = decl[1]
        if isa(var, Slot)
            glslio.vardecl[var] = true
            true
        else
            false
        end
    end

    body = Expr(:body, map(x->NewvarNode(x[1]), slots)..., ast.args...);
    show_unquoted(glslio, body, 2, 0)
    if parentio != nothing
        push!(parentio.dependencies, materialize_io(glslio))
    else
        return materialize_io(glslio)
    end
end
if !isdefined(:gl)
const gl = GLSLIntrinsics
end

function broadcastkernel{F <: Function}(f::F, out, A)
    idx = Vec{2, Int}(gl.GlobalInvocationID())
    out[idx] = f(A[idx])
    return
end
function Base.broadcast!{F <: Function, T}(f::F, A::GLArray{T, 2}, B::GLArray{T, 2})
    func = transpile(broadcastkernel, (F, GLArray{T, 2}, GLArray{T, 2}))
    println(func)
end
function test(b)
    x = sqrt(sin(b*2.0) * b) / 10.0
    y = 33.0x + cos(b)
    y*10.0
end

A = GLArray{Float64, 2}()
B = GLArray{Float64, 2}()

A .= test.(B)
