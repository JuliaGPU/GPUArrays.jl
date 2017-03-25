# In Sugar.jl we rewrite the expressions so that they're
# closer to julias AST returned by macros.
# Here we rewrite the AST to make it easier to transpile, turning it into an
# invalid julia AST. The idea is to remove more logick from the actual printing
# to glsl string

using Sugar, DataStructures, GPUArrays
import GPUArrays: GLBackend
import Sugar: similar_expr, instance
include("intrinsics.jl")
include("gl_codegen.jl")


# constructors... #TODO refine input types
function rewrite_function{T <: gli.Types}(li, f::Type{T}, types::ANY, expr)
    expr.args[1] = glsl_name(T)
    true, expr
end

function rewrite_function{T <: gli.Vecs, I <: gli.int}(
        li, f::typeof(getindex), types::Type{Tuple{T, I}}, expr
    )
    # todo replace static indices
    idx = expr.args[3]
    if isa(idx, Integer) # if value is inlined
        field = (:x, :y, :z, :w)[idx]
        return true, Expr(:call, :getfield, expr.args[2], field)
    end
    true, expr
end

function rewrite_function{V1 <: gli.Vecs, V2 <: gli.Vecs}(
        li, f::Type{V1}, types::Type{Tuple{V2}}, expr
    )
    expr.args[1] = glsl_name(f) # substitute constructor
    true, expr
end

function rewrite_function(li, f::typeof(tuple), types::ANY, expr)
    expr.args[1] = glsl_name(expr.typ) # use returntype
    true, expr
end


function rewrite_function{F <: Function}(li, f::F, types::ANY, expr)
    intrinsic = gli.is_intrinsic(f, types)
    expr.args[1] = glsl_name(f)
    intrinsic, expr
end

function rewrite_function(li, ::typeof(broadcast), types::ANY, expr)
    F = types[1]
    if F <: gli.Functions && all(T-> T <: gli.Types, types[2:end])
        shift!(expr.args) # remove broadcast from call expression
        expr.args[1] = glsl_name(resolve_func(li, expr.args[1])) # rewrite if necessary
        return true, expr
    end
    # if all(x-> x <: gli.Types, args)
    #     ast = Sugar.sugared(f, types, code_lowered)
    #     funcs = extract_funcs(ast)
    #     types = extract_types(ast)
    #     if all(is_intrinsic, funcs) && all(x-> (x <: gli.Types), types)
    #         # if we only have intrinsic functions, numbers and vecs we can rewrite
    #         # this broadcast to be direct function calls. since the intrinsics will
    #         # already be broadcasted
    #         ast = glsl_ast(broadcast, types)
    #         false, :broadcast
    #     end
    # end
    false, expr
end

function glsl_rewrite_pass(li, expr)
    list = Sugar.replace_expr(expr) do expr
        if isa(expr, Slot)
            return true, slotname(li, expr)
        elseif isa(expr, QuoteNode)
            true, expr.value
        elseif isa(expr, Expr)
            args, head = expr.args, expr.head
            if head == :(=)
                lhs = args[1]
                name = slotname(li, lhs)
                rhs = map(x-> glsl_rewrite_pass(li, x), args[2:end])
                res = similar_expr(expr, [name, rhs...])
                if !(lhs in li.decls)
                    push!(li.decls, lhs)
                    decl = Expr(:(::), name, expr_type(li, lhs))
                    return true, (decl, res) # splice in declaration
                end
                return true, res
            elseif head == :call
                func = args[1]
                types = Tuple{map(x-> expr_type(li, x), args[2:end])...}
                 intrinsic, result, f = try
                    f = resolve_func(li, func)
                    intrinsic, result = rewrite_function(li, f, types, similar_expr(expr, args))
                    intrinsic, result, f
                catch e
                    println(STDERR, "Failed to resolve $func $types")
                    rethrow(e)
                end
                if !intrinsic
                    push!(li, (f, types))
                end
                map!(result.args, result.args) do x
                    glsl_rewrite_pass(li, x)
                end
                return true, result
            end
        end
        false, expr
    end
    first(list)
end



glsl_rewrite_pass(x, expr)

typename{T}(::Type{T}) = glsl_name(T)
global operator_replacement
let _operator_id = 0
    const operator_replace_dict = Dict{Char, String}()
    function operator_replacement(char)
        get!(operator_replace_dict, char) do
            _operator_id += 1
            string("op", _operator_id)
        end
    end
end


function typename{T <: Function}(::Type{T})
    x = string(T)
    x = replace(x, ".", "_")
    x = sprint() do io
        for char in x
            if Base.isoperator(Symbol(char))
                print(io, operator_replacement(char))
            else
                print(io, char)
            end
        end
    end
    glsl_name(x)
end
function declare_type(T)
    tname = typename(T)
    sprint() do io
        print(io, "struct ", tname, "{\n")
        fnames = fieldnames(T)
        if isempty(fnames) # structs can't be empty
            # we use bool as a short placeholder type.
            # TODO, are there cases where bool is no good?
            println(io, "bool empty;")
        else
            for name in fieldnames(T)
                FT = fieldtype(T, name)
                print(io, "    ", typename(FT))
                print(io, ' ')
                print(io, name)
                println(io, ';')
            end
        end
        println(io, "};")
    end
end

function getfuncheader!(x::LazyMethod, ::GLSLIO)
    if !isdefined(x, :funcheader)
        x.funcheader = sprint() do io
            args = getfuncargs(x)
            gio = GLSLIO(io)
            print(gio, typename(returntype(x)))
            print(gio, ' ')
            show_name(gio, x.signature[1])
            Base.show_enclosed_list(gio, '(', args, ", ", ')', 0, 0)
        end
    end
    x.funcheader
end
