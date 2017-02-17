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
        return true, Expr(:call, :getfield, expr.args[2], QuoteNode(field))
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


resolve_func{T}(li, ::Type{T}) = T
resolve_func(li, f::Union{GlobalRef, Symbol}) = eval(f)
function resolve_func(li, slot::Slot)
    instance(expr_type(li, slot))
end
function resolve_func(li, f::Expr)
    try
        # TODO figure out what can go wrong here, since this seems rather fragile
        return eval(f)
    catch e
        println("Couldn't resolve $f")
        rethrow(e)
    end
    error("$f not a callable")
end
function expr_type(li, x)
    t = _expr_type(li, x)
    push!(li, t) # add as dependency
    t
end

_expr_type(li, x::Expr) = x.typ
_expr_type(li, x::TypedSlot) = x.type
_expr_type(li, x::GlobalRef) = typeof(eval(x))
_expr_type{T}(li, x::T) = T
_expr_type(li, slot::Union{Slot, SSAValue}) = slottype(li, slot)

function glsl_rewrite_pass(li, expr)
    list = Sugar.replace_expr(expr) do expr

        if isa(expr, Slot)
            return true, slotname(li, expr)
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
                f = try
                     resolve_func(li, func)
                catch e
                    println(STDERR, "Failed to resolve $func $types")
                    rethrow(e)
                end
                intrinsic, result = rewrite_function(li, f, types, similar_expr(expr, args))
                @show intrinsic result
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

type Transpiler
    cache
    Transpiler() = new(Dict())
end

type Decl
    signature
    transpiler::Transpiler
    decls::OrderedSet
    dependencies::OrderedSet{Decl}
    li
    method
    ast::Expr
    source::String
    funcheader::String

    Decl(signature, transpiler) = new(signature, transpiler, OrderedSet(), OrderedSet{Decl}())
end

function isfunction(x::Decl)
    isa(x.signature, Tuple) && length(x.signature) == 2 && isa(x.signature[1], Function)
end
function istype(x::Decl)
    isa(x.signature, DataType)
end
function Base.push!(decl::Decl, signature)
    push!(decl.dependencies, Decl(signature, decl.transpiler))
end
function Base.push!{T <: gli.Types}(decl::Decl, signature::Type{T}) # don't add intrinsics
    #push!(decl.dependencies, Decl(signature, decl.transpiler))
end
function getmethod(x::Decl)
    if !isdefined(x, :method)
        x.method = Sugar.get_method(x.signature...)
    end
    x.method
end
function getcodeinfo!(x::Decl)
    if !isdefined(x, :li)
        x.li = Sugar.get_lambda(code_typed, x.signature...)
    end
    x.li
end


function getast!(x::Decl)
    if !isdefined(x, :ast)
        li = getcodeinfo!(x) # make sure codeinfo is present
        nargs = method_nargs(x)
        for i in 2:nargs # make sure func args don't get redeclared
            push!(x.decls, SlotNumber(i))
        end
        expr = Sugar.sugared(x.signature..., code_typed)
        x.ast = glsl_rewrite_pass(x, expr)
    end
    x.ast
end
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
function getsource!(x::Decl)
    if !isdefined(x, :source)
        if istype(x)
            x.source = declare_type(x.signature)
        else
            x.source = sprint() do io
                Base.show_unquoted(GLSLIO(io), getast!(x), 0, 0)
            end
        end
    end
    x.source
end


ssatypes(tp::Decl) = tp.li.ssavaluetypes
slottypes(tp::Decl) = tp.li.slottypes
slottype(tp::Decl, s::Slot) = slottypes(tp)[s.id]
slottype(tp::Decl, s::SSAValue) = ssatypes(tp)[s.id + 1]

slotnames(tp::Decl) = tp.li.slotnames
slotname(tp::Decl, s::Slot) = slotnames(tp)[s.id]
slotname(tp::Decl, s::SSAValue) = Sugar.ssavalue_name(s)

function getfuncargs(x::Decl)
    sn, st = slotnames(x), slottypes(x)
    n = method_nargs(x)
    Expr(:tuple, map(2:n) do i
        :($(sn[i])::$(st[i]))
    end...)
end

function getfuncheader!(x::Decl)
    @assert isfunction(x)
    if !isdefined(x, :funcheader)
        x.funcheader = sprint() do io
            args = getfuncargs(x)
            gio = GLSLIO(io)
            print(gio, typename(returntype(x)))
            print(gio, ' ')
            show_name(gio, x.signature[1])
            Base.show_unquoted(gio, args, 0, 0)
        end
    end
    x.funcheader
end
function getfuncsource!(x::Decl)
    string(getfuncheader!(x), "\n", getsource!(x))
end



function returntype(x::Decl)
    getcodeinfo!(x).rettype
end

if v"0.6" < VERSION
    function method_nargs(f::Decl)
        m = getmethod(f)
        m.nargs
    end
else
    function method_nargs(f::Decl)
        li = getcodeinfo!(f)
        li.nargs
    end
end
#
# function broadcast_index{T, N}(arg::gli.GLArray{T, N}, shape, idx)
#     sz = size(arg)
#     i = (sz .<= shape) .* idx
#     return arg[i]
# end
# broadcast_index(arg, shape, idx) = arg
#
# function broadcast_kernel{T}(A::gli.GLArray{T, 2}, f, a, b)
#     idx = NTuple{2, Int}(GlobalInvocationID())
#     sz = size(A)
#     A[idx] = f(
#         broadcast_index(a, sz, idx),
#         broadcast_index(b, sz, idx),
#     )
#     return
# end
# f, types = broadcast_kernel, (gli.GLArray{Float64, 2}, typeof(+), gli.GLArray{Float64, 2}, Float64)
# decl = Decl((f, types), Transpiler())
# #ast = getsource!(decl)
# ast = getast!(decl)
# println(getsource!(decl))
# for elem in decl.dependencies
#     println(getsource!(elem))
# end
# lal = getcodeinfo!(decl)
# println(getfuncsource!(decl))
