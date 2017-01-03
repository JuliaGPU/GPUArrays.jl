# TODO move this in own package or some transpiler package
using Sugar

import Base: indent_width, quoted_syms, uni_ops, expr_infix_wide, expr_infix_any
import Base: all_ops, expr_calls, expr_parens, ExprNode, show_block
import Base: show_list, show_enclosed_list, operator_precedence, is_linenumber
import Base: is_quoted, is_expr, TypedSlot, ismodulecall, is_intrinsic_expr
import Base: show_generator, show_call, show_unquoted
import Sugar: ssavalue_name

include("intrinsics.jl")
import GLSLIntrinsics
import GLSLIntrinsics: GLArray
using GeometryTypes

"""
Simple function to determine if an array of types matches any signature
in `signatures`. Since the cases are simple, it should be fine to return first match
"""
function matches_signature(types, signatures)
    for (sig, glsl) in signatures
        (types <: sig) && return true, glsl
    end
    false, :nomatch
end

function is_glslintrinsic(f::Expr, types::ANY)
    if haskey(pirate_loot, f)
        return true, pirate_loot[f][1][2]
    end
    return false, f
end

function is_glslintrinsic(f::Symbol, types::ANY)
    if haskey(pirate_loot, f)
        pf = pirate_loot[f]
        if isa(pf, Vector)
            matches, glsl = matches_signature(Tuple{types...}, pf)
            matches && return true, glsl
            return false, f
        else isa(pf, Symbol)
            return true, pf
        end
    end
    isdefined(GLSLIntrinsics, f) || return false, f
    func = eval(GLSLIntrinsics, f)
    # is there some other function like applicable working with types and not values?
    # could als be a try catch, I suppose
    !isempty(code_lowered(func, types)), f
end



function Base.getindex{T}(x::GLArray{T, 1}, i::Integer)
    GLSLIntrinsics.imageLoad(x, i)
end
function Base.getindex{T}(x::GLArray{T, 2}, i::Integer, j::Integer)
    getindex(x, Vec(i, j))
end
function Base.getindex{T <: Number}(x::GLArray{T, 2}, idx::Vec{2, Int})
    GLSLIntrinsics.imageLoad(x, idx)[1]
end
function Base.setindex!{T}(x::GLArray{T, 1}, val::T, i::Integer)
    GLSLIntrinsics.imageStore(x, i, Vec(val, val, val, val))
end
function Base.setindex!{T}(x::GLArray{T, 2}, val::T, i::Integer, j::Integer)
    setindex!(x, Vec(val, val, val, val), Vec(i, j))
end
function Base.setindex!{T}(x::GLArray{T, 2}, val::T, idx::Vec{2, Int})
    setindex!(x, Vec(val, val, val, val), idx)
end
function Base.setindex!{T}(x::GLArray{T, 2}, val::Vec{4, T}, idx::Vec{2, Int})
    GLSLIntrinsics.imageStore(x, idx, val)
end
function Base.setindex!{T}(x::GLArray{T, 1}, val::Vec{4, T}, i::Integer)
    GLSLIntrinsics.imageStore(x, i, val)
end


import Sugar: ASTIO, get_slottypename, get_type

if !isdefined(:GLSLIO)
immutable GLSLIO{T <: IO} <: ASTIO
    io::T
    vardecl
    lambdainfo::LambdaInfo
    slotnames
    dependencies::Vector{String}
end
end

function GLSLIO(io, lambdainfo, slotnames)
    GLSLIO(io, Dict(), lambdainfo, slotnames, String[])
end

show_linenumber(io::GLSLIO, line)       = print(io, " // line ", line,':')
show_linenumber(io::GLSLIO, line, file) = print(io, " // ", file, ", line ", line, ':')

function Base.show_unquoted(io::GLSLIO, newvar::NewvarNode, ::Int, ::Int)
    typ, name = get_slottypename(io, newvar.slot)
    show_name(io, typ)
    print(io, ' ')
    show_name(io, name)
end
# show a normal (non-operator) function call, e.g. f(x,y) or A[z]
function Base.show_call(io::GLSLIO, head, func, func_args, indent)
    op, cl = expr_calls[head]
    if isa(func, Symbol) || (isa(func, Expr) &&
            (func.head == :. || func.head == :curly))
        show_name(io, func)
    else
        show_name(io, func)
    end
    if head == :(.)
        print(io, '.')
    end
    if !isempty(func_args) && isa(func_args[1], Expr) && func_args[1].head === :parameters
        print(io, op)
        show_list(io, func_args[2:end], ',', indent)
        print(io, "; ")
        show_list(io, func_args[1].args, ',', indent)
        print(io, cl)
    else
        show_enclosed_list(io, op, func_args, ",", cl, indent)
    end
end

function Base.show_unquoted(io::GLSLIO, ssa::SSAValue, ::Int, ::Int)
    # if not already declared, type the declaration location
    # TODO this quite likely won't work for all situations
    if !get(io.vardecl, ssa, false)
        io.vardecl[ssa] = true
        show_name(io, get_type(io, ssa))
        print(io, ' ')
    end
    print(io, Sugar.ssavalue_name(ssa))
end

# show a block, e g if/for/etc
function show_block(io::GLSLIO, head, args::Vector, body, indent::Int)
    if isempty(args)
        print(io, head, '{')
    else
        print(io, head, '(')
        show_list(io, args, ", ", indent)
        print(io, "){")
    end

    ind = is(head, :module) || is(head, :baremodule) ? indent : indent + indent_width
    exs = (is_expr(body, :block) || is_expr(body, :body)) ? body.args : Any[body]
    for (i, ex) in enumerate(exs)
        sep = i == 1 ? "" : ";"
        print(io, sep, '\n', " "^ind)
        show_unquoted(io, ex, ind, -1)
    end
    print(io, ";\n", " "^indent)
end

function show_unquoted(io::GLSLIO, slot::Slot, ::Int, ::Int)
    typ, name = get_slottypename(io, slot)
    print(io, name)
end
function resolve_funcname(io, f::GlobalRef)
    eval(f), f.name
end
function resolve_funcname(io, f::Symbol)
    eval(f), f
end

function typename(T)
    str = if isa(T, Expr) && T.head == :curly
        string(T, "_", join(T.args, "_"))
    elseif isa(T, Symbol)
        T
    elseif isa(T, Type)
        str = string(T.name.name)
        if !isempty(T.parameters)
            str *= string("_", join(T.parameters, "_"))
        end
        str
    else
        error("Not a type $T")
    end
    return glsl_hygiene(str)
end
function resolve_funcname(io, slot::Slot)
    typ, name = get_slottypename(io, slot)
    f = typ.instance
    f, Symbol(f)
end
function resolve_funcname(io, f::Expr)
    if f.head == :curly
        # TODO figure out what can go wrong here, since this seems fragile
        T = eval(f)
        if haskey(pirate_loot, T)
            return T, pirate_loot[T][1][2]
        else
            return T, typename(f)
        end
    end
    error("$f not a func")
end

function resolve_function(io, f, typs)
    func, fname = resolve_funcname(io, f)
    intrinsic, intrfun = is_glslintrinsic(fname, typs)
    if !intrinsic
        if isa(func, Core.IntrinsicFunction)
            warn("$f is intrinsic. Lets hope its intrinsic in OpenGL as well")
        else
            transpile(func, typs, io)
        end
    else
        fname = intrfun
    end
    return fname
end


function show_unquoted(io::GLSLIO, ex::Expr, indent::Int, prec::Int)
    line_number = 0 # TODO include line numbers
    head, args, nargs = ex.head, ex.args, length(ex.args)
    # dot (i.e. "x.y"), but not compact broadcast exps
    if is(head, :(.)) && !is_expr(args[2], :tuple)
        show_unquoted(io, args[1], indent + indent_width)
        print(io, '.')
        if is_quoted(args[2])
            show_unquoted(io, unquoted(args[2]), indent + indent_width)
        else
            print(io, '(')
            show_unquoted(io, args[2], indent + indent_width)
            print(io, ')')
        end

    # infix (i.e. "x<:y" or "x = y")
    elseif (head in expr_infix_any && nargs == 2) || (is(head, :(:)) && nargs == 3)
        func_prec = operator_precedence(head)
        head_ = head in expr_infix_wide ? " $head " : head
        if func_prec <= prec
            show_enclosed_list(io, '(', args, head_, ')', indent, func_prec, true)
        else
            show_list(io, args, head_, indent, func_prec, true)
        end

    # list (i.e. "(1,2,3)" or "[1,2,3]")
    elseif haskey(expr_parens, head)               # :tuple/:vcat
        op, cl = expr_parens[head]
        if head === :vcat
            sep = ";"
        elseif head === :hcat || head === :row
            sep = " "
        else
            sep = ","
        end
        head !== :row && print(io, op)
        show_list(io, args, sep, indent)
        if (head === :tuple || head === :vcat) && nargs == 1
            print(io, sep)
        end
        head !== :row && print(io, cl)

    # function call
    elseif head === :call && nargs >= 1
        func = args[1]

        typs = map(x-> get_type(io, x), args[2:end])
        fname = resolve_function(io, func, typs)

        # sadly we need to special case some type pirated special cases
        # TODO do this for all fixed vectors
        if fname == :getindex && nargs == 3 && first(typs) <: Vec
            show_unquoted(io, args[2]) # vec to be accessed
            if isa(args[3], Integer) # special case for statically inferable
                field = (:x, :y, :z, :w)[args[3]]
                print(io, ".$field")
            else
                print(io, '[')
                show_unquoted(io, args[3])
                print(io, ']')
            end
        else
            # TODO handle getfield
            func_prec = operator_precedence(fname)
            # TODO do this correctly
            if func_prec > 0 || fname in uni_ops
                func = fname
            end
            func = fname
            func_args = args[2:end]

            if (in(ex.args[1], (GlobalRef(Base, :box), :throw)) ||
                ismodulecall(ex) ||
                (ex.typ === Any && is_intrinsic_expr(ex.args[1])))
            end

            # scalar multiplication (i.e. "100x")
            if (func === :* &&
                length(func_args) == 2 && isa(func_args[1], Real) && isa(func_args[2], Symbol))
                if func_prec <= prec
                    show_enclosed_list(io, '(', func_args, "", ')', indent, func_prec)
                else
                    show_list(io, func_args, "", indent, func_prec)
                end

            # unary operator (i.e. "!z")
            elseif isa(func, Symbol) && func in uni_ops && length(func_args) == 1
                show_unquoted(io, func, indent)
                if isa(func_args[1], Expr) || func_args[1] in all_ops
                    show_enclosed_list(io, '(', func_args, ",", ')', indent, func_prec)
                else
                    show_unquoted(io, func_args[1])
                end

            # binary operator (i.e. "x + y")
            elseif func_prec > 0 # is a binary operator
                na = length(func_args)
                if (na == 2 || (na > 2 && func in (:+, :++, :*))) && all(!isa(a, Expr) || a.head !== :... for a in func_args)
                    sep = " $func "
                    if func_prec <= prec
                        show_enclosed_list(io, '(', func_args, sep, ')', indent, func_prec, true)
                    else
                        show_list(io, func_args, sep, indent, func_prec, true)
                    end
                elseif na == 1
                    # 1-argument call to normally-binary operator
                    op, cl = expr_calls[head]
                    show_unquoted(io, func, indent)
                    show_enclosed_list(io, op, func_args, ",", cl, indent)
                else
                    show_call(io, head, func, func_args, indent)
                end

            # normal function (i.e. "f(x,y)")
            else
                show_call(io, head, func, func_args, indent)
            end
        end
    # other call-like expressions ("A[1,2]", "T{X,Y}", "f.(X,Y)")
    elseif haskey(expr_calls, head) && nargs >= 1  # :ref/:curly/:calldecl/:(.)
        funcargslike = head == :(.) ? ex.args[2].args : ex.args[2:end]
        show_call(io, head, ex.args[1], funcargslike, indent)

    # comprehensions
    elseif (head === :typed_comprehension || head === :typed_dict_comprehension) && length(args) == 2
        isdict = (head === :typed_dict_comprehension)
        isdict && print(io, '(')
        show_unquoted(io, args[1], indent)
        isdict && print(io, ')')
        print(io, '[')
        show_generator(io, args[2], indent)
        print(io, ']')

    elseif (head === :comprehension || head === :dict_comprehension) && length(args) == 1
        print(io, '[')
        show_generator(io, args[1], indent)
        print(io, ']')

    elseif (head === :generator && length(args) >= 2) || (head === :flatten && length(args) == 1)
        print(io, '(')
        show_generator(io, ex, indent)
        print(io, ')')

    elseif head === :filter && length(args) == 2
        show_unquoted(io, args[2], indent)
        print(io, " if ")
        show_unquoted(io, args[1], indent)

    elseif is(head, :ccall)
        unsupported_expr("ccall", line_number)

    # comparison (i.e. "x < y < z")
    elseif is(head, :comparison) && nargs >= 3 && (nargs & 1==1)
        comp_prec = minimum(operator_precedence, args[2:2:end])
        if comp_prec <= prec
            show_enclosed_list(io, '(', args, " ", ')', indent, comp_prec)
        else
            show_list(io, args, " ", indent, comp_prec)
        end

    # function calls need to transform the function from :call to :calldecl
    # so that operators are printed correctly
    elseif head === :function && nargs==2 && is_expr(args[1], :call)
        # TODO, not sure what this is about
        show_block(io, head, Expr(:calldecl, args[1].args...), args[2], indent)
        print(io, "}")

    elseif head === :function && nargs == 1
        # TODO empty function in GLSL?
        unsupported_expr("Empty function, $(args[1])", line_number)

    # block with argument
    elseif head in (:for, :while, :function, :if) && nargs==2
        show_block(io, head, args[1], args[2], indent)
        print(io, "}")

    elseif is(head, :module) && nargs==3 && isa(args[1],Bool)
        show_block(io, args[1] ? :module : :baremodule, args[2], args[3], indent)
        print(io, "}")

    # type declaration
    elseif is(head, :type) && nargs==3
        # TODO struct
        show_block(io, args[1] ? :type : :immutable, args[2], args[3], indent)
        print(io, "}")

    elseif is(head, :bitstype) && nargs == 2
        unsupported_expr("Bitstype", line_number)

    # empty return (i.e. "function f() return end")
    elseif is(head, :return) && nargs == 1 && is(args[1], nothing)
        print(io, head)

    # type annotation (i.e. "::Int")
    elseif is(head, Symbol("::")) && nargs == 1
        print(io, "::")
        show_unquoted(io, args[1], indent)

    # var-arg declaration or expansion
    # (i.e. "function f(L...) end" or "f(B...)")
    elseif is(head, :(...)) && nargs == 1
        show_unquoted(io, args[1], indent)
        print(io, "...")

    elseif (nargs == 0 && head in (:break, :continue))
        print(io, head)

    elseif (nargs == 1 && head in (:return, :abstract, :const)) ||
                          head in (:local,  :global, :export)
        print(io, head, ' ')
        show_list(io, args, ", ", indent)

    elseif is(head, :macrocall) && nargs >= 1
        # Use the functional syntax unless specifically designated with prec=-1
        show_unquoted(io, expand(ex), indent)

    elseif is(head, :typealias) && nargs == 2
        print(io, "typealias ")
        show_list(io, args, ' ', indent)

    elseif is(head, :line) && 1 <= nargs <= 2
        show_linenumber(io, args...)

    elseif is(head, :if) && nargs == 3     # if/else
        show_block(io, "if",   args[1], args[2], indent)
        show_block(io, "else", args[3], indent)
        print(io, "}")

    elseif is(head, :try) && 3 <= nargs <= 4
        unsupported_expr("try catch", line_number)

    elseif is(head, :let) && nargs >= 1
        unsupported_expr("let", line_number)

    elseif is(head, :block) || is(head, :body)
        show_block(io, "", ex, indent); print(io, "}")

    elseif is(head, :quote) && nargs == 1 && isa(args[1],Symbol)
        unsupported_expr("Quoted expression", line_number)

    elseif is(head, :gotoifnot) && nargs == 2
        unsupported_expr("Gotoifnot", line_number)

    elseif is(head, :string) && nargs == 1 && isa(args[1], AbstractString)
        unsupported_expr("String type", line_number)

    elseif is(head, :null)
        print(io, "nothing")

    elseif is(head, :kw) && length(args)==2
        unsupported_expr("Keyword arguments", line_number)

    elseif is(head, :string)
        unsupported_expr("String", line_number)

    elseif (is(head, :&)#= || is(head, :$)=#) && length(args) == 1
        print(io, head)
        a1 = args[1]
        parens = (isa(a1, Expr) && a1.head !== :tuple) || (isa(a1,Symbol) && isoperator(a1))
        parens && print(io, "(")
        show_unquoted(io, a1)
        parens && print(io, ")")

    # transpose
    elseif (head === Symbol('\'') || head === Symbol(".'")) && length(args) == 1
        unsupported_expr("Transpose", line_number)

    elseif is(head, :import) || is(head, :importall) || is(head, :using)
        unsupported_expr("imports", line_number)

    elseif is(head, :meta)
        # TODO, just ignore this? Log this? We definitely don't need it in GLSL

    elseif is(head, :return)
        if length(args) == 1# ignore return if no args
            print(io, "return ")
            show_unquoted(io, args[1])
        elseif isempty(args) # ignore
        else
            error("What dis return? $ex")
        end
    else
        println(ex)
        unsupported_expr(string(ex), line_number)
    end
    nothing
end
