# TODO move this in own package or some transpiler package
using Sugar

import Base: indent_width, quoted_syms, uni_ops, expr_infix_wide, expr_infix_any
import Base: all_ops, expr_calls, expr_parens, ExprNode, show_unquoted, show_block
import Base: show_list, show_enclosed_list, operator_precedence, is_linenumber
import Base: is_quoted, is_expr, TypedSlot, ismodulecall, is_intrinsic_expr
import Base: show_generator, show_call


module GLSLIntrinsics
    # GLSL heavily relies on fixed size vector operation, which is why a lot of
    # intrinsics need fixed size vectors.
    using GeometryTypes

    # very simple way to generate a function name, which should be hygienic.
    const hidden_root = gensym()
    hashfunc(sym) = Symbol(string(hidden_root, sym))

    ############################################################################
    # Dict interface to the intrinsic module, to add types and functions

    function hasfunc(f::Symbol, types::ANY)
        isdefined(GLSLIntrinsics, f) || return false
        func = eval(GLSLIntrinsics, f)
        # is there some other function like applicable working with types and not values?
        # could als be a try catch, I suppose
        !isempty(code_lowered(func, types))
    end

    function insertfunc!(expr::Expr)
        eval(GLSLIntrinsics, expr)
    end

    function insertfunc!(name::Symbol, args::ANY, body, return_expr = Any)
        n = length(args)
        argnames = ntuple(n) do i
            Symbol("arg_$i")
        end
        arg_sig = ntuple(n) do i
            Expr(:(::), argnames[i], args[i])
        end
        staticparams = ntuple(n) do i
            :($(argnames[i]) <: $(args[i]))
        end
        type_sig = ntuple(n) do i
            :(::Type{$(argnames[i])})
        end
        hashname = hashfunc(name)
        body_expr = if isa(return_expr, DataType)
            # We don't have a real value, but we need to return something of type returntype for inference
            :(unsafe_load(Ptr{$return_expr}(C_NULL)))
        else
            return_expr
        end
        eval(GLSLIntrinsics, quote
            # no inline, since this function is not doing anything anyways
            @noinline function $name($(arg_sig...))
                $body_expr
            end
            function $hashname{$(staticparams...)}($(type_sig...))
                $body
            end
        end)
    end
    function get!(f, name::Symbol, args::ANY)
        if hasfunc(name, args)
            func1 = eval(GLSLIntrinsics, name)
            x = methods(func1, args)
            if isempty(x) # Lol?
                error("No method found for $f $args")
            end
            m = first(x)
            more_special = any(zip(args, (m.sig.parameters[2:end]...))) do t1_t2
                t1, t2 = t1_t2
                t1 <: t2 && t1 != t2
            end
            if !more_special # we want to replace a more specific function
                hashname = hashfunc(name)
                func2 = eval(GLSLIntrinsics, hashname)
                return func2(args...) # if not more special, return content of func
            end
        end
        body = f()
        insertfunc!(name, args, body)
        body
    end
    ############################################################################
    # GLSL intrinsics

    immutable GLArray{T, N}
        x::Array{T, N}
    end

    @glsl_intrinsic function gl_setindex!{T}(x::GLArray{T, 1}, i::Integer, val::Vec{4, T})::Void
        :imageStore
    end

    function Base.setindex!{T}(x::GLArray{T, 1}, val::T, i::Integer)
        gl_setindex!(x, i, Vec(val, val, val, val))
    end
    function Base.setindex!{T}(x::GLArray{T, 1}, val::Vec{4, T}, i::Integer)
        gl_setindex!(x, i, val)
    end

    @glsl_intrinsic function Base.getindex{T}(x::GLArray{T, 1}, i::Integer)::Vec{4, T}
        :imageLoad
    end
    @glsl_intrinsic function Base.getindex{T, I <: Integer}(x::GLArray{T, 2}, i::Vec{2, I})::Vec{4, T}
        :imageLoad
    end

    @glsl_intrinsic function Base.getindex{T}(x::GLArray{T, 1}, i::Integer)::T
        val = gl_getindex(x, i)
        convert(T, val)
    end
    @glsl_intrinsic function Base.getindex{T}(x::GLArray{T, 1}, i::Integer, j::Integer)::T
        val = gl_getindex(x, Vec(i, j))
        convert(T, val)
    end


    @glsl_intrinsic function cos{T <: AbstractFloat}(x::T)::T
        :cos
    end
    @glsl_intrinsic function sin{T <: AbstractFloat}(x::T)::T
        :sin
    end
    @glsl_intrinsic function sqrt{T <: AbstractFloat}(x::T)::T
        :sqrt
    end

    function Base.getindex{T}(x::GLArray{T, 2}, i::Integer, j::Integer)
        getindex(x, Vec(i, j))
    end

end


immutable GLSLIO{T <: IO} <: AstIO
    io::T
    lambdainfo::LambdaInfo
    slotnames
end


show_linenumber(io::GLSLIO, line)       = print(io, " // line ", line,':')
show_linenumber(io::GLSLIO, line, file) = print(io, " // ", file, ", line ", line, ':')

function Base.show_unquoted(io::GLSLIO, newvar::NewvarNode, ::Int, ::Int)
    typ, name = get_slottypename(io, newvar.slot)
    print(io, typ, " ", name)
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
        fname = isa(func, GlobalRef) ? func.name : func
        func_prec = operator_precedence(fname)
        if func_prec > 0 || fname in uni_ops
            func = fname
        end
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
        elseif isa(func,Symbol) && func in uni_ops && length(func_args) == 1
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
                print(io, "(")
                show_unquoted(io, func, indent)
                print(io, ")")
                show_enclosed_list(io, op, func_args, ",", cl, indent)
            else
                show_call(io, head, func, func_args, indent)
            end

        # normal function (i.e. "f(x,y)")
        else
            show_call(io, head, func, func_args, indent)
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
        show_block(io, "{", ex, indent); print(io, "}")

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

    else
        unsupported_expr(string(ex), line_number)
    end
    nothing
end
