# TODO move this in own package or some transpiler package
using Sugar

import Base: indent_width, quoted_syms, uni_ops, expr_infix_wide, expr_infix_any
import Base: all_ops, expr_calls, expr_parens, ExprNode, show_block
import Base: show_list, show_enclosed_list, operator_precedence, is_linenumber
import Base: is_quoted, is_expr, TypedSlot, ismodulecall, is_intrinsic_expr
import Base: show_generator, show_call, show_unquoted
import Sugar: ssavalue_name, ASTIO, get_slottypename, get_type

type GLSLIO{T <: IO} <: ASTIO
    io::T
end

show_linenumber(io::GLSLIO, line)       = print(io, " // line ", line,':')
show_linenumber(io::GLSLIO, line, file) = print(io, " // ", file, ", line ", line, ':')



# don't print f0 TODO this is a Float32 hack
function Base.show(io::GLSLIO, x::Float32)
    print(io, Float64(x))
end
function Base.show_unquoted(io::IO, sym::Symbol, ::Int, ::Int)
    print(io, Symbol(gli.glsl_hygiene(sym)))
end

function show_unquoted(io::GLSLIO, ex::GlobalRef, ::Int, ::Int)
    # TODO Why is Base.x suddenly == GPUArrays.GLBackend.x
    #if ex.mod == GLSLIntrinsics# || ex.mod == GPUArrays.GLBackend
        print(io, ex.name)
    #else
    #    error("No non Intrinsic GlobalRef's for now!: $ex")
    #end
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
    if !isempty(func_args) && isa(func_args[1], Expr) && func_args[1].head == :parameters
        print(io, op)
        show_list(io, func_args[2:end], ", ", indent)
        print(io, "; ")
        show_list(io, func_args[1].args, ", ", indent)
        print(io, cl)
    else
        show_enclosed_list(io, op, func_args, ", ", cl, indent)
    end
end


function Base.show_unquoted(io::GLSLIO, ssa::SSAValue, ::Int, ::Int)
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

    ind = head === :module || head === :baremodule ? indent : indent + indent_width
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
    show_name(io, name)
end


function show_unquoted(io::GLSLIO, ex::Expr, indent::Int, prec::Int)
    line_number = 0 # TODO include line numbers
    head, args, nargs = ex.head, ex.args, length(ex.args)
    # dot (i.e. "x.y"), but not compact broadcast exps
    if head == :(.) && !is_expr(args[2], :tuple)
        show_unquoted(io, args[1], indent + indent_width)
        print(io, '.')
        if is_quoted(args[2])
            show_unquoted(io, unquoted(args[2]), indent + indent_width)
        else
            print(io, '(')
            show_unquoted(io, args[2], indent + indent_width)
            print(io, ')')
        end

    # variable declaration TODO might this occure in normal code?
    elseif head == :(::) && nargs == 2
        print(io, typename(args[2]))
        print(io, ' ')
        show_name(io, args[1])
    # infix (i.e. "x<:y" or "x = y")

    elseif (head in expr_infix_any && (nargs == 2) || (head == :(:)) && nargs == 3)
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
        f = first(args)
        fname = Symbol(f)
        if fname == :getfield && nargs == 3
            show_unquoted(io, args[2], indent) # type to be accessed
            print(io, '.')
            show_unquoted(io, args[3], indent)
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
                show_unquoted(io, fname, indent)
                if isa(func_args[1], Expr) || func_args[1] in all_ops
                    show_enclosed_list(io, '(', func_args, ",", ')', indent, func_prec)
                else
                    show_unquoted(io, func_args[1])
                end

            # binary operator (i.e. "x + y")
            elseif func_prec > 0 # is a binary operator
                na = length(func_args)
                if (na == 2 || (na > 2 && func in (:+, :++, :*))) && all(!isa(a, Expr) || a.head !== :... for a in func_args)
                    sep = " $fname "
                    if func_prec <= prec
                        show_enclosed_list(io, '(', func_args, sep, ')', indent, func_prec, true)
                    else
                        show_list(io, func_args, sep, indent, func_prec, true)
                    end
                elseif na == 1
                    # 1-argument call to normally-binary operator
                    op, cl = expr_calls[head]
                    show_unquoted(io, fname, indent)
                    show_enclosed_list(io, op, func_args, ",", cl, indent)
                else
                    show_call(io, head, fname, func_args, indent)
                end

            # normal function (i.e. "f(x,y)")
            else
                show_call(io, head, fname, func_args, indent)
            end
        end
    # other call-like expressions ("A[1,2]", "T{X,Y}", "f.(X,Y)")
    elseif haskey(expr_calls, head) && nargs >= 1  # :ref/:curly/:calldecl/:(.)
        funcargslike = head == :(.) ? ex.args[2].args : ex.args[2:end]
        show_call(io, head, ex.args[1], funcargslike, indent)
    # comparison (i.e. "x < y < z")
    elseif (head == :comparison) && nargs >= 3 && (nargs & 1==1)
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

    elseif (head == :module) && nargs==3 && isa(args[1],Bool)
        show_block(io, args[1] ? :module : :baremodule, args[2], args[3], indent)
        print(io, "}")

    # type declaration
    elseif (head == :type) && nargs==3
        # TODO struct
        show_block(io, args[1] ? :type : :immutable, args[2], args[3], indent)
        print(io, "}")

    elseif head == :bitstype && nargs == 2
        unsupported_expr("Bitstype", line_number)

    # empty return (i.e. "function f() return end")
    elseif head == :return && nargs == 1 && (args[1] === nothing)
        # ignore empty return

    # type annotation (i.e. "::Int")
    elseif head == :(::) && nargs == 1
        print(io, ' ')
        show_name(io, args[1])

    # var-arg declaration or expansion
    # (i.e. "function f(L...) end" or "f(B...)")
    elseif (head === :(...)) && nargs == 1
        show_unquoted(io, args[1], indent)
        print(io, "...")

    elseif (nargs == 0 && head in (:break, :continue))
        print(io, head)

    elseif (nargs == 1 && head in (:abstract, :const)) ||
                          head in (:local,  :global, :export)
        print(io, head, ' ')
        show_list(io, args, ", ", indent)

    elseif (head === :macrocall) && nargs >= 1
        # Use the functional syntax unless specifically designated with prec=-1
        show_unquoted(io, expand(ex), indent)

    elseif (head === :typealias) && nargs == 2
        print(io, "typealias ")
        show_list(io, args, ' ', indent)

    elseif (head === :line) && 1 <= nargs <= 2
        show_linenumber(io, args...)

    elseif (head === :if) && nargs == 3     # if/else
        show_block(io, "if",   args[1], args[2], indent)
        show_block(io, "} else", args[3], indent)
        print(io, "}")

    elseif (head === :let) && nargs >= 1
        unsupported_expr("let", line_number)

    elseif (head === :block) || (head === :body)
        show_block(io, "", ex, indent); print(io, "}")


    elseif ((head === :&)#= || (head === :$)=#) && length(args) == 1
        print(io, head)
        a1 = args[1]
        parens = (isa(a1, Expr) && a1.head !== :tuple) || (isa(a1,Symbol) && isoperator(a1))
        parens && print(io, "(")
        show_unquoted(io, a1)
        parens && print(io, ")")


    elseif (head === :meta)
        # TODO, just ignore this? Log this? We definitely don't need it in GLSL

    elseif (head === :return)
        if length(args) == 1
            # return Void must not return anything in GLSL
            if !((isa(args[1], Expr) && args[1].typ == Void) || args[1] == nothing)
                print(io, "return ")
            end
            show_unquoted(io, args[1])
        elseif isempty(args)
            # ignore return if no args or void
        else
            error("What dis return? $ex")
        end
    elseif head == :inbounds # ignore
    else
        println(ex)
        unsupported_expr(string(ex), line_number)
    end
    nothing
end

function show_name(io::GLSLIO, x)
    print(io, glsl_name(x))
end


const global_identifier = "globalvar_"


function declare_global(io::GLSLIO, vars)
    for (i, expr) in enumerate(vars)
        name, typ = expr.args
        if typ <: Function # special casing functions
            print(io, "const ")
            print(io, typename(typ))
            print(io, ' ', global_identifier)
            show_name(io, name)
            print(io, " = ")
            show_name(io, typename(typ))
            println(io, "(false);")
            continue
        end
        qualifiers = String[]
        bindingloc = typ <: gli.GLArray ? "binding " : "location "
        if typ <: gli.GLArray
            push!(qualifiers, image_format(typ))
        end
        push!(qualifiers, string(bindingloc, " = ", i - 1))

        print(io, "layout (", join(qualifiers, ", "), ") ")
        tname = if typ <: gli.GLArray
            "uniform image2D"
        else
            utyp = typ <: GLBuffer ? "in " : "uniform "
            string(utyp, typename(typ))
        end
        print(io, tname, ' ')
        show_name(io, string(global_identifier, name))
        println(io, ';')
    end
end
