# TODO move this in own package or some transpiler package
using Sugar, GLAbstraction, GeometryTypes

import Base: indent_width, quoted_syms, uni_ops, expr_infix_wide, expr_infix_any
import Base: all_ops, expr_calls, expr_parens, ExprNode, show_block
import Base: show_list, show_enclosed_list, operator_precedence, is_linenumber
import Base: is_quoted, is_expr, TypedSlot, ismodulecall, is_intrinsic_expr
import Base: show_generator, show_call, show_unquoted
import Sugar: ssavalue_name, ASTIO, get_slottypename, get_type

include("intrinsics.jl")


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
    #is_glslintrinsic(eval(f), types)
    if haskey(pirate_loot, f)
        for (sig, funcname) in pirate_loot[f]
            if matches_signature(sug, Tuple{types...})
                return true, funcname
            end
        end
        return false, f
    else
        is_glslintrinsic(eval(f), types)
    end
end

function is_glslintrinsic{N, T}(f::Type{Vec{N, T}}, types::ANY)
    true, Symbol(vecname(f))
end
function is_glslintrinsic(f::Function, types::ANY)
    false, glsl_name(f)
end
function is_glslintrinsic(f::Symbol, types::ANY)
    if f == :Vec
        T = Sugar.return_type(Vec, types)
        return true, glsl_name(T)
    end
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
    !isempty(code_lowered(func, types)), glsl_name(func)
end






type GLSLIO <: ASTIO
    io
    vardecl
    lambdainfo::LambdaInfo
    slotnames
    dependencies::Vector
    types::Vector{String}
end

function GLSLIO(io, lambdainfo, slotnames)
    GLSLIO(io, Dict(), lambdainfo, slotnames, [], String[])
end

show_linenumber(io::GLSLIO, line)       = print(io, " // line ", line,':')
show_linenumber(io::GLSLIO, line, file) = print(io, " // ", file, ", line ", line, ':')

function Base.show_unquoted(io::GLSLIO, newvar::NewvarNode, ::Int, ::Int)
    typ, name = get_slottypename(io, newvar.slot)
    try
        show_name(io, typ)
        print(io, ' ')
        show_name(io, name)
    catch e
        @show newvar typ name
        rethrow(e)
    end
end

# don't print f0 TODO this is a Float32 hack
function Base.show(io::GLSLIO, x::Float32)
    print(io, Float64(x))
end

function show_unquoted(io::GLSLIO, ex::GlobalRef, ::Int, ::Int)
    # TODO Why is Base.x suddenly == GPUArrays.GLBackend.x
    if ex.mod == GLSLIntrinsics || ex.mod == GPUArrays.GLBackend
        print(io, ex.name)
    else
        error("No non Intrinsic GlobalRef's for now!: $ex")
    end
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
    show_name(io, name)
end



function resolve_funcname(io, f::GlobalRef)
    _f = eval(f)
    _f, f.name
end
function resolve_funcname(io, f::Symbol)
    _f = eval(f)
    _f, f
end

function resolve_funcname(io, slot::Slot)
    typ, name = get_slottypename(io, slot)
    f = typ.instance
    f, Symbol(f)
end

function resolve_funcname(io, f::Expr)
    try
        if f.head == :curly
            # TODO figure out what can go wrong here, since this seems fragile
            expr = Sugar.similar_expr(f)
            expr.args = map(f.args) do arg
                # TODO, can other static parameters beside literal values escape with code_typed, optimization = false?
                if isa(arg, Expr) && arg.head == :static_parameter
                    arg.args[1]
                else
                    arg
                end
            end
            T = eval(expr)
            if haskey(pirate_loot, T)
                return T, pirate_loot[T][1][2]
            else
                return T, Symbol(T)
            end
        end
    catch e
        println("Couldn't resolve $f")
        rethrow(e)
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
            try
                transpile(func, typs, io, false)
            catch e
                @show f typs
                rethrow(e)
            end
        end
        fname = glsl_name(func)
    else
        fname = intrfun
    end
    return func, fname
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
        func_inst, fname = resolve_function(io, func, typs)
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
        # ignore empty return

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

    elseif (nargs == 1 && head in (:abstract, :const)) ||
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
        if length(args) == 1
            # return Void must not return anything in GLSL
            if get_type(io, args[1]) != Void
                print(io, "return ")
            end
            show_unquoted(io, args[1])
        elseif isempty(args)
            # ignore return if no args or void
        else
            error("What dis return? $ex")
        end
    else
        println(ex)
        unsupported_expr(string(ex), line_number)
    end
    nothing
end


prescripts = Dict(
    Float32 => "",
    Float64 => "",
    Int => "i",
    Int32 => "i",
    UInt => "u",
    Bool => "b"
)
function glsl_hygiene(sym)
    # TODO unicode
    # TODO figure out what other things are not allowed
    # TODO startswith gl_, but allow variables that are actually valid inbuilds
    x = string(sym)
    x = replace(x, "#", "__")
    x = replace(x, "!", "_bang")
    if x == "out"
        x = "_out"
    end
    if x == "in"
        x = "_in"
    end
    x
end
glsl_sizeof(T) = sizeof(T) * 8
# for now we disallow Float64 and map it to Float32 -> super hack alert!!!!
glsl_sizeof(::Type{Float64}) = 32
glsl_length{T <: Number}(::Type{T}) = 1
glsl_length(T) = length(T)

glsl_name(x) = Symbol(glsl_hygiene(_glsl_name(x)))

function _glsl_name(T)
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
    return str
end

function _glsl_name{T, N}(x::Type{gli.GLArray{T, N}})
    if !(N in (1, 2, 3))
        # TODO, fake ND arrays with 1D array
        error("GPUArray can't have more than 3 dimensions for now")
    end
    sz = glsl_sizeof(T)
    len = glsl_length(T)
    "image$(N)D$(len)x$(sz)_bindless"
end
function _glsl_name{N, T}(::Type{Vec{N, T}})
    string(prescripts[T], "vec", N)
end

function _glsl_name(x::Union{AbstractString, Symbol})
    x
end
_glsl_name(x::Type{Void}) = "void"
_glsl_name(x::Type{Float64}) = "float"
_glsl_name(x::Type{Bool}) = "bool"

# TODO this will be annoying on 0.6
_glsl_name(x::typeof(gli.:(.*))) = "*"
_glsl_name(x::typeof(gli.:(.<=))) = "lessThanEqual"
_glsl_name(x::typeof(gli.:(.+))) = "+"

function _glsl_name(f::Function)
    # Taken from base... #TODO make this more stable
    _glsl_name(typeof(f).name.mt.name)
end

function show_name{T, N}(io::GLSLIO, x::Type{gli.GLArray{T, N}})
    print(io, glsl_name(x))
end
function show_name{N, T}(io::GLSLIO, x::Type{Vec{N, T}})
    print(io, glsl_name(x))
end

show_name(io::GLSLIO, x::Type{Void}) = print(io, glsl_name(x))
show_name(io::GLSLIO, x::Type{Float64}) = print(io, glsl_name(x))
show_name(io::GLSLIO, x::Type{Bool}) = print(io, glsl_name(x))


function show_name(io::GLSLIO, f::Function)
    show_name(io, glsl_name(f))
end
function show_name(io::GLSLIO, x::Union{AbstractString, Symbol})
    print(io, glsl_name(x))
end

function declare_type(T)
    tname = glsl_name(T)
    sprint() do io
        print(io, "struct ", tname, "{\n")
        fnames = fieldnames(T)
        if isempty(fnames) # structs can't be empty
            # we use bool as a short placeholder type.
            # TODO, are there corner cases where bool is no good?
            println(io, "bool empty;")
        else
            for name in fieldnames(T)
                FT = fieldtype(T, name)
                print(io, "    ", glsl_name(FT))
                print(io, ' ')
                print(io, name)
                println(io, ';')
            end
        end
        println(io, "};")
    end
end
function show_name(io::GLSLIO, T::DataType)
    tname = glsl_name(T)
    if !get(io.vardecl, T, false)
        push!(io.types, declare_type(T))
        io.vardecl[T] = true
    end
    print(io, tname)
end


function materialize_io(x::GLSLIO)
    result_str = ""
    for str in x.dependencies
        result_str *= str * "\n"
    end
    string(result_str, '\n', takebuf_string(x.io))
end

const global_identifier = "globalvar_"
const shader_program_dir = joinpath(dirname(@__FILE__), "shaders")

const _module_cache = Dict()
function to_globalref(f, typs)
    mlist = methods(f, typs)
    if length(mlist) != 1
        error("$f and $typs ambigious")
    end
    m = first(mlist)
    GlobalRef(m.module, m.name)
end
function get_module_cache()
    _module_cache
end


function transpile(f, typs, parentio = nothing, main = true)
    # make sure that not already transpiled
    # if parentio != nothing && get(parentio.vardecl, (f, typs), false)
    #     return
    # end
    cache = get_module_cache()
    if haskey(cache, (f, typs)) # add to module
        if parentio != nothing
            str, deps = cache[(f, typs)]
            push!(parentio.dependencies, (f, typs))
            for dep in deps
                if !(dep in parentio.dependencies)
                    push!(parentio.dependencies, dep)
                end
            end
        end
    else
        local ast;
        try
            ast = Sugar.sugared(f, typs, code_typed)
        catch e
            println("Failed to get code for $f $typs")
            rethrow(e)
        end
        li = Sugar.get_lambda(code_typed, f, typs)
        slotnames = Base.lambdainfo_slotnames(li)
        ret_type = Sugar.return_type(f, typs)
        io = IOBuffer()
        glslio = GLSLIO(io, li, slotnames)
        #println(glslio, "\n// $f$(join(typs, ", "))\n")
        vars = Sugar.slot_vector(li)
        funcargs = vars[2:li.nargs]

        show_name(glslio, ret_type)
        print(glslio, ' ')
        show_name(glslio, f)
        print(glslio, '(')

        for (i, (slot, (name, T))) in enumerate(funcargs)
            glslio.vardecl[slot] = true
            if T <: Function
                print(glslio, "const ")
            end
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
        show_unquoted(glslio, body, 0, 0)
        # pio = parentio == nothing ? glslio : parentio
        # pio.vardecl[(f, typs)] = true


        typed_args = map(typs) do T
            Expr(:(::), T)
        end
        ft = typeof(f)
        fname = Symbol(ft.name.mt.name)
        expr = quote
            $(fname)($(typed_args...)) = ret($ret_type)
        end
        str = takebuf_string(io)
        close(io)
        if parentio != nothing
            for dep in glslio.dependencies
                if !(dep in parentio.dependencies)
                    push!(parentio.dependencies, dep)
                end
            end
            push!(parentio.dependencies, (f, typs))
        end
        cache[(f, typs)] = (str, glslio.dependencies)
        return glslio, funcargs, str

    end
end


function image_format{T, N}(x::Type{gli.GLArray{T, N}})
    "r32f"
end
function declare_global(io::GLSLIO, vars::Vector)
    for (i, (slot, (name, typ))) in enumerate(vars)
        if typ <: Function # special casing functions
            print(io, "const ")
            show_name(io, typ)
            print(io, ' ', global_identifier)
            show_name(io, name)
            print(io, " = ")
            show_name(io, typ)
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
            utyp * glsl_name(typ)
        end
        print(io, tname, ' ')
        show_name(io, global_identifier*name)
        println(io, ';')
    end
end
