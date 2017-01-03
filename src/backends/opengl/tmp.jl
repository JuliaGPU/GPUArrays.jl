
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

using MacroTools

macro glsl_intrinsic(func)
    expr = @match func begin
        name_{T__}(args__)::RT_ => begin
            quote
                @noinline function $name{$(T...)}($(args...))::$RT
                    unsafe_load(Ptr{$RT}(C_NULL))
                end
            end
        end
        name_(args__)::RT_ => begin
            quote
                @noinline function $name($(args...))::$RT
                    unsafe_load(Ptr{$RT}(C_NULL))
                end
            end
        end
    end
    esc(expr)
end
