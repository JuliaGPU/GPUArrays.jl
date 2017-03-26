# const functions2lift = (
#     fft, fft!, ifft!, ifft
# )
# isarray(x) = x == AbstractArray || x <: Array
# for f in functions2lift
#     for m in methods(f)
#         args = [m.sig.parameters[2:end]...]
#         if any(isarray, args)
#             # replace Array with JLArray
#             @show args
#             ret = Base.return_types(fft, (args...))
#             @show ret
#             if length(ret) != 1
#                 continue
#             end
#             RT = first(ret)
#             @show(RT)
#             sym_t = map(enumerate(args)) do iarg
#                 i, arg = iarg
#                 T = isarray(arg) ? :JLArray : arg
#                 Symbol("arg_$i"), T, isarray(arg)
#             end
#             argtyped = map(sym_t) do st
#                 Expr(:(::), st[1], st[2])
#             end
#             argslifted = map(sym_t) do stb
#                 stb[3] ? :(buffer($(stb[1]))) : :($(stb[1]))
#             end
#             body = :($f($(argslifted...)))
#             if isarray(RT)
#                 body = :(JLArray($body))
#             end
#             expr = :($f($(argtyped...)) = $body)
#             println(expr)
#             #@eval expr
#         end
#     end
# end


single_arg = (:fft, :fft!, :ifft, :ifft!)
for f in single_arg
    # TODO, could we just be fine with being AbstractArray?
    @eval Base.$f{T, N}(A::JLArray{T, N}) = JLArray{T, N}($f(buffer(A)))
end
