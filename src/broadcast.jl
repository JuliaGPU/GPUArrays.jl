
@generated function newindex{N}(I, keep::NTuple{N}, Idefault)
    exprs = Expr(:tuple)
    for i = 1:N
        push!(exprs.args, :(keep[$i] ? I[$i] : Idefault[$i]))
    end
    exprs
end

@inline function Base.Broadcast.broadcast_t(
        f, T, shape, iter, A::AbstractAccArray, Bs::Vararg{Any,N}
    ) where N
    C = similar(A, T, shape)
    keeps, Idefaults = Base.Broadcast.map_newindexer(shape, A, Bs)
    Base.Broadcast._broadcast!(f, C, keeps, Idefaults, A, Bs, Val{N}, iter)
    return C
end
using Base.Broadcast
function get_cartesian_index(A::AbstractAccArray, shape)
    sub2ind(shape, linear_index(A))
end

for N = 0:10
    nargs = N + 1
    inner_expr = []
    args = []
    valargs = []
    for i = 1:N
        Ai = Symbol("A_", i); keepi = Symbol("keep_$i");
        Idefault_i = Symbol("idefault_$i")
        val_i = Symbol("val_", i); I_i = Symbol("I_", i)
        push!(args, Ai)
        inner = quote
            # destructure the keeps and As tuples
            $keepi = keeps[$i]
            $Idefault_i = Idefaults[$i]
            # reverse-broadcast the indices
            $I_i = newindex(I, $keepi, $Idefault_i)
            # extract array values
            @inbounds $val_i = Broadcast._broadcast_getindex(A_i, I_i)
            # call the function and store the result
        end
        push!(inner_expr, inner)
        push!(valargs, val_i)
    end
    eval(quote
        function broadcast_kernel!(kernel, B, shape, keeps, Idefaults, $(args...))
            $(Expr(:meta, :inline))
            I = get_cartesian_index(B, shape)
            $(inner_expr...)
            result = kernel($(valargs...))
            @inbounds B[I] = result
            return
        end
    end)
end

function Base.Broadcast._broadcast!(
        func, out::AbstractAccArray,
        keeps::K, Idefaults::ID,
        A::AT, Bs::BT, unused1, unused2 # we don't need those arguments
    ) where {K, ID, AT, BT}
    shape = Int32.(size(B))
    gpu_call(out, broadcast_kernel!, func, out, shape, keeps, Idefaults, A, Bs...)
    B
end
