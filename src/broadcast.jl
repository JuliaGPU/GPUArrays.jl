
function gpu_ind2sub{T}(dims, ind::T)
    Base.@_inline_meta
    _ind2sub(dims, ind - T(1))
end
_ind2sub{T}(::Tuple{}, ind::T) = (ind + T(1),)
function _ind2sub{T}(indslast::NTuple{1}, ind::T)
    Base.@_inline_meta
    ((ind + T(1)),)
end
function _ind2sub{T}(inds, ind::T)
    Base.@_inline_meta
    r1 = inds[1]
    indnext = div(ind, r1)
    f = T(1); l = r1
    (ind-l*indnext+f, _ind2sub(Base.tail(inds), indnext)...)
end

function gpu_sub2ind{N, T}(dims::NTuple{N, T}, I::NTuple{N, T})
    Base.@_inline_meta
    _sub2ind(dims, T(1), T(1), I...)
end
_sub2ind(x, L, ind) = ind
function _sub2ind{T}(::Tuple{}, L, ind, i::T, I::T...)
    Base.@_inline_meta
    ind + (i - T(1)) * L
end
function _sub2ind(inds, L, ind, i::IT, I::IT...) where IT
    Base.@_inline_meta
    r1 = inds[1]
    _sub2ind(Base.tail(inds), L * r1, ind + (i - IT(1)) * L, I...)
end

# don't do anything for empty tuples
newindex(I, keep, Idefault, size::Tuple{}) = Cint(0)

# optimize for arrays of same shape to leading array
newindex{N}(I::NTuple{N}, ilin, keep::NTuple{N}, Idefault, size) = ilin

# differently shaped arrays
@generated function newindex{N}(I, ilin, keep::NTuple{N}, Idefault, size)
    exprs = Expr(:tuple)
    for i = 1:N
        push!(exprs.args, :(Bool(keep[$i]) ? I[$i] : Idefault[$i]))
    end
    :(gpu_sub2ind(size, $exprs))
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

function get_cartesian_index(A, shape)
    gpu_ind2sub(shape, Cint(linear_index(A)))
end

arg_length(x::GPUArray) = Cint.(size(x))
arg_length(x) = ()

immutable BroadcastDescriptor{Typ, N}
    padd::Float32 # the following might all be empty, so we need at least one field
    size::NTuple{N, Cint}
    keep::NTuple{N, Cint}
    idefault::NTuple{N, Cint}
end

function BroadcastDescriptor(val, keep, idefault)
    N = length(keep)
    BroadcastDescriptor{Broadcast.containertype(val), N}(
        0f0,
        arg_length(val),
        Cint.(keep),
        Cint.(idefault)
    )
end

 Base.@propagate_inbounds function _broadcast_getindex(
        ::BroadcastDescriptor{Array}, A::Ref, I
    )
    A[]
end
Base.@propagate_inbounds function _broadcast_getindex(
        ::BroadcastDescriptor{Broadcast.ScalarType}, A, I
    )
    A
end
Base.@propagate_inbounds function _broadcast_getindex(any, A, I)
    A[I]
end

for N = 0:3
    nargs = N + 1
    inner_expr = []
    args = []
    valargs = []
    for i = 1:N
        Ai = Symbol("A_", i);
        val_i = Symbol("val_", i); I_i = Symbol("I_", i);
        desi = Symbol("deref_", i)
        push!(args, Ai)
        inner = quote
            # destructure the keeps and As tuples
            $desi = descriptor[$i]
            # reverse-broadcast the indices
            $I_i = newindex(
                I, ilin,
                $desi.keep,
                $desi.idefault,
                $desi.size
            )
            # extract array values
            @inbounds $val_i = _broadcast_getindex($desi, $Ai, $I_i)
            # call the function and store the result
        end
        push!(inner_expr, inner)
        push!(valargs, val_i)
    end
    final_expr = quote
        function broadcast_kernel!(func, B, shape, descriptor_ref, $(args...))
            descriptor = descriptor_ref[1]
            ilin = linear_index(B)
            # this will hopefully get dead code removed,
            # if only arrays with linear index are involved
            I = gpu_ind2sub(shape, ilin)
            $(inner_expr...)
            result = func($(valargs...))
            @inbounds B[ilin] = result
            return
        end
    end
    eval(final_expr)
end
using Transpiler


function Base.Broadcast._broadcast!(
        func, out::AbstractAccArray,
        keeps::K, Idefaults::ID,
        A::AT, Bs::BT, ::Type{Val{N}}, unused2 # we don't need those arguments
    ) where {N, K, ID, AT, BT}
    shape = Cint.(size(out))
    args = (A, Bs...)
    descriptor_tuple = ntuple(length(args)) do i
        BroadcastDescriptor(args[i], keeps[i], Idefaults[i])
    end
    descriptor = GPUArray([descriptor_tuple])
    gpu_call(out, broadcast_kernel!, (func, out, shape, descriptor, A, Bs...))
    out
end
