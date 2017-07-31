using Base.Broadcast
import Base.Broadcast: broadcast!, _broadcast!, broadcast_t
using Base.Broadcast: map_newindexer
using Base: @propagate_inbounds, @pure

@inline function const_kernel(state, A, op, len)
    idx = linear_index(A, state)
    @inbounds if idx <= len
        A[idx] = op()
    end
    return
end
@inline function const_kernel2(state, A, x, len)
    idx = linear_index(A, state)
    @inbounds if idx <= len
        A[idx] = x
    end
    return
end
function broadcast!(f, A::AbstractAccArray)
    gpu_call(const_kernel, A, (A, f, Cuint(length(A))))
end
function broadcast!(f::typeof(identity), A::AbstractAccArray, val::Number)
    valconv = convert(eltype(A), val)
    gpu_call(const_kernel2, A, (A, valconv, Cuint(length(A))))
end

@inline function broadcast_t(
        f, ::Type{T}, shape, iter, A::AbstractAccArray, Bs::Vararg{Any,N}
    ) where {N, T}
    C = similar(A, T, shape)
    keeps, Idefaults = map_newindexer(shape, A, Bs)
    _broadcast!(f, C, keeps, Idefaults, A, Bs, Val{N}, iter)
    return C
end
@inline function broadcast_t(
        f, ::Type{T}, shape, iter, A::AbstractAccArray, B::AbstractAccArray, rest::Vararg{Any,N}
    ) where {N, T}
    C = similar(A, T, shape)
    Bs = (B, rest...)
    keeps, Idefaults = map_newindexer(shape, A, Bs)
    _broadcast!(f, C, keeps, Idefaults, A, Bs, Val{N}, iter)
    return C
end

@inline function broadcast_t(
        f, T, shape, iter, A::Any, B::AbstractAccArray, rest::Vararg{Any, N}
    ) where N
    C = similar(B, T, shape)
    Bs = (B, rest...)
    keeps, Idefaults = map_newindexer(shape, A, Bs)
    _broadcast!(f, C, keeps, Idefaults, A, Bs, Val{N}, iter)
    return C
end

function _broadcast!(
        func, out::AbstractAccArray,
        keeps::K, Idefaults::ID,
        A::AT, Bs::BT, ::Type{Val{N}}, unused2 # we don't need those arguments
    ) where {N, K, ID, AT, BT}
    shape = Cuint.(size(out))
    args = (A, Bs...)
    descriptor_tuple = ntuple(length(args)) do i
        BroadcastDescriptor(args[i], keeps[i], Idefaults[i])
    end
    gpu_call(broadcast_kernel!, out, (func, out, shape, Cuint.(length(out)), descriptor_tuple, A, Bs...))
    out
end

function Base.foreach(func, over::AbstractAccArray, Bs...)
    shape = Cuint.(size(over))
    keeps, Idefaults = map_newindexer(shape, over, Bs)
    args = (over, Bs...)
    descriptor_tuple = ntuple(length(args)) do i
        BroadcastDescriptor(args[i], keeps[i], Idefaults[i])
    end
    gpu_call(foreach_kernel, over, (func, shape, Cuint.(length(over)), descriptor_tuple, over, Bs...))
    return
end


arg_length(x::AbstractAccArray) = Cuint.(size(x))
arg_length(x) = ()

abstract type BroadcastDescriptor{Typ} end

immutable BroadcastDescriptorN{Typ, N} <: BroadcastDescriptor{Typ}
    size::NTuple{N, Cuint}
    keep::NTuple{N, Cuint}
    idefault::NTuple{N, Cuint}
end

function BroadcastDescriptor(val, keep, idefault)
    N = length(keep)
    typ = if isa(val, Ref{<: AbstractAccArray})
        Any # special case ref, so we can upload it unwrapped already!
    else
        Broadcast.containertype(val)
    end
    BroadcastDescriptorN{typ, N}(arg_length(val), Cuint.(keep), Cuint.(idefault))
end

@propagate_inbounds @inline function _broadcast_getindex(
        ::BroadcastDescriptor{Array}, A, I
    )
    A[I]
end
@inline _broadcast_getindex(any, A, I) = A

for N = 0:10
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
        end
        push!(inner_expr, inner)
        push!(valargs, val_i)
    end
    @eval begin

        @inline function apply_broadcast(ilin, state, func, shape, len, descriptor, $(args...))
            # this will hopefully get dead code removed,
            # if only arrays with linear index are involved, because I should be unused in that case
            I = gpu_ind2sub(shape, ilin)
            $(inner_expr...)
            # call the function and store the result
            func($(valargs...))
        end

        @inline function broadcast_kernel!(state, func, B, shape, len, descriptor, $(args...))
            ilin = linear_index(B, state)
            if ilin <= len
                @inbounds B[ilin] = apply_broadcast(ilin, state, func, shape, len, descriptor, $(args...))
            end
            return
        end

        function foreach_kernel(state, func, shape, len, descriptor, A, $(args...))
            ilin = linear_index(A, state)
            if ilin <= len
                apply_broadcast(ilin, state, func, shape, len, descriptor, A, $(args...))
            end
            return
        end

        function mapidx_kernel(state, f, A, len, $(args...))
            i = linear_index(A, state)
            @inbounds if i <= len
                f(i, A, $(args...))
            end
            return
        end
    end

end

function mapidx{N}(f, A::AbstractAccArray, args::NTuple{N, Any})
    gpu_call(mapidx_kernel, A, (f, A, Cuint(length(A)), args...))
end
# Base functions that are sadly not fit for the the GPU yet (they only work for Int64)
@pure @noinline function gpu_ind2sub{N, T}(dims::NTuple{N}, ind::T)
    _ind2sub(NTuple{N, T}(dims), ind - T(1))
end
@pure @inline _ind2sub{T}(::Tuple{}, ind::T) = (ind + T(1),)
@pure @inline function _ind2sub{T}(indslast::NTuple{1}, ind::T)
    ((ind + T(1)),)
end
@pure @inline function _ind2sub{T}(inds, ind::T)
    r1 = inds[1]
    indnext = div(ind, r1)
    f = T(1); l = r1
    (ind-l*indnext+f, _ind2sub(Base.tail(inds), indnext)...)
end

@pure function gpu_sub2ind{N, T}(dims::NTuple{N}, I::NTuple{N, T})
    Base.@_inline_meta
    _sub2ind(NTuple{N, T}(dims), T(1), T(1), I...)
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
@pure newindex(I, ilin, keep::Tuple{}, Idefault::Tuple{}, size::Tuple{}) = Cuint(0)

# optimize for 1D arrays
@pure newindex(I::NTuple{1}, ilin, keep::NTuple{1}, Idefault, size) = ilin

# differently shaped arrays
@generated function newindex{N, T}(I, ilin::T, keep::NTuple{N}, Idefault, size)
    exprs = Expr(:tuple)
    for i = 1:N
        push!(exprs.args, :(Bool(keep[$i]) ? T(I[$i]) : T(Idefault[$i])))
    end
    :(Base.@_inline_meta; gpu_sub2ind(size, $exprs))
end
