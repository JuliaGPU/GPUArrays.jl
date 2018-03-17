
BroadcastStyle(::Type{T}) where T <: GPUArray = ArrayStyle{T}()
BroadcastStyle(::Type{Any}, ::Type{T}) where T <: GPUArray = ArrayStyle{T}()
BroadcastStyle(::Type{T}, ::Type{Any}) where T <: GPUArray = ArrayStyle{T}()
function broadcast_similar(f, ::ArrayStyle{T}, ::Type{ElType}, inds, As...) where {T <: GPUArray, ElType}
    similar(T, ElType, length.(inds))
end

 # RefValue doesn't work with CUDAnative so we use Tuple, which should have the same behaviour
deref(x) = x
deref(x::RefValue) = (x[],)

broadcast!(func, out::GPUArray, ::Nothing, args...) = gpu_broadcast!(func, out, args)
# this is not mentioned in the docs, but the above specialisation doesn't seem to get hit for e.g.
# x .= identity.(2.0)
broadcast!(func, out::GPUArray, ::Scalar, args...) = gpu_broadcast!(func, out, args)
function gpu_broadcast!(
        func, out::GPUArray, _args
    )
    args = deref.(_args)
    shape = broadcast_indices(out)
    gshape = UInt32.(size(out))
    keeps, Idefaults = map_newindexer(shape, args)
    descriptor_tuple = ntuple(length(args)) do i
        BInfo(args[i], keeps[i], Idefaults[i])
    end
    gpu_call(broadcast_kernel!, out, (func, out, gshape, descriptor_tuple, args))
    out
end

@inline function broadcast_kernel!(state, func, out, shape, descriptor, args)
    ilin = @linearidx(out, state)
    @inbounds out[ilin] = apply_broadcast(ilin, func, shape, descriptor, args)
    return
end

arg_length(x::Tuple) = (UInt32(length(x)),)
arg_length(x::AbstractArray) = UInt32.(size(x))
arg_length(x) = () # Scalar

struct BInfo{Typ, N}
    size::NTuple{N, UInt32}
    keep::NTuple{N, UInt32}
    idefault::NTuple{N, UInt32}
end

function BInfo(val, keep, idefault)
    N = length(keep)
    typ = typeof(combine_styles(val))
    BInfo{typ, N}(arg_length(val), UInt32.(keep), UInt32.(idefault))
end

@propagate_inbounds @inline function _broadcast_getindex(
        ::BInfo{<: ArrayStyle}, A, I
    )
    A[I]
end
@propagate_inbounds @inline function _broadcast_getindex(
        ::BInfo{Style{Tuple}}, A, I
    )
    A[I]
end
@inline _broadcast_getindex(any, A, I) = A


# don't do anything for empty tuples
@pure newindex(I, ilin, keep::Tuple{}, Idefault::Tuple{}, size::Tuple{}) = UInt32(1)

# optimize for 1D arrays
@pure function newindex(I::NTuple{1}, ilin, keep::NTuple{1}, Idefault, size)
    Bool(keep[1]) ? ilin : Idefault[1]
end

# differently shaped arrays
@generated function newindex(I, ilin::T, keep::NTuple{N}, Idefault, size) where {N, T}
    exprs = Expr(:tuple)
    for i = 1:N
        push!(exprs.args, :(T(Bool(keep[$i]) ? T(I[$i]) : T(Idefault[$i]))))
    end
    :(Base.@_inline_meta; gpu_sub2ind(size, $exprs))
end

for N = 0:15
    nargs = N + 1
    inner_expr = []
    valargs = []
    for i = 1:N
        val_i = Symbol("val_", i); I_i = Symbol("I_", i);
        desi = Symbol("deref_", i)
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
            @inbounds $val_i = _broadcast_getindex($desi, args[$i], $I_i)
        end
        push!(inner_expr, inner)
        push!(valargs, val_i)
    end
    @eval begin
        @inline function apply_broadcast(ilin, func, shape, descriptor, args::NTuple{$N, Any})
            # this will hopefully get dead code removed,
            # if only arrays with linear index are involved, because I should be unused in that case
            I = gpu_ind2sub(shape, ilin)
            $(inner_expr...)
            # call the function and store the result
            func($(valargs...))
        end
    end
end

function foreach_kernel(state, func, shape, descriptor, args)
    ilin = @linearidx(args[1], state)
    apply_broadcast(ilin, func, shape, descriptor, args)
    return
end

function Base.foreach(func, over::GPUArray, Bs...)
    shape = UInt32.(size(over))
    keeps, Idefaults = map_newindexer(shape, over, Bs)
    args = (over, Bs...)
    descriptor_tuple = ntuple(length(args)) do i
        BInfo(args[i], keeps[i], Idefaults[i])
    end
    gpu_call(foreach_kernel, over, (func, shape, descriptor_tuple, (over, deref.(Bs)...)))
    return
end
function mapidx_kernel(state, f, A, args)
    ilin = @linearidx(A, state)
    f(ilin, A, args...)
    return
end
function mapidx(f, A::GPUArray, args::NTuple{N, Any}) where N
    gpu_call(mapidx_kernel, A, (f, A, args))
end
