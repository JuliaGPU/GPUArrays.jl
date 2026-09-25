# map-reduce
#
# Base's reductions of GPU arrays, implemented with AcceleratedKernels' primitives: `AK.mapreduce`
# for scalar results, `AK.mapreducedim!` into a destination otherwise. AcceleratedKernels applies
# `init` once and needs no neutral element; everything that only exists because Base says so
# (result types, empty results and their errors, the one-element result, `dims` rules) is decided
# here, on the host.

const AbstractArrayOrBroadcasted = Union{AbstractArray,Broadcast.Broadcasted}

# AcceleratedKernels' primitives, on the backend `backend` (a `Broadcasted` source may hold arrays
# without one, such as `EachIndex`). The reference back-end (JLArrays) runs them on its storage
# instead; these are internal, not an extension point for back-ends.
_ak_mapreduce(f, op, A; kwargs...) = AK.mapreduce(f, op, A; kwargs...)
_ak_mapreducedim!(f, op, R, A; kwargs...) = AK.mapreducedim!(f, op, R, A; kwargs...)

# `Base.mapreducedim!` folds into `R`'s values
Base.mapreducedim!(f, op, R::AnyGPUArray, A::AbstractArray) =
    _ak_mapreducedim!(f, op, R, A; backend=get_backend(R))
Base.mapreducedim!(f, op, R::AnyGPUArray, A::Broadcast.Broadcasted) =
    _ak_mapreducedim!(f, op, R, A; backend=get_backend(R))

# GPUArrays used to route its reductions through this function, which back-ends overrode. It is
# no longer called, and kept (with AcceleratedKernels' implementation: `init` is applied once,
# else `R` is folded into) only so that back-ends that still extend it keep loading.
function mapreducedim!(f, op, R::AnyGPUArray, A::AbstractArrayOrBroadcasted; init=nothing)
    backend = get_backend(R)
    if init === nothing
        _ak_mapreducedim!(f, op, R, A; backend)
    else
        _ak_mapreducedim!(f, op, R, A; backend, init)
    end
    return R
end

# `neutral_element` lives in GPUArraysCore, so that packages building on GPUArraysCore share it
import GPUArraysCore: neutral_element

# `init` when none is given (an explicit `init=nothing` is an initial value, as in Base)
struct _NoInit end

# resolve ambiguities
Base.mapreduce(f, op, A::AnyGPUArray, As::AbstractArrayOrBroadcasted...;
               dims=:, init=_NoInit()) = _mapreduce(f, op, A, As...; dims=dims, init=init)
Base.mapreduce(f, op, A::Broadcast.Broadcasted{<:AbstractGPUArrayStyle}, As::AbstractArrayOrBroadcasted...;
               dims=:, init=_NoInit()) = _mapreduce(f, op, A, As...; dims=dims, init=init)

function _mapreduce(f::F, op::OP, As::Vararg{Any,N}; dims::D, init) where {F,OP,N,D}
    if !(dims isa Colon)
        all(d -> d isa Integer, dims) ||
            throw(ArgumentError("reduced dimension(s) must be integers"))
        all(d -> d >= 1, dims) || throw(ArgumentError("region dimension(s) must be ≥ 1, got $dims"))
    end

    # apply the mapping function to the input arrays
    if N == 1
        # ... with only a single input, we can defer this to the reduce step
        A = only(As)
    else
        # mapreduce should apply `f` like `map` does, consuming elements like iterators
        A = if allequal(size.(As)...)
            Broadcast.instantiate(Broadcast.broadcasted(f, As...))
        else
            # TODO: can we avoid the reshape + view?
            indices = LinearIndices.(As)
            common_length = minimum(length.(indices))
            Bs = map(As) do A
                view(reshape(A, length(A)), 1:common_length)
            end
            Broadcast.instantiate(Broadcast.broadcasted(f, Bs...))
        end
        f = identity
    end
    S = _source_eltype(A)
    M = Base.promote_op(f, S)
    backend = _source_backend(As)

    if dims isa Colon
        # Base's results for empty and one-element inputs; otherwise AcceleratedKernels' result,
        # converted to the type Base's fold settles on
        if init isa _NoInit
            length(A) == 0 &&
                return Base.mapreduce_empty_iter(f, op, Array{S}(undef, 0), Base.HasEltype())
            length(A) == 1 && return @allowscalar Base.mapreduce_first(f, op, first(A))
            return convert(_fold_type(op, Union{}, M), _ak_mapreduce(f, op, A; backend))
        else
            length(A) == 0 && return init
            length(A) == 1 && return @allowscalar op(init, f(first(A)))
            return convert(_fold_type(op, typeof(init), M), _ak_mapreduce(f, op, A; backend, init))
        end
    end

    # along `dims`: Base's shape, and Base's element type (`typeof(init)`, else the fold type)
    rax = Base.reduced_indices(axes(A), dims)
    if !(init isa _NoInit)
        R = similar(A, typeof(init), length.(rax))
        _ak_mapreducedim!(f, op, R, A; backend, init)
        return R
    end
    if any(d -> d <= ndims(A) && size(A)[d] == 0, dims)
        # Base's values for an empty reduction (or its error), from a host stand-in; `f` may be
        # called, and may index device arrays
        h = @allowscalar Base.reducedim_init(f, op, Array{S}(undef, size(A)), dims)
        R = similar(A, eltype(h), size(h))
        isempty(R) || copyto!(R, h)
        return R
    end
    T = _fold_type(op, Union{}, M)
    R = similar(A, T, length.(rax))
    z = _reducedim_zero(op, T)
    if z === nothing
        _ak_mapreducedim!(f, op, R, A; backend, overwrite=true)
    else
        # Base's sums along `dims` start from zero (`reducedim_init`), so a slice of `-0.0`s sums
        # to `0.0`; the other operators' initial values do not change the result
        _ak_mapreducedim!(f, op, R, A; backend, init=z)
    end
    return R
end

_reducedim_zero(op, ::Type) = nothing
_reducedim_zero(::Union{typeof(+), typeof(Base.add_sum)}, ::Type{T}) where {T} =
    hasmethod(zero, Tuple{Type{T}}) ? zero(T) : nothing

# The element type of a reduction source before mapping
_source_eltype(A::AbstractArray) = eltype(A)
_source_eltype(bc::Broadcast.Broadcasted) = Broadcast.combine_eltypes(bc.f, bc.args)

# The backend of the first GPU array among a reduction's sources
_source_backend(A::AnyGPUArray) = get_backend(A)
_source_backend(bc::Broadcast.Broadcasted) = _source_backend(bc.args)
_source_backend(::Tuple{}) = nothing
function _source_backend(t::Tuple)
    b = _source_backend(first(t))
    return b === nothing ? _source_backend(Base.tail(t)) : b
end
_source_backend(_) = nothing

# The type Base's fold `op(op(init, x₁), x₂)...` settles on, for elements of type `M` (without
# `init`, `I === Union{}`, from `op(x₁, x₂)`): the types `op` returns, not `init`'s
function _fold_type(op, ::Type{I}, ::Type{M}) where {I, M}
    T = I === Union{} ? Base.promote_op(op, M, M) : Base.promote_op(op, I, M)
    for _ in 1:8
        S = promote_type(T, Base.promote_op(op, T, M))
        S == T && return T
        T = S
    end
    return T
end

# `any` and `all` follow Base. A scalar call with a predicate that returns a `Bool` uses
# AcceleratedKernels' short-circuiting implementation; one that can return `missing` reduces a code
# of each value (`false` < `missing` < `true`) with `max` or `min`, which gives Base's three-valued
# logic without storing `missing`. Along `dims`, as in Base, the predicate must return a `Bool`.
# Values the predicate cannot return (by inference) are an error before launching, and nothing is
# checked for an empty array, whose elements Base never passes to the predicate.
struct _Bool3{F} <: Function
    f::F
end
@inline (c::_Bool3)(x) = _bool3(c.f(x))
@inline _bool3(x::Bool) = x ? 0x02 : 0x00
@inline _bool3(::Missing) = 0x01
@inline _bool3(x) = throw(TypeError(:any, "", Union{Bool, Missing}, x))
_unbool3(c::UInt8) = c == 0x01 ? missing : c == 0x02

struct _BoolOnly{F} <: Function
    f::F
end
@inline function (b::_BoolOnly)(x)
    y = b.f(x)
    y isa Bool || throw(TypeError(:any, "", Bool, y))
    return y
end

for (fname, op, op3, empty3) in ((:any, :|, :max, 0x00), (:all, :&, :min, 0x02))
    @eval function Base.$fname(f::Function, A::AnyGPUArray; dims=:)
        T = Base.promote_op(f, eltype(A))
        if dims === Colon()
            isempty(A) && return $(fname === :all)
            T === Union{} || Bool <: T || T <: Union{Bool, Missing} || throw(ArgumentError(
                "the predicate of `$($fname)` must return a `Bool` or `missing`, not `$T`"))
            T <: Bool && return AK.$fname(f, A)
            return _unbool3(mapreduce(_Bool3(f), $op3, A; init=$empty3))
        else
            # (`reduced_indices` checks `dims` as Base does)
            isempty(A) && return fill!(similar(A, Bool, length.(Base.reduced_indices(axes(A), dims))),
                                       $(fname === :all))
            T === Union{} || Bool <: T || throw(ArgumentError(
                "the predicate of `$($fname)` along `dims` must return a `Bool`, not `$T`"))
            return mapreduce(_BoolOnly(f), $op, A; dims)
        end
    end
end
Base.any(A::AnyGPUArray{<:Union{Bool, Missing}}; dims=:) = any(identity, A; dims)
Base.all(A::AnyGPUArray{<:Union{Bool, Missing}}; dims=:) = all(identity, A; dims)

Base.count(pred::Function, A::AnyGPUArray; dims=:, init=0) =
    mapreduce(Base._bool(pred), Base.add_sum, A; init=init, dims=dims)

# The in-place reductions, with Base's `init::Bool` but without Base's pass that initializes `r`
# (`initarray!`): `init=true` reduces from the operator's identity, as an `init`, or where it has
# none overwrites `r`; `init=false` folds into `r`'s values. When every slice is empty, Base's
# result (or error) comes from a host stand-in.
for (fname, op, idfun) in ((:sum, :(Base.add_sum), zero), (:prod, :(Base.mul_prod), one),
                           (:maximum, :max, nothing), (:minimum, :min, nothing),
                           (:extrema, :(Base._extrema_rf), nothing),
                           (:any, :|, Returns(false)), (:all, :&, Returns(true)),
                           (:count, :(Base.add_sum), zero))
    fname! = Symbol(fname, '!')
    mapf = fname === :extrema ? :(Base.ExtremaMap(f)) : fname === :count ? :(Base._bool(f)) : :f
    ftype = fname === :count ? :Any : :Function
    @eval function Base.$(fname!)(f::$ftype, r::AnyGPUArray, A::AnyGPUArray; init::Bool=true)
        if isempty(A) && !isempty(r)
            hr = Array(r)
            Base.$(fname!)(f, hr, Array{eltype(A)}(undef, size(A)); init)
            return copyto!(r, hr)
        end
        backend = get_backend(r)
        if !init
            _ak_mapreducedim!($mapf, $op, r, A; backend)
        elseif $(idfun === nothing)
            _ak_mapreducedim!($mapf, $op, r, A; backend, overwrite=true)
        else
            _ak_mapreducedim!($mapf, $op, r, A; backend, init=$idfun(eltype(r)))
        end
        return r
    end
end

LinearAlgebra.ishermitian(A::AbstractGPUMatrix) = mapreduce(==, &, A, adjoint(A))


# comparisons

# ignores missing
function Base.isequal(A::AnyGPUArray, B::AnyGPUArray)
    if A === B return true end
    if axes(A) != axes(B)
        return false
    end
    mapreduce(isequal, &, A, B; init=true)
end

# returns `missing` when missing values are involved
function Base.:(==)(A::AnyGPUArray, B::AnyGPUArray)
    if axes(A) != axes(B)
        return false
    end

    function mapper(a, b)
        eq = (a == b)
        if ismissing(eq)
            (; is_missing=true, is_equal=#=don't care=#false)
        else
            (; is_missing=false, is_equal=eq)
        end
    end
    function reducer(a, b)
        if a.is_missing || b.is_missing
            (; is_missing=true, is_equal=#=don't care=#false)
        else
            (; is_missing=false, is_equal=a.is_equal & b.is_equal)
        end
    end
    res = mapreduce(mapper, reducer, A, B; init=(; is_missing=false, is_equal=true))
    res.is_missing ? missing : res.is_equal
end
