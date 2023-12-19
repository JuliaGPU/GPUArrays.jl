# convenience and indirect construction

# conversions from CPU arrays rely on constructors
Base.convert(::Type{T}, a::AbstractArray) where {T<:AbstractGPUArray} = a isa T ? a : T(a)
# TODO: can we implement constructors to and from ::AbstractArray here? by calling the undef
#       constructor and doing a `copyto!`. this is tricky, due to ambiguities, and no easy
#       way to go from <:AbstractGPUArray{T,N} to e.g. CuArray{S,N}


## convenience constructors

function Base.fill!(A::AnyGPUArray{T}, x) where T
    length(A) == 0 && return A
    gpu_call(A, convert(T, x)) do ctx, a, val
        idx = @linearidx(a)
        @inbounds a[idx] = val
        return
    end
    A
end


## identity matrices

function identity_kernel(ctx::AbstractKernelContext, res::AbstractArray{T}, stride, val) where T
    i = linear_index(ctx)
    ilin = (stride * (i - 1)) + i
    ilin > length(res) && return
    @inbounds res[ilin] = val
    return
end

function (T::Type{<: AnyGPUArray{U}})(s::UniformScaling, dims::Dims{2}) where {U}
    res = similar(T, dims)
    fill!(res, zero(U))
    gpu_call(identity_kernel, res, size(res, 1), s.λ; elements=minimum(dims))
    res
end

(T::Type{<: AnyGPUArray})(s::UniformScaling{U}, dims::Dims{2}) where U = T{U}(s, dims)

(T::Type{<: AnyGPUArray})(s::UniformScaling, m::Integer, n::Integer) = T(s, Dims((m, n)))

function Base.copyto!(A::AbstractGPUMatrix{T}, s::UniformScaling) where T
    fill!(A, zero(T))
    gpu_call(identity_kernel, A, size(A, 1), s.λ; elements=minimum(size(A)))
    A
end

function _one(unit::T, x::AbstractGPUMatrix) where {T}
    m,n = size(x)
    m==n || throw(DimensionMismatch("multiplicative identity defined only for square matrices"))
    I = similar(x, T)
    fill!(I, zero(T))
    gpu_call(identity_kernel, I, m, unit; elements=m)
    I
end

Base.one(x::AbstractGPUMatrix{T}) where {T} = _one(one(T), x)
Base.oneunit(x::AbstractGPUMatrix{T}) where {T} = _one(oneunit(T), x)


## type checking

export contains_eltype, explain_allocatedinline

# back-ends often do not support all kinds of element types, so provide some helpers
# to detect unsupported caes

function hasfieldcount(@nospecialize(dt))
    try
        fieldcount(dt)
    catch
        return false
    end
    return true
end

# for finding specific element types, e.g., when Float64 is unsupported
function contains_eltype(T, typ)
    if T === typ
      return true
    elseif T isa Union
        for U in Base.uniontypes(T)
            contains_eltype(U, typ) && return true
        end
    elseif hasfieldcount(T)
        for U in fieldtypes(T)
            contains_eltype(U, typ) && return true
        end
    end
    return false
end

# Types that are allocated inline include:
# 1. plain bitstypes (`Int`, `(Float16, Float32)`, plain immutable structs, etc).
#    these are simply stored contiguously in the buffer.
# 2. structs of unions (`struct Foo; x::Union{Int, Float32}; end`)
#    these are stored with a selector at the end (handled by Julia).
# 3. bitstype unions (`Union{Int, Float32}`, etc)
#    these are stored contiguously and require a selector array (handled by us)
#
# This function explains why a type is not allocated inline.
function explain_allocatedinline(@nospecialize(T), depth=0; maxdepth=10)
    depth > maxdepth && return ""

    if T isa Union
      msg = "  "^depth * "$T is a union that's not allocated inline\n"
      for U in Base.uniontypes(T)
        if !Base.allocatedinline(U)
          msg *= explain_eltype(U, depth+1)
        end
      end
    elseif Base.ismutabletype(T)
      msg = "  "^depth * "$T is a mutable type\n"
    elseif hasfieldcount(T)
      msg = "  "^depth * "$T is a struct that's not allocated inline\n"
      for U in fieldtypes(T)
          if !Base.allocatedinline(U)
              msg *= explain_nonisbits(U, depth+1)
          end
      end
    else
      msg = "  "^depth * "$T is not allocated inline\n"
    end
    return msg
end


## derived arrays

# to avoid needless array wrappers (which make it harder to write generic wrappers) we
# materialize several kinds of operations:
# - contiguous views
# - reinterpret
# - reshape
#
# For this functionality to be reusable, we expect the back-end to provide a `derive`
# operation that makes it possible to create a new array object, with a different type or
# size, but backed by the same data. The `additional_offset` is the number of elements
# to offset the new array from the original array.

derive(::Type, a::AbstractGPUArray, osize::Dims, additional_offset::Int) =
    error("Not implemented")
