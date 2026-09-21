# on-device array type and kernel argument conversion

struct Adaptor end
jlconvert(arg) = adapt(Adaptor(), arg)

# FIXME: add Ref to Adapt.jl (but make sure it doesn't cause ambiguities with CUDAnative's)
struct JlRefValue{T} <: Ref{T}
  x::T
end
Base.getindex(r::JlRefValue) = r.x
Adapt.adapt_structure(to::Adaptor, r::Base.RefValue) = JlRefValue(adapt(to, r[]))

## executed on-device

# array type
@static if !isdefined(JLArrays.KernelAbstractions, :POCL) # KA v0.9
    struct JLDeviceArray{T, N} <: AbstractDeviceArray{T, N}
        data::Vector{UInt8}
        offset::Int
        dims::Dims{N}
    end

    Base.elsize(::Type{<:JLDeviceArray{T}}) where {T} = sizeof(T)

    Base.size(x::JLDeviceArray) = x.dims
    Base.sizeof(x::JLDeviceArray) = Base.elsize(x) * length(x)

    Base.unsafe_convert(::Type{Ptr{T}}, x::JLDeviceArray{T}) where {T} =
        convert(Ptr{T}, pointer(x.data)) + x.offset

    # conversion of untyped data to a typed Array
    function typed_data(x::JLDeviceArray{T}) where {T}
        unsafe_wrap(Array, pointer(x), x.dims)
    end

    @inline Base.getindex(A::JLDeviceArray, index::Integer) = getindex(typed_data(A), index)
    @inline Base.setindex!(A::JLDeviceArray, x, index::Integer) = setindex!(typed_data(A), x, index)
end
