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
        tagoffset::Int  # see `JLArray`
        dims::Dims{N}
    end

    Base.elsize(::Type{<:JLDeviceArray{T}}) where {T} = aligned_sizeof(T)

    Base.size(x::JLDeviceArray) = x.dims
    Base.sizeof(x::JLDeviceArray) = Base.elsize(x) * length(x)

    Base.unsafe_convert(::Type{Ptr{T}}, x::JLDeviceArray{T}) where {T} =
        convert(Ptr{T}, pointer(x.data)) + x.offset

    # conversion of untyped data to a typed Array
    function typed_data(x::JLDeviceArray{T}) where {T}
        unsafe_wrap(Array, pointer(x), x.dims)
    end

    @inline function Base.getindex(A::JLDeviceArray{T}, index::Integer) where {T}
        if Base.isbitsunion(T)
            @boundscheck checkbounds(A, index)
            union_load(A, index)
        else
            getindex(typed_data(A), index)
        end
    end
    @inline function Base.setindex!(A::JLDeviceArray{T}, x, index::Integer) where {T}
        if Base.isbitsunion(T)
            @boundscheck checkbounds(A, index)
            union_store!(A, convert(T, x)::T, index)
            return A
        else
            setindex!(typed_data(A), x, index)
        end
    end

    # elements of isbits unions are stored as the member type that their selector byte
    # indexes in `Base.uniontypes(T)`

    @generated function union_load(A::JLDeviceArray{T}, index::Integer) where {T}
        ex = :(error("invalid selector byte"))
        for (sel, U) in Iterators.reverse(enumerate(Base.uniontypes(T)))
            val = Base.issingletontype(U) ? U.instance : :(unsafe_load(convert(Ptr{$U}, ptr)))
            ex = :(selector == $(UInt8(sel - 1)) ? $val : $ex)
        end
        quote
            data = A.data
            selector = @inbounds data[A.tagoffset + index]
            GC.@preserve data begin
                ptr = pointer(data) + A.offset + (index - 1) * $(Base.aligned_sizeof(T))
                $ex
            end
        end
    end

    @generated function union_store!(A::JLDeviceArray{T}, x, index::Integer) where {T}
        sel = findfirst(==(x), Base.uniontypes(T))
        quote
            data = A.data
            @inbounds data[A.tagoffset + index] = $(UInt8(sel - 1))
            GC.@preserve data begin
                ptr = pointer(data) + A.offset + (index - 1) * $(Base.aligned_sizeof(T))
                $(Base.issingletontype(x) ? nothing : :(unsafe_store!(convert(Ptr{$x}, ptr), x)))
            end
            return
        end
    end
end
