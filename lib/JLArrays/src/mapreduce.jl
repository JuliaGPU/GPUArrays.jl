# reductions

struct ArrayNoCopy end
# (an `Array` can't wrap the storage of isbits unions, whose selector bytes are elsewhere)
Adapt.adapt_storage(::ArrayNoCopy, x::JLArray{T}) where {T} =
    Base.isbitsunion(T) ? jlconvert(x) : typed_data(x)
@static if !isdefined(KernelAbstractions, :POCL) # KA v0.9
    # (AcceleratedKernels picks its host algorithms by the back-end)
    KernelAbstractions.get_backend(::JLDeviceArray) = KernelAbstractions.CPU()
end

# GPUArrays reduces with AcceleratedKernels, whose kernels do not run on JLArrays' back-end. The
# reference implementation runs AcceleratedKernels' host algorithms on the arrays' storage
# instead, through GPUArrays' internal indirection for this purpose. (The storage is wrapped
# without a copy, so the arrays must be preserved; a `Broadcasted` source is materialized on the
# host.)
_host_source(A::Broadcast.Broadcasted) =   # (`copy` of a 0-dimensional one gives a scalar)
    Array(copyto!(similar(A, Broadcast.combine_eltypes(A.f, A.args)), A))
_host_source(A) = adapt(ArrayNoCopy(), A)

# (as a `JLArray`, a `StaticDimsArray` over one is reduced on the host)
const StaticDimsJLArray = GPUArrays.StaticDimsArray{<:Any, <:Any, <:Any, <:JLArray}

GPUArrays._ak_mapreduce(f, op, A::Union{AnyJLArray, StaticDimsJLArray, Broadcast.Broadcasted{<:JLArrayStyle}};
                        backend=nothing, kwargs...) =
    GC.@preserve A @allowscalar GPUArrays.AK.mapreduce(f, op, _host_source(A); kwargs...)

function GPUArrays._ak_mapreducedim!(f, op, R::Union{AnyJLArray, StaticDimsJLArray},
                                     A::Union{AbstractArray, Broadcast.Broadcasted};
                                     backend=nothing, kwargs...)
    GC.@preserve R A @allowscalar GPUArrays.AK.mapreducedim!(
        f, op, adapt(ArrayNoCopy(), R), _host_source(A); kwargs...)
    return R
end
