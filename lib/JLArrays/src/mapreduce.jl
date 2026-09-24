# reductions

struct ArrayNoCopy end
Adapt.adapt_storage(::ArrayNoCopy, x::JLArray) = typed_data(x)

# GPUArrays reduces with AcceleratedKernels, whose kernels do not run on JLArrays' back-end. The
# reference implementation runs AcceleratedKernels' host algorithms on the arrays' storage
# instead, through GPUArrays' internal indirection for this purpose. (The storage is wrapped
# without a copy, so the arrays must be preserved; a `Broadcasted` source is materialized on the
# host.)
_host_source(A::Broadcast.Broadcasted) =   # (`copy` of a 0-dimensional one gives a scalar)
    Array(copyto!(similar(A, Broadcast.combine_eltypes(A.f, A.args)), A))
_host_source(A) = adapt(ArrayNoCopy(), A)

GPUArrays._ak_mapreduce(f, op, A::Union{AnyJLArray, Broadcast.Broadcasted{<:JLArrayStyle}};
                        backend=nothing, kwargs...) =
    GC.@preserve A @allowscalar GPUArrays.AK.mapreduce(f, op, _host_source(A); kwargs...)

function GPUArrays._ak_mapreducedim!(f, op, R::AnyJLArray,
                                     A::Union{AbstractArray, Broadcast.Broadcasted};
                                     backend=nothing, kwargs...)
    GC.@preserve R A @allowscalar GPUArrays.AK.mapreducedim!(
        f, op, adapt(ArrayNoCopy(), R), _host_source(A); kwargs...)
    return R
end
