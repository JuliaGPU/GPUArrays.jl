module JLKernels

using ..JLArrays
using ..JLArrays: jlconvert

import KernelAbstractions
import KernelAbstractions: Kernel, StaticSize, DynamicSize, partition, launch_config

import Adapt


## back-end

export JLBackend

const MAXTHREADS = 256

struct JLBackend <: KernelAbstractions.GPU
    static::Bool
    JLBackend(;static::Bool=false) = new(static)
end

KernelAbstractions.get_backend(a::JLA) where JLA <: JLArray = JLBackend()
KernelAbstractions.get_backend(a::JLA) where JLA <: Union{JLSparseMatrixCSC, JLSparseMatrixCSR, JLSparseVector} = JLBackend()

function KernelAbstractions.mkcontext(kernel::Kernel{JLBackend}, I, _ndrange, iterspace, ::Dynamic) where Dynamic
    return KernelAbstractions.CompilerMetadata{KernelAbstractions.ndrange(kernel), Dynamic}(I, _ndrange, iterspace)
end

KernelAbstractions.allocate(::JLBackend, ::Type{T}, dims::Tuple) where T = JLArray{T}(undef, dims)

@inline function launch_config(kernel::Kernel{JLBackend}, ndrange, workgroupsize)
    if ndrange isa Integer
        ndrange = (ndrange,)
    end
    if workgroupsize isa Integer
        workgroupsize = (workgroupsize, )
    end

    if KernelAbstractions.workgroupsize(kernel) <: DynamicSize && workgroupsize === nothing
        workgroupsize = (MAXTHREADS,) # Vectorization, 4x unrolling, minimal grain size
    end
    iterspace, dynamic = partition(kernel, ndrange, workgroupsize)
    # partition checked that the ndrange's agreed
    if KernelAbstractions.ndrange(kernel) <: StaticSize
        ndrange = nothing
    end

    return ndrange, workgroupsize, iterspace, dynamic
end

@static if isdefined(KernelAbstractions, :isgpu) # KA v0.9
    KernelAbstractions.isgpu(b::JLBackend) = false
end

@static if !isdefined(KernelAbstractions, :POCL) # KA v0.9
    function convert_to_cpu(obj::Kernel{JLBackend, W, N, F}) where {W, N, F}
        return Kernel{typeof(KernelAbstractions.CPU(; static = obj.backend.static)), W, N, F}(KernelAbstractions.CPU(; static = obj.backend.static), obj.f)
    end
else
    function convert_to_cpu(obj::Kernel{JLBackend, W, N, F}) where {W, N, F}
        return Kernel{typeof(KernelAbstractions.POCLBackend()), W, N, F}(KernelAbstractions.POCLBackend(), obj.f)
    end
end

function (obj::Kernel{JLBackend})(args...; ndrange=nothing, workgroupsize=nothing)
    ndrange, workgroupsize, _, _ = launch_config(obj, ndrange, workgroupsize)
    # Use `map` rather than `jlconvert.(args)` to skip the broadcast
    # machinery (broadcasted/materialize/ntuple) that would otherwise
    # be specialized per unique arg-tuple type.
    device_args = map(jlconvert, args)
    new_obj = convert_to_cpu(obj)
    new_obj(device_args...; ndrange, workgroupsize)
end

Adapt.adapt_storage(::JLBackend, a::Array) = Adapt.adapt(JLArrays.JLArray, a)
Adapt.adapt_storage(::JLBackend, a::JLArrays.JLArray) = a

@static if !isdefined(KernelAbstractions, :POCL) # KA v0.9
    Adapt.adapt_storage(::KernelAbstractions.CPU, a::JLArrays.JLArray) = convert(Array, a)
else
    Adapt.adapt_storage(::KernelAbstractions.POCLBackend, a::JLArrays.JLArray) = convert(Array, a)
end

end
