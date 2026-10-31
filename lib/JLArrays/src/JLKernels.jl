module JLKernels

using ..JLArrays
using ..JLArrays: jlconvert

import KernelAbstractions: KernelAbstractions, POCLBackend
import KernelInterface as KI

import Adapt


## back-end

export JLBackend

struct JLBackend <: KI.Backend end

KI.get_backend(a::JLA) where JLA <: JLArray = JLBackend()
KI.get_backend(a::JLA) where JLA <: Union{JLSparseMatrixCSC, JLSparseMatrixCSR, JLSparseVector} = JLBackend()

KI.allocate(::JLBackend, ::Type{T}, dims::Tuple) where T = JLArray{T}(undef, dims)

## KernelInterface

# JLArrays are executed by POCL: convert arguments to their POCL-compatible (CPU)
# counterparts, and forward compilation and launches to the POCL backend. KernelAbstractions
# kernels are launched through these methods too.

KI.functional(::JLBackend) = true

# POCL launches synchronously, and JLArrays live in host memory
KI.synchronize(::JLBackend) = nothing

function KI.copyto!(::JLBackend, A, B)
    length(A) == length(B) ||
        throw(ArgumentError("Arrays must match in length, got $(length(A)) and $(length(B))"))
    Base.copyto!(A, B)
    return A
end

KI.argconvert(::JLBackend, arg) = KI.argconvert(POCLBackend(), jlconvert(arg))

# the POCL kernel, and the callable it was compiled from, which has to be kept alive since
# its converted form only holds pointers to the arrays it captures
struct JLKernel{K, F}
    kernel::K
    f::F
end

function KI.kernel_function(backend::JLBackend, f::F, tt::TT = Tuple{}; kwargs...) where {F, TT}
    kernel = KI.kernel_function(POCLBackend(), jlconvert(f), tt; kwargs...)
    kern = JLKernel(kernel, f)
    return KI.Kernel{JLBackend, typeof(kern)}(backend, kern)
end

function KI.launch(obj::KI.Kernel{JLBackend}, groups::Dims{3}, items::Dims{3}, args::Tuple; kwargs...)
    # Use `map` rather than `jlconvert.(args)` to skip the broadcast
    # machinery (broadcasted/materialize/ntuple) that would otherwise
    # be specialized per unique arg-tuple type.
    f = obj.kern.f
    GC.@preserve f args KI.launch(obj.kern.kernel, groups, items, map(jlconvert, args); kwargs...)
end

KI.max_work_group_size(kernel::KI.Kernel{JLBackend}) = KI.max_work_group_size(kernel.kern.kernel)
KI.launch_configuration(kernel::KI.Kernel{JLBackend}; kwargs...) =
    KI.launch_configuration(kernel.kern.kernel; kwargs...)
for f in (:max_work_group_size, :max_work_group_dims, :max_num_groups, :sub_group_size,
          :multiprocessor_count, :supports_subgroups, :supports_linear_subgroups,
          :supports_independent_subgroups, :supports_partitionable_subgroups,
          :supports_float64, :supports_atomics)
    @eval KI.$f(::JLBackend) = KI.$f(POCLBackend())
end
KI.supports_shuffle(::JLBackend, ::Type{T}) where {T} = KI.supports_shuffle(POCLBackend(), T)

Adapt.adapt_storage(::JLBackend, x) = Adapt.adapt(JLArray, x)

end
