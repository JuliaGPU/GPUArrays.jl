module StaticArrayInterfaceExt

# Static sizes of arrays with a mix of static and dynamic dimensions, for packages that build
# on StaticArrayInterface.jl (e.g. StrideArraysCore.jl and LoopVectorization.jl).

using GPUArrays: GPUArrays, StaticDimsArray, StaticDimsDeviceArray
using StaticArrayInterface: StaticArrayInterface

const AnyStaticDimsArray = Union{StaticDimsArray, StaticDimsDeviceArray}

StaticArrayInterface.static_size(a::AnyStaticDimsArray) = GPUArrays.static_size(a)
StaticArrayInterface.known_size(::Type{A}) where {A <: AnyStaticDimsArray} =
    GPUArrays.known_size(A)

end # module StaticArrayInterfaceExt
