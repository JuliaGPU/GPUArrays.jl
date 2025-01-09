# Interface

To extend the above functionality to a new array type, you should use the types and
implement the interfaces listed on this page. GPUArrays is designed around having two
different array types to represent a GPU array: one that exists only on the host, and
one that actually can be instantiated on the device (i.e. in kernels).
Device functionality is then handled by [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).

## Host abstractions

You should provide an array type that builds on the `AbstractGPUArray` supertype, such as:

```julia
mutable struct CustomArray{T, N} <: AbstractGPUArray{T, N}
    data::DataRef{Vector{UInt8}}
    offset::Int
    dims::Dims{N}
    ...
end

```

This will allow your defined type (in this case `JLArray`) to use the GPUArrays interface where available.
To be able to actually use the functionality that is defined for `AbstractGPUArray`s, you need to define the backend, like so:

```julia
import KernelAbstractions: Backend
struct CustomBackend <: KernelAbstractions.GPU
KernelAbstractions.get_backend(a::CA) where CA <: CustomArray = CustomBackend()
```

There are numerous examples of potential interfaces for GPUArrays, such as with [JLArrays](https://github.com/JuliaGPU/GPUArrays.jl/blob/master/lib/JLArrays/src/JLArrays.jl), [CuArrays](https://github.com/JuliaGPU/CUDA.jl/blob/master/src/gpuarrays.jl), and [ROCArrays](https://github.com/JuliaGPU/AMDGPU.jl/blob/master/src/gpuarrays.jl).

## Caching Allocator

```@docs
GPUArrays.@cached
GPUArrays.@uncached
```
