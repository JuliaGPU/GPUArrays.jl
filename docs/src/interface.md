# Interface

To extend the GPUArrays functionality to a new array type, you should use the types and
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

## Device abstractions

!!! warning
    Work in progress.

## Test suite

GPUArrays provides an extensive test suite that covers all of the functionality that should
be available after implementing the required interfaces. This test suite is part of this
package, but for dependency reasons it is not available when importing the package. Instead,
you should include the code from your `runtests.jl` as follows:

```julia
import GPUArrays
gpuarrays = pathof(GPUArrays)
gpuarrays_root = dirname(dirname(gpuarrays))
include(joinpath(gpuarrays_root, "test", "testsuite.jl"))
```

With this set-up, you can run the test suite like this:

```julia
TestSuite.test(MyGPUArrayType)
```

If you don't want to run the whole suite, you can also run parts of it:

```julia
T = JLArray
GPUArrays.allowscalar(false) # fail tests when slow indexing path into Array type is used.

TestSuite.test_gpuinterface(T) # interface functions like gpu_call, threadidx, etc
TestSuite.test_base(T) # basic functionality like launching a kernel on the GPU and Base operations
TestSuite.test_blas(T) # tests the blas interface
TestSuite.test_broadcasting(T) # tests the broadcasting implementation
TestSuite.test_construction(T) # tests all kinds of different ways of constructing the array
TestSuite.test_linalg(T) # linalg function tests
TestSuite.test_mapreduce(T) # mapreduce sum, etc
TestSuite.test_indexing(T) # indexing tests
TestSuite.test_random(T) # randomly constructed arrays
TestSuite.test_io(T)
```
