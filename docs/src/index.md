# GPUArrays Documentation

GPUArrays is an abstract interface for GPU computations.
Think of it as the AbstractArray interface in Julia Base but for GPUs.
It allows you to write generic julia code for all GPU platforms and implements common algorithms for the GPU.
Like Julia Base, this includes BLAS wrapper, FFTs, maps, broadcasts and mapreduces.
So when you inherit from GPUArrays and overload the interface correctly, you will get a lot
of functionality for free.
This will allow to have multiple GPUArray implementation for different purposes, while
maximizing the ability to share code.
Currently there are two packages implementing the interface namely [CLArrays](https://github.com/JuliaGPU/CLArrays.jl) and [CuArrays](https://github.com/JuliaGPU/CuArrays.jl).
As the name suggests, the first implements the interface using OpenCL and the latter uses CUDA.



# The Abstract GPU interface

Different GPU computation frameworks like CUDA and OpenCL, have different
names for accessing the same hardware functionality.
E.g. how to launch a GPU Kernel, how to get the thread index and so forth.
GPUArrays offers a unified abstract interface for these functions.
This makes it possible to write generic code that can be run on all hardware.
GPUArrays itself even contains a pure [Julia implementation](https://github.com/JuliaGPU/GPUArrays.jl/blob/master/src/jlbackend.jl) of this interface.
The julia reference implementation is a great way to debug your GPU code, since it
offers more informative errors and debugging information compared to the GPU backends - which
mostly silently error or give cryptic errors (so far).

You can use the reference implementation by using the `GPUArrays.JLArray` type.

The functions that are currently part of the interface:

The low level dim + idx function, with a similar naming scheme as in CUDA:
```Julia
# with * being either of x, y or z
blockidx_*(state), blockdim_*(state), threadidx_*(state), griddim_*(state)
# Known in OpenCL as:
get_group_id,      get_local_size,    get_local_id,       get_num_groups
```

Higher level functionality:

```@docs
gpu_call(f, A::GPUArray, args::Tuple, configuration = length(A))

linear_index(state)

global_size(state)

@linearidx(A, statesym = :state)

@cartesianidx(A, statesym = :state)

synchronize_threads(state)

device(A::AbstractArray)

synchronize(A::AbstractArray)
```


# The abstract TestSuite

Since all array packages inheriting from GPUArrays need to offer the same functionality
and interface, it makes sense to test them in the same way.
This is why GPUArrays contains a test suite which can be called with the array type
you want to test.

You can run the test suite like this:

```Julia
using GPUArrays, GPUArrays.TestSuite
TestSuite.run_tests(MyGPUArrayType)
```
If you don't want to run the whole suite, you can also run parts of it:


```Julia
Typ = JLArray
GPUArrays.allowslow(false) # fail tests when slow indexing path into Array type is used.

TestSuite.run_gpuinterface(Typ) # interface functions like gpu_call, threadidx, etc
TestSuite.run_base(Typ) # basic functionality like launching a kernel on the GPU and Base operations
TestSuite.run_blas(Typ) # tests the blas interface
TestSuite.run_broadcasting(Typ) # tests the broadcasting implementation
TestSuite.run_construction(Typ) # tests all kinds of different ways of constructing the array
TestSuite.run_fft(Typ) # fft tests
TestSuite.run_linalg(Typ) # linalg function tests
TestSuite.run_mapreduce(Typ) # mapreduce sum, etc
TestSuite.run_indexing(Typ) # indexing tests
```
