# GPUArrays Documentation


# Abstract GPU interface

GPUArrays supports different platforms like CUDA and OpenCL, which all have different
names for function that offer the same functionality on the hardware.
E.g. how to call a function on the GPU, how to get the thread index etc.
GPUArrays offers an abstract interface for these functions which are overloaded
by the packages like [CLArrays](https://github.com/JuliaGPU/CLArrays.jl) and [CuArrays](https://github.com/JuliaGPU/CuArrays.jl).
This makes it possible to write generic code that can be run on all hardware.
GPUArrays itself even contains a pure Julia implementation of this interface.
The julia reference implementation is also a great way to debug your GPU code, since it
offers many more errors and debugging information compared to the GPU backends - which
mostly silently error or give cryptic errors (so far).
You can use the reference implementation by using the `GPUArrays.JLArray` type.

The functions that are currently part of the interface:

The low level dim + idx function, with a similar naming as in CUDA (with `*` indicating `(x, y, z)`):
```Julia
blockidx_*(state), blockdim_*(state), threadidx_*(state), griddim_*(state)
# Known in OpenCL as:
get_group_id,      get_local_size,    get_local_id,       get_num_groups
```

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
