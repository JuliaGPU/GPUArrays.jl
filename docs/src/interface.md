# Interface

To extend the above functionality to a new array type, you should use the types and
implement the interfaces listed on this page. GPUArrays is design around having two
different array types to represent a GPU array: one that only ever lives on the host, and
one that actually can be instantiated on the device (i.e. in kernels).

## Host-side

Your host-side array type should build on the `AbstractGPUArray` supertype:

```@docs
AbstractGPUArray
```

First of all, you should implement operations that are expected to be defined for any
`AbstractArray` type. Refer to the Julia manual for more details, or look at the `JLArray`
reference implementation.

To be able to actually use the functionality that is defined for `AbstractGPUArray`s, you
should provide implementations of the following interfaces:

```@docs
GPUArrays.unsafe_reinterpret
```

### Devices

```@docs
GPUArrays.device
GPUArrays.synchronize
```

### Execution

```@docs
GPUArrays.AbstractGPUBackend
GPUArrays.backend
```

```@docs
GPUArrays._gpu_call
```

### Linear algebra

```@docs
GPUArrays.blas_module
GPUArrays.blasbuffer
```


## Device-side

To work with GPU memory on the device itself, e.g. within a kernel, we need a different
type: Most functionality will behave differently when running on the GPU, e.g., accessing
memory directly instead of copying it to the host. We should also take care not to call into
any host library, such as the Julia runtime or the system's math library.

```@docs
AbstractDeviceArray
```

Your device array type should again implement the core elements of the `AbstractArray`
interface, such as indexing and certain getters. Refer to the Julia manual for more details,
or look at the `JLDeviceArray` reference implementation.

You should also provide implementations of several "GPU intrinsics". To make sure the
correct implementation is called, the first argument to these intrinsics will be the kernel
state object from before.

```@docs
GPUArrays.LocalMemory
GPUArrays.synchronize_threads
GPUArrays.blockidx_x
GPUArrays.blockidx_y
GPUArrays.blockidx_z
GPUArrays.blockdim_x
GPUArrays.blockdim_y
GPUArrays.blockdim_z
GPUArrays.threadidx_x
GPUArrays.threadidx_y
GPUArrays.threadidx_z
GPUArrays.griddim_x
GPUArrays.griddim_y
GPUArrays.griddim_z
```
