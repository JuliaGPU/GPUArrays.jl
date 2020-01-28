# Interface

To extend the above functionality to a new array type, you should use the types and
implement the interfaces listed on this page. GPUArrays is design around having two
different array types to represent a GPU array: one that only ever lives on the host, and
one that actually can be instantiated on the device (i.e. in kernels).


## Device functionality

Several types and interfaces are related to the device and execution of code on it. First of
all, you need to provide a type that represents your device and exposes some properties of
it:

```@docs
GPUArrays.AbstractGPUDevice
GPUArrays.threads
```

Another important set of interfaces relates to executing code on the device:

```@docs
GPUArrays.AbstractGPUBackend
GPUArrays.AbstractKernelContext
GPUArrays.gpu_call
GPUArrays.synchronize
GPUArrays.thread_block_heuristic
```

Finally, you need to provide implementations of certain methods that will be executed on the
device itself:

```@docs
GPUArrays.AbstractDeviceArray
GPUArrays.LocalMemory
GPUArrays.synchronize_threads
GPUArrays.blockidx
GPUArrays.blockdim
GPUArrays.threadidx
GPUArrays.griddim
```


## Host abstractions

You should provide an array type that builds on the `AbstractGPUArray` supertype:

```@docs
AbstractGPUArray
```

First of all, you should implement operations that are expected to be defined for any
`AbstractArray` type. Refer to the Julia manual for more details, or look at the `JLArray`
reference implementation.

To be able to actually use the functionality that is defined for `AbstractGPUArray`s, you
should provide implementations of the following interfaces:

```@docs
GPUArrays.backend
GPUArrays.device
GPUArrays.unsafe_reinterpret
```
