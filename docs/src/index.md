# GPUArrays.jl

GPUArrays is a package that provides reusable GPU array functionality for Julia's various
GPU backends. Think of it as the `AbstractArray` interface from Base, but for GPU array
types. It allows you to write generic julia code for all GPU platforms and implements common
algorithms for the GPU. Like Julia Base, this includes BLAS wrapper, FFTs, maps, broadcasts
and mapreduces. So when you inherit from GPUArrays and overload the interface correctly, you
will get a lot of functionality for free. This will allow to have multiple GPUArray
implementation for different purposes, while maximizing the ability to share code.

**This package is not intended for end users!** Instead, you should use one of the packages
that builds on GPUArrays.jl. There is currently only a single package that actively builds
on these interfaces, namely [CuArrays.jl](https://github.com/JuliaGPU/CuArrays.jl).

In this documentation, you will find more information on the interface that you are expected
to implement, the functionality you gain by doing so, and the test suite that is available
to verify your implementation. GPUArrays.jl also provides a reference implementation of
these interfaces on the CPU: The `JLArray` array type uses Julia's parallel programming
functionality to simulate GPU execution, and will be used throughout this documentation to
illustrate functionality.
