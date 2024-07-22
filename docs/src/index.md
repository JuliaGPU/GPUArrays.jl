# GPUArrays.jl

GPUArrays is a package that provides reusable GPU array functionality for Julia's various
GPU backends. Think of it as the `AbstractArray` interface from Base, but for GPU array
types. It allows you to write generic julia code for all GPU platforms and implements common
algorithms for the GPU. Like Julia Base, this includes BLAS wrapper, FFTs, maps, broadcasts
and mapreduces. So when you inherit from GPUArrays and overload the interface correctly, you
will get a lot of functionality for free. This will allow to have multiple GPUArray
implementation for different purposes, while maximizing the ability to share code.

**This package is not intended for end users!** Instead, you should use one of the packages
that builds on GPUArrays.jl such as [CUDA](https://github.com/JuliaGPU/CUDA.jl), [AMDGPU](https://github.com/JuliaGPU/AMDGPU.jl), [OneAPI](https://github.com/JuliaGPU/oneAPI.jl), or [Metal](https://github.com/JuliaGPU/Metal.jl).

This documentation is meant for users who might wish to implement a version of GPUArrays for another GPU backend and will cover the features you will need
to implement, the functionality you gain by doing so, and the test suite that is available
to verify your implementation. GPUArrays.jl also provides a reference implementation of
these interfaces on the CPU: The `JLArray` array type uses Julia's parallel programming
functionality to simulate GPU execution, and will be used throughout this documentation to
illustrate functionality.
