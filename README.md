# GPUArrays

[![Build Status](https://travis-ci.org/SimonDanisch/GPUArrays.jl.svg?branch=master)](https://travis-ci.org/SimonDanisch/GPUArrays.jl)


GPU Array prototype, implementing the Base AbstractArray interface for Julia's various GPU backends.
For now, this is mainly to establish a baseline for our GPU adventures, but might grow into a mature and performant cross-plattform GPU Array library!

#### Main type:

```Julia
type GPUArray{T, N, B, C} <: DenseArray{T, N}
    buffer::B # GPU buffer, allocated by context
    size::NTuple{N, Int} # size of the array
    context::C # GPU context
end
```

#### Scope
Planned backends: OpenGL, OpenCL, Vulkan and CUDA

To be implemented for all backends:
```Julia
map(f, ::GPUArray...)
map!(f, dest::GPUArray, ::GPUArray...)

# maps
map_idx(f, A::GPUArray, args...) #= do idx, a, args...
    e.g
    if idx < length(A)
        a[idx+1] = a[idx]
    end
end
=#


broadcast(f, ::GPUArray...)
broadcast!(f, dest::GPUArray, ::GPUArray...)

stencil(f, window::Shape, ::GPUArray...)

# GPUArray's must be "registered" if you want to use them in the loop body
# will translate into map_idx(A, B)
@gpu A::GPUArray, B::GPUArray for loop_head
    body
end

```
Currently, the compilation of the Julia function `f` is done for CUDA by CUDAnative.jl
and for OpenCL and OpenGL a simple transpiler is used so far.
In the further future it's planned to replace the transpiler by the same approach
CUDAnative.jl is doing.

Furthermore, OpenGL interop with CUDA and OpenCL are going to be fully supported.
Also, it would be nice to hook up all the libraries like CLFFT, CUFFT, CLBLAS and CUBLAS.
