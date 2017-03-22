# GPUArrays

[![Build Status](https://travis-ci.org/SimonDanisch/GPUArrays.jl.svg?branch=master)](https://travis-ci.org/SimonDanisch/GPUArrays.jl)


Prototype for a GPU Array library. 
It implements the Base AbstractArray interface for Julia's various GPU backends.

#### Main type:

```Julia
type GPUArray{T, N, B, C} <: DenseArray{T, N}
    buffer::B # GPU buffer, allocated by context
    size::NTuple{N, Int} # size of the array
    context::C # GPU context
end
```

#### Scope

Backends: OpenCL, CUDA
Planned backends: OpenGL, Vulkan

Implemented for all backends:
```Julia
map(f, ::GPUArray...)
map!(f, dest::GPUArray, ::GPUArray...)

# maps
mapidx(f, A::GPUArray, args...) do idx, a, args...
    # e.g
    if idx < length(A)
        a[idx+1] = a[idx]
    end
end


broadcast(f, ::GPUArray...)
broadcast!(f, dest::GPUArray, ::GPUArray...)

stencil(f, window::Shape, ::GPUArray...)

# GPUArray's must be "registered" if you want to use them in the loop body
# will translate into map_idx(A, B)
@gpu A::GPUArray, B::GPUArray for loop_head
    body
end

```
Currently, the compilation of the Julia function `f` is done for CUDA by [CUDAnative.jl](https://github.com/JuliaGPU/CUDAnative.jl/)
and for OpenCL and OpenGL [Transpiler.jl](https://github.com/SimonDanisch/Transpiler.jl) will be used.
In the further future it's planned to replace the transpiler by the same approach
CUDAnative.jl is using (via LLVM + SPIR-V).

CLFFT, CUFFT, CLBLAS and CUBLAS will soon be supported.
A prototype of the support can be found here: https://github.com/JuliaGPU/GPUArrays.jl/blob/sd/glsl/src/blas.jl

