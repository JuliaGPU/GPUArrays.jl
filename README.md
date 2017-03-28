# GPUArrays

[![Build Status](https://travis-ci.org/JuliaGPU/GPUArrays.jl.svg?branch=master)](https://travis-ci.org/SimonDanisch/GPUArrays.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/2aa4bvmq7e9rh338/branch/master?svg=true)](https://ci.appveyor.com/project/SimonDanisch/gpuarrays-jl-8n74h/branch/master)

Prototype for a GPU Array library.
It implements the Base AbstractArray interface for Julia's various GPU backends.

We're using Julia's JIT to generate optimized kernels for map/broadcast operations.
The compilation for the GPU is done with [CUDAnative.jl](https://github.com/JuliaGPU/CUDAnative.jl/)
and for OpenCL and OpenGL [Transpiler.jl](https://github.com/SimonDanisch/Transpiler.jl) is used.
In the further future it's planned to replace the transpiler by the same approach
CUDAnative.jl is using (via LLVM + SPIR-V).

This allows to get more involved functionality, like complex arithmetic, for free, since we can compile what's already in Julia Base.

GPUArrays relies heavily on dot broadcasting. The great thing about dot broadcasting in Julia is, that it [actually fuses operations syntactically](http://julialang.org/blog/2017/01/moredots), which is vital for performance on the GPU.
E.g.:

```Julia
out .= a .+ b ./ c .+ 1
```

Will result in one GPU kernel call to a function that combines the operations without any extra allocations.
This allows GPUArrays to offer a lot of functionality with minimal code.

#### Main type:

```Julia
type GPUArray{T, N, B, C} <: DenseArray{T, N}
    buffer::B # GPU buffer, allocated by context
    size::NTuple{N, Int} # size of the array
    context::C # GPU context
end
```

#### Scope

Current backends: OpenCL, CUDA

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

```

# Usage

```Julia
using GPUArrays
# A backend will be initialized by default on first call to the GPUArray constructor
# But can be explicitely called like e.g.: CLBackend.init(), CUBackend.init(), JLBackend.init()

a = GPUArray(rand(Float32, 32, 32)) # can be constructed from any Julia Array
b = similar(a) # similar and other Julia.Base operations are defined
b .= a .+ 1f0 # broadcast in action, only works on 0.6 for .+. on 0.5 do: b .= (+).(a, 1f0)!
c = a * b # calls to BLAS
function test(a, b)
    Complex64(sin(a / b))
end
complex_c = test.(c, b)
fft!(complex_c) # fft!/ifft! is currently implemented for JLBackend and CLBackend

```


CLFFT, CUFFT, CLBLAS and CUBLAS will soon be supported.
A prototype of generic support of these libraries can be found in [blas.jl](https://github.com/JuliaGPU/GPUArrays.jl/blob/sd/glsl/src/blas.jl).
The OpenCL backend already supports mat mul via `CLBLAS.gemm!` and `fft!`/`ifft!`.
CUDAnative could support these easily as well, but we currently run into problems with the interactions of `CUDAdrv` and `CUDArt`.


# Benchmarks

We have only benchmarked Blackscholes and not much time has been spent to optimize our kernels yet.
So please treat these numbers with care!

[source](https://github.com/JuliaGPU/GPUArrays.jl/blob/master/examples/blackscholes.jl)

![blackscholes](https://github.com/JuliaGPU/GPUArrays.jl/blob/master/examples/blackscholebench.png?raw=true)

Interestingly, on the GTX950, the CUDAnative backend outperforms the OpenCL backend by a factor of 10.
This is most likely due to the fact, that LLVM is great at unrolling and vectorizing loops,
while it seems that the nvidia OpenCL compiler isn't. So with our current primitive kernel,
quite a bit of performance is missed out with OpenCL right now!
This can be fixed by putting more effort into emitting specialized kernels, which should
be straightforward with Julia's great meta programming and `@generated` functions.


Times in a table:

| Backend | Time in Seconds N = 10^7 |
| ---- | ---- |
| OpenCL FirePro W9100 | 8.138e-6 |
| CUDA GTX950 | 0.00354474 |
| OpenCL GTX950 | 0.0335097 |
| OpenCL hd4400 | 0.0420179 |
| 8 Threads i7-6700 | 0.199975 |
| 4 Threads i3-4130 | 0.374679 |
| Julia i7-6700 | 0.937901 |
| Julia i3-4130 | 1.04109 |

# TODO / up for grabs

* mapreduce (there is a first working version for cudanative)
* stencil operations
* more tests and benchmarks
* tests, that actually only switch the backend but use the same code
* performance improvements!!
* implement push!, append!, resize!, getindex, setindex!
* interop between OpenCL, CUDA and OpenGL is there as a protype, but needs proper hooking up via `Base.copy!` / `convert`
* share implementation of broadcast etc between backends. Currently they don't, since there are still subtle differences which should be elimated over time!
