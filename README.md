# GPUArrays

[![Build Status](https://travis-ci.org/JuliaGPU/GPUArrays.jl.svg?branch=master)](https://travis-ci.org/JuliaGPU/GPUArrays.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/2aa4bvmq7e9rh338/branch/master?svg=true)](https://ci.appveyor.com/project/SimonDanisch/gpuarrays-jl-8n74h/branch/master)

GPU Array package for Julia's various GPU backends.
The compilation for the GPU is done with [CUDAnative.jl](https://github.com/JuliaGPU/CUDAnative.jl/)
and for OpenCL [Transpiler.jl](https://github.com/SimonDanisch/Transpiler.jl) is used.
In the future it's planned to replace the transpiler by a similar approach
CUDAnative.jl is using (via LLVM + SPIR-V).

# Why another GPU array package in yet another language?

Julia offers countless advantages for a GPU array package.
E.g., we can use Julia's JIT to generate optimized kernels for map/broadcast operations.

This works even for things like complex arithmetic, since we can compile what's already in Julia Base.

GPUArrays relies heavily on Julia's dot broadcasting.
The great thing about dot broadcasting in Julia is, that it
[actually fuses operations syntactically](http://julialang.org/blog/2017/01/moredots), which is vital for performance on the GPU.
E.g.:

```Julia
out .= a .+ b ./ c .+ 1
#turns into this one broadcast (map):
broadcast!(out, a, b, c) do a, b, c
    a + b / c + 1
end
```

Will result in one GPU kernel call to a function that combines the operations without any extra allocations.
This allows GPUArrays to offer a lot of functionality with minimal code.

Also, when compiling Julia to the GPU, we can use all the cool features from Julia, e.g.
higher order functions, multiple dispatch, meta programming and generated functions.
Checkout the examples, to see how this can be used to emit specialized code while not loosing flexibility:
[unrolling](https://github.com/JuliaGPU/GPUArrays.jl/blob/master/examples/juliaset.jl),
[vector loads/stores](https://github.com/JuliaGPU/GPUArrays.jl/blob/master/examples/vectorload.jl)

In theory, we could go as far as inspecting user defined callbacks (we can get the complete AST), count operations and estimate register usage and use those numbers to optimize our kernels!

### Automatic Differentiation

Because of neuronal netorks, automatic differentiation is super hyped right now!
Julia offers a couple of packages for that, e.g. [ReverseDiff](https://github.com/JuliaDiff/ReverseDiff.jl).
It heavily relies on Julia's strength to specialize generic code and dispatch to different implementations depending on the Array type, allowing an almost overheadless automatic differentiation.
Making this work with GPUArrays will be a bit more involved, but the
first [prototype](https://github.com/JuliaGPU/GPUArrays.jl/blob/master/examples/logreg.jl) looks already promising!
There is also [ReverseDiffSource](https://github.com/JuliaDiff/ReverseDiffSource.jl), which should already work for simple functions.

#### Main type:

```Julia
type GPUArray{T, N, B, C} <: DenseArray{T, N}
    buffer::B # GPU buffer, allocated by context
    size::NTuple{N, Int} # size of the array
    context::C # GPU context
end
```

#### Scope

Current backends: OpenCL, CUDA, Julia Threaded

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

# calls `f` on args, with queues, block heuristics and context taken from `array`
# f can be a julia function or a tuple (String, Symbol),
# being a C kernel source string + the name of the kernel function
gpu_call(array::GPUArray, f, args::Tuple)

```
Example for [gpu_call](https://github.com/JuliaGPU/GPUArrays.jl/blob/master/examples/custom_kernels.jl)

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
A prototype of generic support of these libraries can be found in [blas.jl](https://github.com/JuliaGPU/GPUArrays.jl/blob/master/src/backends/blas.jl).
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
| OpenCL FirePro W9100 | 0.0014526 |
| CUDA GTX950 | 0.00354474 |
| OpenCL GTX950 | 0.0335097 |
| OpenCL hd4400 | 0.0420179 |
| 8 Threads i7-6700 | 0.199975 |
| 4 Threads i3-4130 | 0.374679 |
| Julia i7-6700 | 0.937901 |
| Julia i3-4130 | 1.04109 |

# TODO / up for grabs

* stencil operations
* more tests and benchmarks
* tests, that only switch the backend but use the same code
* performance improvements!!
* implement push!, append!, resize!, getindex, setindex!
* interop between OpenCL, CUDA and OpenGL is there as a protype, but needs proper hooking up via `Base.copy!` / `convert`
* share implementation of broadcast etc between backends. Currently they don't, since there are still subtle differences which should be eliminated over time!


# Installation

For the cudanative backend, you need to install [CUDAnative.jl manually](https://github.com/JuliaGPU/CUDAnative.jl/#installation).
The cudanative backend only works on 0.6, while the other backends also support Julia 0.5.
Make sure to have CUDA and OpenCL driver installed correctly.
`Pkg.build("GPUArrays")` will pick those up and should include the working backends.
So if your system configuration changes, make sure to run `Pkg.build("GPUArrays")` again.
The rest should work automatically:
```Julia
Pkg.add("GPUArrays")
Pkg.build("GPUArrays") # should print out information about what backends are added
# Test it!
Pkg.test("GPUArrays")
```
