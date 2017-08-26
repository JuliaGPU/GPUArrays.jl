# GPUArrays

[![Build Status](https://travis-ci.org/JuliaGPU/GPUArrays.jl.svg?branch=master)](https://travis-ci.org/JuliaGPU/GPUArrays.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/2aa4bvmq7e9rh338/branch/master?svg=true)](https://ci.appveyor.com/project/SimonDanisch/gpuarrays-jl-8n74h/branch/master)

[Benchmarks](https://github.com/JuliaGPU/GPUBenchmarks.jl/blob/master/results/results.md)

GPU Array package for Julia's various GPU backends.
The compilation for the GPU is done with [CUDAnative.jl](https://github.com/JuliaGPU/CUDAnative.jl/)
and for OpenCL [Transpiler.jl](https://github.com/SimonDanisch/Transpiler.jl) is used.
In the future it's planned to replace the transpiler by a similar approach
CUDAnative.jl is using (via LLVM + SPIR-V).


# Why another GPU array package in yet another language?

Julia offers countless advantages for a GPU array package.
E.g., we can use Julia's JIT to generate optimized kernels for map/broadcast operations.

This works even for things like complex arithmetic, since we can compile what's already in Julia Base.
This isn't restricted to Julia Base, GPUArrays works with all kind of user defined types and functions!

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

Also, when compiling Julia for the GPU, we can use all the cool features from Julia, e.g.
higher order functions, multiple dispatch, meta programming and generated functions.
Checkout the examples, to see how this can be used to emit specialized code while not losing flexibility:
[unrolling](https://github.com/JuliaGPU/GPUArrays.jl/blob/master/examples/juliaset.jl),
[vector loads/stores](https://github.com/JuliaGPU/GPUArrays.jl/blob/master/examples/vectorload.jl)

In theory, we could go as far as inspecting user defined callbacks (we can get the complete AST), count operations and estimate register usage and use those numbers to optimize our kernels!


### Automatic Differentiation

Because of neural networks, automatic differentiation is super hyped right now!
Julia offers a couple of packages for that, e.g. [ReverseDiff](https://github.com/JuliaDiff/ReverseDiff.jl).
It heavily relies on Julia's strength to specialize generic code and dispatch to different implementations depending on the Array type, allowing an almost overheadless automatic differentiation.
Making this work with GPUArrays will be a bit more involved, but the
first [prototype](https://github.com/JuliaGPU/GPUArrays.jl/blob/master/examples/logreg.jl) looks already promising!
There is also [ReverseDiffSource](https://github.com/JuliaDiff/ReverseDiffSource.jl), which should already work for simple functions.

# Scope

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

# calls `f` on args, with queues and context taken from `array`
# f can be a julia function or a tuple (String, Symbol),
# being a C kernel source string + the name of the kernel function.
# first argument needs to be an untyped arg for global state. This can be mostly ignored, but needs to be passed to 
# e.g. `linear_index(array::GPUArray, state)`, which gives you a linear, per thread index into `array` on all backends.
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
fft!(complex_c) # fft!/ifft!/plan_fft, plan_ifft, plan_fft!, plan_ifft!

"""
When you program with GPUArrays, you can just write normal julia functions, feed them to gpu_call and depending on what backend you choose it will use Transpiler.jl or CUDAnative.
"""
#Signature, global_size == cuda blocks, local size == cuda threads
gpu_call(kernel::Function, DispatchDummy::GPUArray, args::Tuple, global_size = length(DispatchDummy), local_size = nothing)
with kernel looking like this:

function kernel(state, arg1, arg2, arg3) # args get splatted into the kernel call
    # state gets always passed as the first argument and is needed to offer the same 
    # functionality across backends, even though they have very different ways of of getting e.g. the thread index
    # arg1 can be any gpu array - this is needed to dispatch to the correct intrinsics.
    # if you call gpu_call without any further modifications to global/local size, this should give you a linear index into 
    # DispatchDummy
    idx = linear_index(state, arg1::GPUArray) 
    arg1[idx] = arg2[idx] + arg3[idx]
    return #kernel must return void
end
```

# Currently supported subset of Julia Code

working with immutable isbits (not containing pointers) type should be completely supported
non allocating code (so no constructs like `x = [1, 2, 3]`). Note that tuples are isbits, so this works x = (1, 2, 3).
Transpiler/OpenCL has problems with putting GPU arrays on the gpu into a struct - so no views and actually no multidimensional indexing. For that `size` is needed which would need to be part of the array struct. A fix for that is in sight, though.

| CLContext: FirePro w9100 | 0.0013 s| 7831 | 789.8|

# TODO / up for grabs

* stencil operations
* more tests and benchmarks
* tests, that only switch the backend but use the same code
* performance improvements!!
* implement push!, append!, resize!, getindex, setindex!
* interop between OpenCL, CUDA and OpenGL is there as a protype, but needs proper hooking up via `Base.copy!` / `convert`
* share implementation of broadcast etc between backends. Currently they don't, since there are still subtle differences which should be eliminated over time!


# Installation

I recently added a lot of features and bug fixes to the master branch.

For the cudanative backend, you need to install [CUDAnative.jl manually](https://github.com/JuliaGPU/CUDAnative.jl/#installation) and it works only on osx + linux with a julia source build.
Make sure to have either CUDA and/or OpenCL drivers installed correctly.
`Pkg.build("GPUArrays")` will pick those up and should include the working backends.
So if your system configuration changes, make sure to run `Pkg.build("GPUArrays")` again.
The rest should work automatically:

```Julia
Pkg.add("GPUArrays")
Pkg.checkout("GPUArrays") # optional but recommended to checkout master branch
Pkg.build("GPUArrays") # should print out information about what backends are added
# Test it!
Pkg.test("GPUArrays")
```
If a backend is not supported by the hardware, you will see build errors while running `Pkg.add("GPUArrays")`.
Since GPUArrays selects only working backends when running `Pkg.build("GPUArrays")`
**these errors can be ignored**.
