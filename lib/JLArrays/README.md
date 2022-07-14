# JLArrays.jl

This package serves as a reference implementation for [GPUArrays.jl](https://github.com/JuliaGPU/GPUArrays.jl),
running on the CPU. It was used internally for tests, and is now a registered sub-package to enable easier use
in testing other packages.

```julia
julia> using JLArrays
[ Info: Precompiling JLArrays [27aeb0d3-9eb9-45fb-866b-73c2ecf80fcb]

julia> jl(ones(3)) .+ 1
3-element JLArray{Float64, 1}:
 2.0
 2.0
 2.0

julia> ans[2] 
┌ Warning: Performing scalar indexing on task Task (runnable) @0x0000000108fc4010.
│ Invocation of getindex resulted in scalar indexing of a GPU array.
│ This is typically caused by calling an iterating implementation of a method.
│ Such implementations *do not* execute on the GPU, but very slowly on the CPU,
│ and therefore are only permitted from the REPL for prototyping purposes.
│ If you did intend to index this array, annotate the caller with @allowscalar.
└ @ GPUArraysCore ~/.julia/packages/GPUArraysCore/rSIl2/src/GPUArraysCore.jl:81
2.0
```