# GPUArrays.jl release history

This document lists the noteworthy changes in every breaking release of GPUArrays.jl,
newest first, aimed at the authors of back-end packages. For the complete list of merged
pull requests, see the [GitHub releases](https://github.com/JuliaGPU/GPUArrays.jl/releases).


## v12.0

Base's reductions, sorting, scans, `findall`, logical indexing and `reverse` are now
implemented once for every `AnyGPUArray`, on top of
[AcceleratedKernels.jl](https://github.com/JuliaGPU/AcceleratedKernels.jl) (AK) 0.5. A
back-end gets all of them by defining nothing beyond the array interface and a
KernelAbstractions back-end, and they follow Base's semantics: stable sorting by default,
`init` applied once, no neutral element needed, Base's result types, empty results and
errors.

*Breaking changes for back-ends*:

- The `GPUArrays.mapreducedim!` hook is removed and GPUArrays no longer picks neutral
  elements. Every Base reduction goes to AK directly. Back-ends must delete their methods
  for the hook, together with the reduction kernels only it used.
- `GPUArrays.default_rng` and the deprecated `RNG(state::AbstractGPUArray)` and
  `seed!(rng, ::Vector{UInt32})` methods are removed. Construct the RNG with `RNG{AT}()`
  and seed it with an integer.
- KernelAbstractions 0.9.43 or later and Adapt 4.7.2 or later are required.

*What back-ends should delete*:

Methods narrower than GPUArrays' keep winning dispatch, so a back-end that keeps its own
`findall`, `to_index`/`to_indices`, `_accumulate!`/`accumulate`, `sort!`/`sortperm!`/
`partialsort!`, or `reverse!` bypasses GPUArrays' Base conformance, and some of those
methods are ambiguous with GPUArrays' (e.g. `findall(f, ::MyArray)` against
`findall(::Fix2{typeof(in)}, ::AnyGPUArray)`). Delete them.

A vendor sort or scan (e.g. one built on MPSGraph) can stay as a fast path, as long as it
falls back to GPUArrays' method for the cases it doesn't handle and accepts AK's
algorithm objects (`alg=AK.MergeSort()` etc.). There is no extension point for vendor
reductions; `GPUArrays._ak_mapreducedim!` is internal.

*Known regressions*:

- Small reductions along `dims` can be slower than the back-ends' own kernels were (up to
  about 4× for a 1000×10 array on CUDA), while large ones are faster. This is tracked in
  [AcceleratedKernels.jl#135](https://github.com/JuliaGPU/AcceleratedKernels.jl/issues/135).

JLArrays 0.4.0 is the matching release of the reference back-end.
