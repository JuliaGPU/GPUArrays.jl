# GPUArrays

*Reusable GPU array functionality for Julia's various GPU backends.*

| **Documentation**                                                         | **Build Status**                                                                                   | **Coverage**                    |
|:-------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------:|:-------------------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![][gh-img]][gh-url] [![][buildkite-img]][buildkite-url] [![PkgEval][pkgeval-img]][pkgeval-url] | [![][codecov-img]][codecov-url] |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: http://JuliaGPU.github.io/GPUArrays.jl/stable/

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: http://JuliaGPU.github.io/GPUArrays.jl/dev/

[gitlab-img]: https://gitlab.com/JuliaGPU/GPUArrays.jl/badges/master/pipeline.svg
[gitlab-url]: https://gitlab.com/JuliaGPU/GPUArrays.jl/commits/master

[gh-img]: https://github.com/JuliaGPU/GPUArrays.jl/actions/workflows/Test.yml/badge.svg?branch=master
[gh-url]: https://github.com/JuliaGPU/GPUArrays.jl/actions/workflows/Test.yml

[buildkite-img]: https://badge.buildkite.com/05f9b27c5ce6c3906566fb66cfc42d44586e16d88a805a0b7b.svg?branch=master
[buildkite-url]: https://buildkite.com/julialang/gpuarrays-dot-jl

[pkgeval-img]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/G/GPUArrays.svg
[pkgeval-url]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/G/GPUArrays.html

[codecov-img]: https://codecov.io/gh/JuliaGPU/GPUArrays.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/JuliaGPU/GPUArrays.jl

This package is the counterpart of Julia's `AbstractArray` interface, but for GPU array
types: It provides functionality and tooling to speed-up development of new GPU array types.
**This package is not intended for end users!** Instead, you should use one of the packages
that builds on GPUArrays.jl, such as [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl), [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl), [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl), or [Metal.jl](https://github.com/JuliaGPU/Metal.jl).
