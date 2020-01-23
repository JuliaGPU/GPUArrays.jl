# GPUArrays

*Reusable GPU array functionality for Julia's various GPU backends.*

| **Documentation**                                                         | **Build Status**                                                                            |
|:-------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![][gitlab-img]][gitlab-url] [![][travis-img]][travis-url] [![][codecov-img]][codecov-url] |

[gitlab-img]: https://gitlab.com/JuliaGPU/CuArrays.jl/badges/master/pipeline.svg
[gitlab-url]: https://gitlab.com/JuliaGPU/CuArrays.jl/commits/master

[travis-img]: https://api.travis-ci.org/JuliaGPU/GPUArrays.jl.svg?branch=master
[travis-url]: https://travis-ci.org/JuliaGPU/GPUArrays.jl

[codecov-img]: https://codecov.io/gh/JuliaGPU/CuArrays.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/JuliaGPU/CuArrays.jl

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: http://JuliaGPU.github.io/GPUArrays.jl/stable/

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: http://JuliaGPU.github.io/GPUArrays.jl/dev/

This package is the counterpart of Julia's `AbstractArray` interface, but for GPU array
types: It provides functionality and tooling to speed-up development of new GPU array types.
**This package is not intended for end users!** Instead, you should use one of the packages
that builds on GPUArrays.jl, such as [CuArrays](https://github.com/JuliaGPU/CuArrays.jl).
