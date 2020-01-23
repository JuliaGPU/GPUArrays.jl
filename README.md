# GPUArrays

*Abstract GPU array functionality for Julia's various GPU backends.*

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


# Functionality

The GPUArrays.jl package essentially provides two abstract array types: `AbstractGPUArray`
for GPU arrays that live on the hose, and `AbstractDeviceArray` for the device-side
counterpart.

## `AbstractGPUArray`

TODO: describe functionality

## `AbstractDeviceArray`

TODO: describe functionality


# Interfaces

To extend the above functionality to a new array type, you should implement the following
interfaces:

TODO


# Test suite

GPUArrays also provides an extensive test suite that covers all of the functionality that
should be available after implementing the required interfaces. This test suite is part of
this package, but for dependency reasons it is not available when importing the package.
Instead, you should include the code from your `runtests.jl` as follows:

```julia
import GPUArrays
gpuarrays = pathof(GPUArrays)
gpuarrays_root = dirname(dirname(gpuarrays))
include(joinpath(gpuarrays_root, "test", "testsuite.jl"))
```

This however implies that the test system will not know about extra dependencies that are
required by the test suite. To remedy this, you should add the following dependencies to
your `Project.toml`:

```
[extras]
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
...

[targets]
test = [..., "FFTW", "ForwardDiff", "FillArrays"]
```


# `JLArray`

The `JLArray` type is a reference implementation of the GPUArray interfaces. It does not run
on the GPU, but rather uses Julia's async constructs as its backend. It is constructed as
follows:

```julia
gA = JLArray(A)
```
