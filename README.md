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

## Interface methods

To support a new GPU backend, you will need to implement various interface methods for your backend's array types.
Some (CPU based) examples can be see in the testing library `JLArrays` (located in the `lib` directory of this package).

### Dense array support

### Sparse array support (optional)

`GPUArrays.jl` provides **device-side** array types for `CSC`, `CSR`, `COO`, and `BSR` matrices, as well as sparse vectors.
It also provides abstract types for these layouts that you can create concrete child types of in order to benefit from the
backend-agnostic wrappers. In particular, `GPUArrays.jl` provides out-of-the-box support for broadcasting and `mapreduce` over
GPU sparse arrays.

For **host-side** types, your custom sparse types should implement:

- `dense_array_type` - the corresponding dense array type. For example, for a `CuSparseVector` or `CuSparseMatrixCXX`, the `dense_array_type` is `CuArray`
- `sparse_array_type` - the **untyped** sparse array type corresponding to a given parametrized type. A `CuSparseVector{Tv, Ti}` would have a `sparse_array_type` of `CuSparseVector` -- note the lack of type parameters!
- `csc_type(::Type{T})` - the compressed sparse column type for your backend. A `CuSparseMatrixCSR` would have a `csc_type` of `CuSparseMatrixCSC`. 
- `csr_type(::Type{T})` - the compressed sparse row type for your backend. A `CuSparseMatrixCSC` would have a `csr_type` of `CuSparseMatrixCSR`. 
- `coo_type(::Type{T})` - the coordinate sparse matrix type for your backend. A `CuSparseMatrixCSC` would have a `coo_type` of `CuSparseMatrixCOO`.

To use `SparseArrays.findnz`, your host-side type **must** implement `sortperm`. This can be done with scalar indexing, but will be very slow.

Additionally, you need to teach `GPUArrays.jl` how to translate your backend's specific types onto the device. `GPUArrays.jl` provides the device-side types:

- `GPUSparseDeviceVector`
- `GPUSparseDeviceMatrixCSC`
- `GPUSparseDeviceMatrixCSR`
- `GPUSparseDeviceMatrixBSR`
- `GPUSparseDeviceMatrixCOO`

You will need to create a method of `Adapt.adapt_structure` for each format your backend supports. **Note** that if your backend supports separate address spaces,
as CUDA and ROCm do, you need to provide a parameter to these device-side arrays to indicate in which address space the underlying pointers live. An example of adapting
an array to the device-side struct:

```julia
function GPUArrays.GPUSparseDeviceVector(iPtr::MyDeviceVector{Ti, A},
                                         nzVal::MyDeviceVector{Tv, A},
                                         len::Int,
                                         nnz::Ti) where {Ti, Tv, A}
    GPUArrays.GPUSparseDeviceVector{Tv, Ti, MyDeviceVector{Ti, A}, MyDeviceVector{Tv, A}, A}(iPtr, nzVal, len, nnz)
end

function Adapt.adapt_structure(to::MyAdaptor, x::MySparseVector)
    return GPUArrays.GPUSparseDeviceVector(
        adapt(to, x.iPtr),
        adapt(to, x.nzVal),
        length(x), x.nnz
    )
end
```

You'll also need to inform `GPUArrays.jl` and `GPUCompiler.jl` how to adapt your sparse arrays by extending `KernelAbstractions.jl`'s `get_backend()`:

```julia
KA.get_backend(::MySparseVector) = MyBackend()
```  
