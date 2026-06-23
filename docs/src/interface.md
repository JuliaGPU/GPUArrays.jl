# Interface

To extend the above functionality to a new array type, you should use the types and
implement the interfaces listed on this page. GPUArrays is designed around having two
different array types to represent a GPU array: one that exists only on the host, and
one that actually can be instantiated on the device (i.e. in kernels).
Device functionality is then handled by [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).

## Host abstractions

You should provide an array type that builds on the `AbstractGPUArray` supertype, such as:

```julia
mutable struct CustomArray{T, N} <: AbstractGPUArray{T, N}
    data::DataRef{Vector{UInt8}}
    offset::Int
    dims::Dims{N}
    ...
end

```

This will allow your defined type (in this case `JLArray`) to use the GPUArrays interface where available.
To be able to actually use the functionality that is defined for `AbstractGPUArray`s, you need to define the backend, like so:

```julia
import KernelAbstractions: Backend
struct CustomBackend <: KernelAbstractions.GPU
KernelAbstractions.get_backend(a::CA) where CA <: CustomArray = CustomBackend()
```

There are numerous examples of potential interfaces for GPUArrays, such as with [JLArrays](https://github.com/JuliaGPU/GPUArrays.jl/blob/main/lib/JLArrays/src/JLArrays.jl), [CuArrays](https://github.com/JuliaGPU/CUDA.jl/blob/main/src/gpuarrays.jl), and [ROCArrays](https://github.com/JuliaGPU/AMDGPU.jl/blob/main/src/gpuarrays.jl).

## Sparse arrays

Sparse arrays cannot share the `AbstractGPUArray` supertype: that is a
`DenseArray`, whereas a sparse array must be an `AbstractSparseArray`. GPUArrays
therefore provides a parallel hierarchy, and generic sparse functionality
(broadcast, `mapreduce`, sparse/dense matrix multiplication, `norm`/`opnorm`,
`findnz`, `triu`/`tril`/`kron`, format conversion, indexing) is defined on it.

A back-end provides one mutable struct per supported format, subtyping the
matching abstract type and using the conventional field names (the generic code
reads them directly):

| supertype | fields |
|:--|:--|
| `AbstractGPUSparseVector{Tv,Ti}`    | `iPtr`, `nzVal`, `len`, `nnz` |
| `AbstractGPUSparseMatrixCSC{Tv,Ti}` | `colPtr`, `rowVal`, `nzVal`, `dims`, `nnz` |
| `AbstractGPUSparseMatrixCSR{Tv,Ti}` | `rowPtr`, `colVal`, `nzVal`, `dims`, `nnz` |
| `AbstractGPUSparseMatrixCOO{Tv,Ti}` | `rowInd`, `colInd`, `nzVal`, `dims`, `nnz` |

where the index/pointer/value arrays are the back-end's own dense vector type
(so `nonzeros(A)`, `SparseArrays.nonzeroinds`, `rowvals`, `getcolptr` work, and
`get_backend(nonzeros(A))` identifies the compute backend).

On top of the structs, a back-end implements:

  * constructors from the component arrays (e.g.
    `MyCSR(rowPtr, colVal, nzVal, dims)`) and the inter-format and
    dense↔sparse conversions (`MyCSR(::MyCOO)`, `MyCSC(::SparseMatrixCSC)`,
    `SparseMatrixCSC(::MyCSC)`, …);

  * `Base.similar` — structure-preserving (`similar(A)`, `similar(A, ::Type)`)
    and empty-of-a-shape (`similar(A, ::Type, dims)`), exactly as for dense
    arrays. The generic algorithms allocate their outputs through `similar`
    rather than by naming a type;

  * the format-conversion verbs `GPUArrays.sparse_csc`, `GPUArrays.sparse_csr`
    and `GPUArrays.sparse_coo`, for the formats other than the one each
    produces (converting to the format already held is the generic identity).
    These are the value-level equivalent of `SparseArrays.sparse` and back the
    generic code that needs a particular layout (e.g. matrix multiplication
    funnels through COO).

Dense↔sparse conversion is provided generically and on-device: `to_sparse(::Type{ST},
A)` scans a dense array into a sparse one (`ST` is an `AbstractGPUSparseVector` or
`AbstractGPUSparseMatrixCOO` -- CSR/CSC follow via the conversion verbs), and `to_dense(A)`
is the inverse, scattering a sparse array into a dense one of the same back-end. A
back-end's dense↔sparse constructors (`MyArray(::MySparse…)`, `MyCSR(::MyDense)`) can be
defined in terms of these.

## Caching Allocator

```@docs
GPUArrays.@cached
GPUArrays.@uncached
```
