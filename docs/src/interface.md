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

A sparse array can't share the `AbstractGPUArray` supertype — that is a `DenseArray`,
whereas a sparse array must be an `AbstractSparseArray` — so GPUArrays keeps a parallel
sparse hierarchy with its own generic functionality. Integrating a back-end has three
parts: the storage types it provides, the methods it implements to plug them in, and the
functionality it then gets for free.

### Storage types to provide

One mutable struct per supported format, subtyping the matching abstract type and using
the conventional field names (generic code reads them directly):

| supertype | fields |
|:--|:--|
| `AbstractGPUSparseVector{Tv,Ti}`    | `iPtr`, `nzVal`, `len`, `nnz` |
| `AbstractGPUSparseMatrixCSC{Tv,Ti}` | `colPtr`, `rowVal`, `nzVal`, `dims`, `nnz` |
| `AbstractGPUSparseMatrixCSR{Tv,Ti}` | `rowPtr`, `colVal`, `nzVal`, `dims`, `nnz` |
| `AbstractGPUSparseMatrixCOO{Tv,Ti}` | `rowInd`, `colInd`, `nzVal`, `dims`, `nnz` |

The pointer/index/value arrays are the back-end's own dense vector type. Provide only the
formats you need, but note that several generic operations route through COO.

### Interface to implement

  * **Constructors** — from component arrays (`MyCSR(rowPtr, colVal, nzVal, dims)`),
    between formats (`MyCSR(::MyCOO)`, …), and to/from host `SparseArrays`
    (`MyCSC(::SparseMatrixCSC)`, `SparseMatrixCSC(::MyCSC)`).
  * **`Base.similar`** — structure-preserving (`similar(A)`, `similar(A, ::Type)`) and
    empty-of-a-shape (`similar(A, ::Type, dims)`), as for dense arrays; generic code
    allocates its outputs through `similar`, never by naming a type.
  * **Conversion verbs** `GPUArrays.sparse_csc`/`sparse_csr`/`sparse_coo`, for the formats
    other than the one each produces (the identity case is generic) — the value-level
    analogue of `SparseArrays.sparse`.
  * **`KernelAbstractions.get_backend`** for the sparse types (usually
    `get_backend(nonzeros(A))`).
  * **`Adapt.adapt_structure`** converting each host struct to its device counterpart
    (`GPUArrays.GPUSparseDeviceVector`, `GPUSparseDeviceMatrixCSC`/`CSR`/`COO`), so the
    generic kernels can consume it inside `@kernel`s.
  * **`GPUArrays._sptranspose`/`_spadjoint`** — materialize a (conjugate) transpose; used
    by `kron`/`triu`/`tril` on lazily wrapped operands.

`SparseArrays`' accessors (`nnz`, `nonzeros`, `nonzeroinds`, `rowvals`, `getcolptr`) come
for free from the field names. Dense↔sparse conversion is generic and on-device:
`to_sparse(::Type{ST}, dense)` scans into a sparse array (`ST` a vector or COO type;
CSR/CSC follow via the verbs) and `to_dense(A)` scatters back to a dense array of the
back-end — so a back-end's `MyArray(::MySparse…)` and dense→sparse constructors can simply
call them.

### Functionality you get

Broadcasting; `mapreduce` and reductions (`sum`, `norm`, `opnorm`); sparse–dense and
sparse–vector multiplication (`*`, `mul!`, including transposed/adjoint operands);
`findnz`, `triu`/`tril`/`kron`/`reshape`/`droptol!`; `iszero`/`issymmetric`/`ishermitian`;
scalar and slice indexing; `copy`/`copyto!`/`collect`/`Array`; and conversion between
formats and to/from dense.

## Caching Allocator

```@docs
GPUArrays.@cached
GPUArrays.@uncached
```
