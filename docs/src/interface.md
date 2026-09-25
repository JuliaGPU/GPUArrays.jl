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

GPUArrays implements the sparse array formats itself, for every back-end. There are four,
all of them subtypes of `AbstractGPUSparseArray{Tv,Ti,N}` (which subtypes SparseArrays'
`AbstractSparseArray`):

| type | layout | fields |
|:--|:--|:--|
| `GPUSparseVector{Tv,Ti,Vi,Vv}` | like `SparseVector` | `nzInd`, `nzVal`, `len` |
| `GPUSparseMatrixCSC{Tv,Ti,Vi,Vv}` | compressed sparse columns, like `SparseMatrixCSC` | `colPtr`, `rowVal`, `nzVal`, `dims` |
| `GPUSparseMatrixCSR{Tv,Ti,Vi,Vv}` | compressed sparse rows | `rowPtr`, `colVal`, `nzVal`, `dims` |
| `GPUSparseMatrixCOO{Tv,Ti,Vi,Vv}` | coordinates, sorted in row-major order | `rowInd`, `colInd`, `nzVal`, `dims` |

`Tv` is the element type and `Ti` the index type. `Vi` and `Vv` are the types of the
dense vectors that store the indices and values, typically the back-end's GPU vectors
(`MtlVector{Int32,Metal.PrivateStorage}`, `CuVector{Float32}`, ...). They are part of the
type so that generic code can allocate results with `similar` on the buffers it already
has, and so that methods can dispatch on the storage. A matrix is constructed from its
buffers, with the dimensions last:

```julia
A = GPUSparseMatrixCSR(rowPtr, colVal, nzVal, (m, n))
```

The constructors only check what is known on the host (buffer lengths, and whether the
dimensions and number of stored entries fit the index type). Like SparseArrays, the
formats are *canonical*: the indices within a row (CSR), column (CSC) or vector are sorted
and unique, and COO entries are sorted by row and then column without duplicates.
Explicitly stored zeros are allowed. Every operation relies on these invariants and
preserves them; `GPUArrays.check_structure(A)` verifies them on the device.

The structs are immutable, and the same struct is used inside kernels: `adapt` converts
each buffer, so a kernel that receives a `GPUSparseMatrixCSR` can read `A.rowPtr`,
`A.colVal` and `A.nzVal` directly. Operations that change the structure of an existing
array (`copyto!` between sparse arrays, for example) resize its buffers, which is only
safe when no other object shares them; constructors, conversions and `copy` therefore
always return arrays with their own storage.

### What a back-end provides

Nothing sparse-specific is required. The generic implementation relies on the back-end's
dense support: its vector type, with `similar`, `copyto!` from and to the host, `resize!`,
a KernelAbstractions back-end, and implementations of `sortperm` (at least for `Int64`
keys), `accumulate!` and `findall` that run on the device.

A back-end will usually add the following, all optional:

1. **Aliases** for its storage, for naming and dispatch, with constructors on them:

   ```julia
   const MtlSparseMatrixCSR{Tv,Ti} = GPUSparseMatrixCSR{Tv,Ti,<:MtlVector{Ti},<:MtlVector{Tv}}
   MtlSparseMatrixCSR(A::AbstractArray) = GPUSparseMatrixCSR(adapt(MtlArray, A))
   ```

   Constructors must be defined on the alias itself (including the `{Tv}` and `{Tv,Ti}`
   forms): methods of the generic type do not apply to a constrained alias.
2. **A transfer policy** for its opinionated adaptor (`mtl`, `cu`), one method per host
   format on `adapt_structure`, e.g. producing CSR matrices with `Int32` indices and
   narrowing the values as the dense adaptor does. Without it, those adaptors treat a host
   sparse array like any other array.
3. **A dense constructor** that densifies on the device, e.g.
   `MtlArray{T,2}(A::MtlSparseMatrixCSR) = copyto!(similar(nonzeros(A), T, size(A)), A)`.
   Otherwise the generic dense constructor copies through the host.
4. **Vendor library overrides**, as methods of the public functions that are more specific
   in storage, element and index type than the generic ones. When a vendor routine does
   not apply, such a method can call the generic implementation, which never dispatches
   back to the public function:

   | operation | public function | generic implementation |
   |:--|:--|:--|
   | sparse × dense vector | `mul!(y, tA, A, x, α, β)` | `GPUArrays.generic_spmv!` |
   | sparse × dense, dense × sparse | `mul!(C, tA, tB, A, B, α, β)` | `GPUArrays.generic_spmm!` |
   | sparse × sparse | `*`, `mul!(C, tA, tB, A, B, α, β)` with a sparse `C` | `GPUArrays.generic_spgemm`, `GPUArrays.generic_spgemm!` |
   | CSR ↔ CSC | the target format's constructor | `GPUArrays.generic_regroup` |
   | CSR ↔ COO | the target format's constructor | `GPUArrays.generic_expand`, `GPUArrays.generic_compress` |
   | assembly from coordinates | `sparse(I, J, V, m, n, combine)`, `sparsevec` | `GPUArrays.generic_assemble` |

   The product methods are LinearAlgebra's storage-level `mul!`, which it calls on Julia
   1.13 and later with a character per operand ('N', 'T', 'C', or 'S'/'s'/'H'/'h' for the
   upper or lower triangle of a Symmetric or Hermitian matrix); on older versions GPUArrays
   forwards LinearAlgebra's internal `generic_matvecmul!` and `generic_matmatmul!` to
   them. A vendor method looks like:

   ```julia
   function LinearAlgebra.mul!(y::CuVector{T}, tA::AbstractChar, A::CuSparseMatrixCSR{T},
                               x::DenseCuVector{T}, α::Number, β::Number) where {T<:BlasFloat}
       tA in ('N', 'T', 'C') || return GPUArrays.generic_spmv!(y, tA, A, x, α, β)
       # call the vendor library
   end
   ```

   The generic implementations make no promise of bitwise reproducibility across versions
   or back-ends.
5. **Formats that GPUArrays does not implement** (block-sparse formats, batched matrices)
   as back-end structs subtyping `AbstractGPUSparseArray`, with constructors to and from
   the generic formats.

### Allocation and conversion

Generic code never names a back-end type; it allocates through the buffers it has:

| need | idiom |
|:--|:--|
| a dense array on the same device and storage | `similar(nonzeros(A), T, dims)` |
| a new index or value buffer | `similar(A.colVal, Ti, n)`, `similar(A.nzVal, Tv, n)` |
| an empty matrix of the same format | `similar(A, Tv, (m, n))` |

`similar` follows SparseArrays: `similar(A)`, `similar(A, Tv)` and `similar(A, Tv, Ti)`
copy the structure and leave the values uninitialized, `similar(A, Tv, dims)` returns an
array without stored entries (a matrix of the same format for two dimensions, a vector for
one, a dense array for more), and `zero(A)` is an empty array of the same format.

Conversions:

- between formats (on the device), with the target format's constructor: `GPUSparseMatrixCSC(A)`, and
  `GPUSparseMatrixCSC{Tv,Ti}(A)` to also change the element and index types;
  `convert(GPUSparseMatrixCSC, A)` returns `A` itself if it already has that format;
- to the host, with `SparseMatrixCSC(A)`, `SparseVector(x)` or `Array(A)`;
- from a dense GPU array, dropping its zeros, with `sparse(A; fmt=:csc)` or the format's
  constructor (`GPUSparseMatrixCSR(A)`); the index type defaults to `Int`;
- to a dense array on the device, with `copyto!(similar(nonzeros(A), T, size(A)), A)`;
- transposes: `transpose(A)` and `adjoint(A)` stay lazy, `copy(transpose(A))` and
  `permutedims(A)` materialize them in the same format, and a format constructor
  (`GPUSparseMatrixCSC(transpose(A))`) in another one;
- between host and device, with `adapt`. `adapt(MtlArray, S)` and `adapt(Array, A)` move
  the storage and keep the format, element and index type (a CSC matrix stays CSC, and
  comes back as a `SparseMatrixCSC`); adapting never densifies. `adapt(T, S)` with a
  sparse type `T` calls the constructor `T(S)`.

The generic types cannot be constructed from host arrays, because they do not determine
where to store the result: use a back-end alias, `adapt`, or the back-end's adaptor.

### Reference

```@docs
GPUArrays.AbstractGPUSparseArray
GPUArrays.GPUSparseMatrixCSR
GPUArrays.GPUSparseMatrixCSC
GPUArrays.GPUSparseMatrixCOO
GPUArrays.GPUSparseVector
GPUArrays.check_structure
GPUArrays.generic_regroup
GPUArrays.generic_expand
GPUArrays.generic_assemble
GPUArrays.generic_spmv!
GPUArrays.generic_spmm!
GPUArrays.generic_spgemm
GPUArrays.generic_spgemm!
```

## Caching Allocator

```@docs
GPUArrays.@cached
GPUArrays.@uncached
```
