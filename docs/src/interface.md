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

There are numerous examples of potential interfaces for GPUArrays, such as with [JLArrays](https://github.com/JuliaGPU/GPUArrays.jl/blob/main/lib/JLArrays/src/JLArrays.jl), [CuArrays](https://github.com/JuliaGPU/CUDA.jl/blob/main/CUDACore/src/array.jl), and [ROCArrays](https://github.com/JuliaGPU/AMDGPU.jl/blob/main/src/array.jl).

## Alias detection

Without the hook below, all views of one buffer are treated as aliasing each other, so Base
and GPUArrays copy one of them before operations that write to the other (e.g. `copyto!`,
broadcasting, `accumulate!`). Report where an array's elements are stored to make that check
precise:

```@docs
GPUArrays.memory_location
```

## Sparse arrays

GPUArrays implements the sparse array formats itself, for every back-end. There are four,
all of them subtypes of `AbstractGPUSparseArray{Tv,Ti,N}` (which subtypes SparseArrays'
`AbstractSparseArray`):

| type | layout | fields |
|:--|:--|:--|
| `GPUSparseVector{Tv,Ti,Vi,Vv}` | like `SparseVector` | `nzInd`, `nzVal`, `len` |
| `GPUSparseMatrixCSC{Tv,Ti,Vp,Vi,Vv}` | compressed sparse columns, like `SparseMatrixCSC` | `colPtr`, `rowVal`, `nzVal`, `dims` |
| `GPUSparseMatrixCSR{Tv,Ti,Vp,Vi,Vv}` | compressed sparse rows | `rowPtr`, `colVal`, `nzVal`, `dims` |
| `GPUSparseMatrixCOO{Tv,Ti,Vi,Vv}` | coordinates, sorted in row-major order | `rowInd`, `colInd`, `nzVal`, `dims` |

`Tv` is the value type and `Ti` the coordinate type (`indtype(A)`). Compressed pointers
have their own storage `Vp` and integer type `eltype(compressed_ptr(A))`. `Vi` and `Vv` are the types of the
dense vectors that store the indices and values, typically the back-end's GPU vectors
(`MtlVector{Int32,Metal.PrivateStorage}`, `CuVector{Float32}`, ...). They are part of the
type so that generic code can allocate results with `similar` on the buffers it already
has, and so that methods can dispatch on the storage. A matrix is constructed from its
buffers, with the dimensions last:

```julia
A = GPUSparseMatrixCSR(rowPtr, colVal, nzVal, (m, n))
```

The constructors only check host metadata: buffer lengths, dimensions fitting the
coordinate type, and compressed `nnz+1` fitting the pointer type. COO/vector entry
counts are not limited by their coordinate type. Like SparseArrays, the
formats are *canonical*: the indices within a row (CSR), column (CSC) or vector are sorted
and unique, and COO entries are sorted by row and then column without duplicates.
Explicitly stored zeros are allowed. Every operation relies on these invariants and
preserves them; `GPUArrays.check_structure(A)` verifies them on the device.

The structs are immutable, and the same struct is used inside kernels: `adapt` converts
each buffer, so a kernel that receives a `GPUSparseMatrixCSR` can read `A.rowPtr`,
`A.colVal` and `A.nzVal` directly. Operations that change the structure of an existing
array (`copyto!` between sparse arrays, for example) resize its buffers, which is only
safe with exclusive structural access and storage supporting `GPUArrays.supports_resize`.
Raw buffer constructors borrow their inputs. Allocating format constructors and `copy`
return independent buffers; identity `convert` and `adapt` may share. Updating
`nonzeros(A)` preserves the pattern. Structural updates invalidate retained structural
views, device representations and prepared analyses. Known unsupported storage, shapes
and representability errors are rejected before any destination write. Allocation
failure does not promise transactional recovery.

### What a back-end provides

Nothing sparse-specific is required. The generic implementation relies on the back-end's
dense support: its vector type, with `similar`, `copyto!` from and to the host, `resize!`,
a KernelAbstractions back-end, and implementations of `sortperm` (at least for `Int64`
keys), `accumulate!` and `findall` that run on the device.

A back-end will usually add the following, all optional:

1. **Aliases** for its storage, for naming and dispatch, with constructors on them:

   ```julia
   const MtlSparseMatrixCSR{Tv,Ti} = GPUSparseMatrixCSR{Tv,Ti,<:MtlVector{<:Integer},<:MtlVector{Ti},<:MtlVector{Tv}}
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
4. **Vendor bindings and resource primitives** remain backend responsibilities.
   GPUArrays extensions own public-operation validation and selection. They implement
   `vendor_spmv`/`vendor_spmm` capability selection and the corresponding selected
   executor, rather than overriding public `mul!` methods. `GPUArrays.spmv!` and
   `GPUArrays.spmm!` accept `Automatic()`, `Portable()`, and `Vendor()` policies.
   Forced portable execution includes subordinate sparse conversions. Forced vendor
   execution rejects unsupported combinations before output writes; execution failures
   propagate and never trigger a fallback retry.

   The callable `generic_spmv!`, `generic_spmm!`, `generic_regroup` and
   `generic_assemble` implementations remain available. GPUArrays retains the
   Julia <1.13 forwarding methods for LinearAlgebra internals.
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
one, a dense array for more), and `zero(A)` is an empty array of the same format. Copying, value-only type changes
and compressed-to-compressed conversion preserve both integer widths. Explicit
`{Tv,Ti}` constructors and `similar(A,Tv,Ti)` request uniform `Ti`, with checked
narrowing. Binary operations promote coordinate and pointer widths independently.
Without a compressed-pointer prototype, new compressed pointers use
`promote_type(Int,Ti)`. COO and vectors have no terminal-pointer count restriction.
Adapting split-width CSC to host SparseArrays promotes both index buffers to a common
type; adapting back cannot recover the original split. Storage aliases such as
`CuSparseMatrixCSR{T,Ti}` are UnionAll types: downstream structs should parameterize
on `typeof(A)` rather than assume that alias is concrete.

Conversions:

- between formats (on the device), with the target format's constructor: `GPUSparseMatrixCSC(A)`, and
  `GPUSparseMatrixCSC{Tv,Ti}(A)` to also change the element and index types;
  `convert(GPUSparseMatrixCSC, A)` returns `A` itself if it already has that format;
- to the host, with `SparseMatrixCSC(A)`, `SparseVector(x)` or `Array(A)`;
- from a dense GPU array, dropping its zeros, with `sparse(A; fmt=:csc)` or the format's
  constructor (`GPUSparseMatrixCSR(A)`); canonical constructors default to `Int`.
  Dense `sparse(A)` preserves CUDA/AMD's existing uniform `Int32` default and BSR option;
  this convenience policy is maintained by GPUArrays' extensions, not backend overrides;
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

### Supported surface and limits

The canonical formats support construction, transfer, conversions, products, reductions,
range slicing, stored-value updates, value equality and sparse in-place broadcast.
Stored zeros are permitted. Non-zero-preserving allocating broadcast and dimensional
reductions may return dense device arrays. Dense/non-zero-preserving broadcast into a
sparse destination and nonzero fill throw before mutation. Individual insertion,
concatenation and blockdiag are not implemented; reshape and linearization copy.
Arbitrary sparse views and complete SparseArrays compatibility are not promised.
Diagonal extraction follows the dense-array convention of returning an empty device
vector for offsets outside the matrix, including extreme signed offsets.
`A ± λI` and `λI ± A` remain supported for rectangular GPU sparse matrices: the
identity contributes through `min(size(A)...)`. This preserves existing CUDA/AMD
functionality independently of Julia version; newer SparseArrays restricts these
operations to square matrices. Square results follow SparseArrays.

For `*(α, A, B)` with a canonical sparse matrix (or its transpose/adjoint) on
the left, GPUArrays scales `A` with normal element-type promotion before the
matrix product, including dense GPU right-hand sides. This is equivalent to
`(α*A)*B`; Julia's fused CPU method may instead accumulate in the input type
before scaling. The evaluation order can change rounding, overflow and underflow.

Assembly preserves duplicate operand order with unspecified parenthesization. Custom
combiners must be associative; commutativity is not required. Floating-point rounding
can vary across algorithms, backends and releases. Stable sorting uses checked packed
keys with a lexicographic fallback. Generic ESC SpGEMM checks candidate counts and has
a default 1 GiB temporary-memory estimate limit, overridable through
`generic_spgemm(A,B; max_temp_bytes=...)`; its memory follows candidate products.

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
