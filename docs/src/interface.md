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
To be able to actually use the functionality that is defined for `AbstractGPUArray`s, you need to define a [KernelInterface](https://github.com/JuliaGPU/KernelAbstractions.jl/tree/main/lib/KernelInterface) back-end, which compiles and launches kernels, like so:

```julia
import KernelInterface as KI
struct CustomBackend <: KI.Backend end
KI.get_backend(a::CA) where CA <: CustomArray = CustomBackend()
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

## Vendor BLAS

GPUArrays implements LinearAlgebra's matrix products and triangular products and solves for
all GPU arrays. It checks dimensions, turns transposed, symmetric and triangular wrappers
into BLAS characters, and then calls one of the hooks below. Their default methods use
GPUArrays' kernels; a back-end with a vendor BLAS library adds methods for the operand types
and eltypes that library supports, and everything else keeps using the defaults:

```julia
GPUArrays.blas_gemm!(tA, tB, α::Union{Bool,T}, A::StridedCuVecOrMat{T}, B::StridedCuVecOrMat{T},
                     β::Union{Bool,T}, C::StridedCuVecOrMat{T}) where {T<:CublasFloat} =
    CUBLAS.gemm!(tA, tB, α, A, B, β, C)
```

The hooks are only called with operands that a BLAS library can address:

- Operands are GPU arrays or strided views of them. Matrix operands have a unit row stride
  and a column stride of at least their number of rows; vectors have a positive stride.
- Dimensions match, and no operand is empty.
- Operands have the same eltype, except for `blas_gemm!` and `blas_gemv!`.
- Scalars are `Bool`s or of the eltype of the output, as LinearAlgebra converts them where
  that is lossless; only `blas_gemm!` can also receive other scalars.
- Outputs don't share memory with inputs, except where a hook documents otherwise.

The hooks mirror the functions in `LinearAlgebra.BLAS`:

```@docs
GPUArrays.blas_gemm!
GPUArrays.blas_symm!
GPUArrays.blas_syrk!
GPUArrays.blas_trmm!
GPUArrays.blas_trsm!
GPUArrays.blas_gemv!
GPUArrays.blas_symv!
GPUArrays.blas_trmv!
```

## Caching Allocator

```@docs
GPUArrays.@cached
GPUArrays.@uncached
```
