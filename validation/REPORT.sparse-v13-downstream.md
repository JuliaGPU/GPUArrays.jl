# Downstream source audit

This is a scoped source audit, not a passing downstream package test suite. Existing
compatibility bounds deliberately exclude the unreleased stack; they were not loosened
to manufacture a passing installation.

| Source revision | Finding | Required migration |
| --- | --- | --- |
| [Lux 393dc2175f](https://github.com/LuxDL/Lux.jl/tree/393dc2175fa242a13c40673ba0cb6098b63db6a8/lib/MLDataDevices), MLDataDevices1.17.10 | GPUArraysSparseArraysExt dispatches on AbstractGPUSparseMatrixCSC / AbstractGPUSparseVector. GPUArrays compat11.3.2, older backend majors. | Deprecated CSC binding preserves dispatch; the CPU-transfer body is covered by the focused regression fixture. Review other extensions and release compatibility for the coordinated stack. No whole-package pass claimed. |
| [QuantumToolbox 6cf548f2f0](https://github.com/qutip/QuantumToolbox.jl/tree/6cf548f2f0931ca6de04a0bf29a81d7d58af5425/lib/QuantumToolboxCore), QuantumToolboxCore0.3.1 | GPUArraysExt imports dense_array_type for sparse-to-dense conversion. GPUArrays compat11.4.1/12.1; CUDACore6. | Allocate from nonzeros(A), then copyto! from A. Keep element-type conversion and initialize structural zeros. Release updated compatibility after backend validation. |
| [SparseMatrixColorings 3ec3aef981](https://github.com/JuliaDiff/SparseMatrixColorings.jl/tree/3ec3aef981e48520904655ef82dd7def66466467),0.4.29 | GPUArraysExt uses dense_array_type after CPU compression. GPUArrays compat11.5/12. | Allocate on A's buffer storage, then copy the compressed CPU result into that destination. Audit its CUDA extension and update compatibility. |

The v13 regression suite checks all three deprecated format bindings with partial
parameters, the MLDataDevices-style real/complex matrix/vector CPU precision policy,
and buffer-based sparse-to-dense allocation/copy with NaN-seeded destinations. These
tests validate the relevant idioms, not extension loading or complete consumer behavior.

The old format names are canonical aliases, not abstract extension points. Independent
formats must subtype AbstractGPUSparseArray directly. Separate device sparse types,
BSR abstract aliases, removed fields and old complete type-parameter layouts are not
restored. CUDA BSR/batched and AMD BSR remain actual backend representations and are
tested by their native suites, separately from this audit.
