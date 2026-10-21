# GPUArrays sparse arrays: v13 design review

**Recommendation: keep PR #745's GPUArrays-owned formats, but do not release its current interface unchanged.** The storage-parametric design removes the right backend dependencies. Before switching backends, separate compressed-pointer storage from coordinate storage, make GPUArrays own vendor selection, and finish the ownership, mutation, device-access and Julia-array contracts. These are foundations; implementing every sparse operation or a universal planning framework is not a prerequisite.

The implementation sequence and release gates are in [PLAN.sparse-v13.md](PLAN.sparse-v13.md). This report records the evidence behind that plan. No production code or remote branches were changed during this investigation.

## Scope and revisions

Investigated on 2026-10-21:

- [GPUArrays #745](https://github.com/JuliaGPU/GPUArrays.jl/pull/745), head `2a50d28cbce47cf1a7fb471ca3d9a33f20236917`, merge base `47056426de4a165e95c400ef7f4a96ed6da5b567` on `tb/permutedims-empty`.
- GPUArrays main `6fa6490`, version 12.1.1. Main and the PR have 53 and 13 commits exclusive to their respective histories. The PR's intended 12.0 release has already been overtaken by the AK release; this work must target 13.0.
- The current PR supersedes its original backend-hook design. The seven inline review comments concern that older version; the rewrite addresses their substantive points. The requested CUDA #3189 `findnz` regression test is present.
- The existing `GPUArrays-sparse/{PLAN,HANDOFF}.md` and Metal sparse notes were read as historical context. Their September test counts, old hardware assumptions and earlier permissions are not evidence or authorization for this session.

This is a design review and implementation handoff, not a completed migration. CUDA/cuSPARSE, AMDGPU/rocSPARSE and oneAPI vendor integrations have not been migrated or validated here.

## What is sound

Owning vector, CSR, CSC and COO containers in GPUArrays is the right direction. Backend-specific structs and format/allocation hooks are unnecessary when component vectors already determine allocation, device adaptation and element storage. The PR's `similar`-based allocation lets a new portable operation work on every participating dense backend. Direct adaptation of immutable containers into kernel arguments works; the Metal representation is isbits.

Keep the canonical representation: one-based, sorted, unique coordinates, with explicit stored zeros permitted. COO is row-major. Unsorted or duplicate coordinates belong at the assembly boundary. This keeps ordinary operations simple and provides suitable input to vendor routines. A future uncoalesced assembly buffer can be a separate type; it need not weaken existing invariants.

Keep one open `AbstractGPUSparseArray` root and closed unions for algorithms that specifically understand these four layouts. An arbitrary BSR subtype cannot inherit a CSR traversal algorithm merely by subtyping the root. Document conversions and basic accessors as the extension route. Portable BSR, batched arrays, alternative formats and additional operations can subsequently be added centrally.

Keep vendor descriptors and analysis state out of the array fields. Keep `nnz(A) == length(nonzeros(A))`: backing allocation capacity is distinct from logical stored length. Preserve structural `adapt` rather than having it silently densify or change CSR into CSC.

## What must change in the foundation

### Separate pointer and coordinate storage before aliases spread

The proposed CSR/CSC representation uses the same `Vi` for both compressed pointers and coordinates. That couples the representation of `nnz+1` to the representation of row/column indices and also prevents different vector wrappers for the two buffers.

Use `GPUSparseMatrixCSR/CSC{Tv,Ti,Vp,Vi,Vv}`, with `Ti` describing coordinates and `eltype(ptr)` describing offsets. Keep vector and COO at four parameters. Existing compressed inputs preserve their widths; new compressed outputs without a pointer prototype default to `promote_type(Int,Ti)`. This permits Int64 pointers with Int32 coordinates, without adding a pattern object, capacity field or algorithm state. cuSPARSE already distinguishes the two index types in its CSR descriptor, although support must still be checked per operation. [CSR descriptor API](https://docs.nvidia.com/cuda/cusparse/index.html#cusparsecreatecsr).

This is a deliberate v13 representation change. The PR advertises full type parameters and storage aliases as extension interfaces; postponing it would make a useful, existing representation requirement expensive to add. Exact width preservation and explicit conversion semantics are settled in the plan.

### Keep selection and vendor integration under GPUArrays maintenance

The PR recommends more-specific backend methods on public `mul!`, format constructors and assembly. Those methods would intercept future GPUArrays selection and semantic fixes. An `alg` keyword alone cannot fix this: Julia chooses the positional method before processing keywords.

GPUArrays must own validation, wrapper normalization, selection and dispatch into portable or vendor implementations. Put vendor integration in GPUArrays package extensions, analogous to AK's backend extensions. Backends retain dense storage/runtime facilities, vendor bindings and resource-management primitives. Their sparse aliases must not create a competing selection layer.

This requires coordinated migration, not simply moving files. CUDA has CUDACore/cuSPARSE/cuSOLVER packages; cuSPARSE's current Julia wrappers still assume its old sparse types. AMDGPU and oneAPI package their vendor APIs differently. Each needs a narrow executable integration seam. New vendor API bindings may still require backend changes; new portable algorithms and selection among existing bindings should not.

### Specify borrowing and mutation honestly

The documentation says constructors always return independent storage. Raw constructors actually borrow their buffers, as SparseArrays does. `convert` may return its argument, and `adapt` may preserve buffers. These are useful semantics, but require an operation-by-operation ownership contract.

Retain immutable structs. Structural mutation is conditional on supported resizable dense buffers (GPU array wrappers represented on the host, or ordinary CPU vectors) and caller-guaranteed exclusive structural access. Raw views remain useful for reads and value updates; unsupported structural updates must be rejected before any destination write. Check every affected buffer first and avoid `resize!` when its length is unchanged. Do not infer ownership from `AbstractGPUVector` membership or from the existence of a `resize!` method.

Add `dataids`/alias handling based on component buffers. Do not add a generic sparse `unsafe_free!` that blindly frees borrowed buffers. No public structural-generation counter can reliably detect user kernels editing exposed index arrays. Future reusable analysis must snapshot its pattern or use an explicit borrowing/invalidation contract.

### oneAPI makes resource lifetime an immediate requirement

oneAPI's sparse structs currently contain a lazy matrix handle, register finalizers, and invalidate it on storage replacement. `sparse_optimize_gemv!` also prepares reusable state. Aliasing those types to GPUArrays containers removes this machinery.

Migrate ordinary operations to a resource owner with an explicit operation lifetime. Preserve optimized repeated execution through an explicit prepared vendor operation. Keep originating device/context/queue and buffers alive through asynchronous use and safe destruction. Existing oneAPI release code defers queue work out of finalizers; replacing its type must preserve that safety property. A global cache keyed only by pointers, object identity or `nnz` is not correct for mutable exposed indices.

This requires a working oneAPI lifetime design now, not a universal public sparse-plan API now.

## Confirmed findings

Paths below refer to PR head `2a50d28`; line numbers are useful until rebasing. Full repro scripts and logs are under `/Users/tim/Julia/tmp/gpuarrays745-review/`.

| Priority | Finding and trigger | Evidence / required correction |
|---|---|---|
| P1 | Structural mutation corrupts accepted view-backed input before throwing. | `types.jl:87`, `structure.jl:25–56`: CSC with dense pointer/index buffers and `view(values,1:2:4)` values; `dropzeros!` throws `MethodError(resize!, SubArray)` after pointers become `[1,2,2]` and index/value lengths become 1/2. Reproduced on JLArrays and Metal. Preflight the complete destination before writing. |
| P1 | Symmetric/Hermitian filtering multiplies excluded terms by zero. | `linalg.jl:60–67,76–124`: `Symmetric(sparse([1. 2.;4. 3.]),:U)*[Inf,1.]` gives `[NaN,NaN]`, versus `[Inf,Inf]` on SparseArrays. Reproduced on JLArrays and Metal. Skip excluded terms instead of forming `0*Inf`. |
| P1 | Compression narrows the end-search sentinel into the coordinate type. | `primitives.jl:24–26`: `compress(JLArray(Int8[127]),127)` ends in 1 rather than 2. The constructor accepts dimension 127. Use wide search arithmetic; narrow only representable stored pointers. |
| P1 | Sparse-vector broadcast loses a real maximum-valued index. | `broadcast.jl:311–330,587–590`: vector length 127 with `Int8[127]`, value 2; `x .+ x` has zero stored entries instead of value 4. `typemax(Ti)` is both a legal coordinate and the absent-entry sentinel. Use a separate flag or an impossible position. |
| P2 | The new public sparse indexing method cannot run inside kernels. | `indexing.jl:14,22,28,37`: unconditional host `assertscalar` produces Metal InvalidIR. Existing kernel tests access fields only. Removing the outer check and relying on component-buffer host guards was prototyped successfully. This is a missing device capability, not an established regression from the old bare-bones device types. |
| P2 | Alias detection misses shared structure. | `types.jl`: two CSC containers sharing pointers/indices but different values give `Base.mightalias(A,B) == false`. Expose component `dataids` and cover overlapping inputs and outputs. |
| P2 | Claimed no-host-fallback contract is not enforced. | `hcat(A,A)` on JLArrays-backed CSC returns a host `SparseMatrixCSC`. Explicit download constructors are reached through SparseArrays fallbacks. Implement promised operations centrally; explicitly reject other known host fallbacks until supported. |
| P2 | A recorded nightly ambiguity remains. | `conversions.jl:284` `copyto!(::Array, ::GPUSparseArray)` conflicts with the newer SparseArrays vector method. CI log on Julia 1.14.0-DEV.3366 shows `sparse/ambiguities` failing. Add the appropriate vector specialization and rerun on current nightly. |

The narrow-index examples are small tests of general arithmetic errors. Int32 has the same sentinel problems at its boundary. Separately, broadcast counts and prefix sums use `Ti` before checking the resulting size (`broadcast.jl:343–352,583–594`); the implementation must check growth before narrowing, including allocation sizes and products. This latter finding is source-traced, not a large-allocation experiment.

Several important gaps are inherited, rather than introduced by #745:

- Sparse `norm` uses naive powers: `norm` of one stored Float32 `1f30` returns `Inf`, and `1f-30` returns zero. SparseArrays returns the finite magnitude. Delegate to the robust dense-buffer norm where its semantics agree. Main already has scaled dense norm code.
- `==`, in-place sparse broadcast, and common wrapper operations reach scalar fallbacks. Direct probes of `A == copy(A)`, `A .*= 2`, `sum(transpose(A))` fail with scalar indexing disabled; `nnz(transpose(A))` has no method. These are production surface gaps, not all PR regressions.
- `reshape` copies, unlike Base's shared-storage reshape. The old GPU implementation also copied. Document this existing limitation; do not introduce fragile shared structural storage to mimic CPU aliasing.
- Matrix linear slicing `A[:]` exists in CUDA and must not disappear during migration. Vendor solves, preconditioners, BSR and batched formats need a similar parity audit.

## Julia and external alignment

SparseArrays is the behavioral reference for stored zeros, duplicate combination, accessors, copying, `similar`, wrapper mathematics and index validation. It is not a safe source of inherited GPU implementations: many methods on `AbstractSparseVector` or `AbstractSparseMatrix` scalar-loop or download via constructors. GPU CSC cannot inherit both the current GPU root and `AbstractSparseMatrixCSC`; retain the GPU root and implement the required accessor/behavior contracts explicitly. [SparseArrays documentation](https://docs.julialang.org/en/v1/stdlib/SparseArrays/).

| Reference | Relevant lesson for this design |
|---|---|
| [SciPy CSR](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html), [CuPy sparse](https://docs.cupy.dev/en/stable/reference/scipy_sparse.html) | Familiar compressed storage and normalization are useful references. Their mutable canonicality flags are unnecessary if GPUArrays requires canonical arrays and confines duplicate handling to assembly. |
| [PyTorch sparse](https://docs.pytorch.org/docs/2.14/sparse.html) | Distinguish uncoalesced COO from canonical compressed storage. GPUArrays can deliberately offer a stricter COO contract without importing tensor batching or autograd conventions into its core matrix types. |
| [JAX sparse / BCOO](https://docs.jax.dev/en/latest/jax.experimental.sparse.html) | Static stored capacity supports tracing and transformations. It is not a reason to burden Julia's ordinary dynamic sparse containers with padding/sentinel semantics. A future capacity-aware type can be separate. |
| [cuSPARSE SpMV](https://docs.nvidia.com/cuda/cusparse/index.html#cusparsespmv) | Preprocessing state and workspace have lifetimes distinct from matrix storage. Pattern reuse permits value changes under documented conditions. Independent prepared operations should own their descriptors. |
| [rocSPARSE generic operations](https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/reference/generic.html) | Buffer sizing, preprocessing and execution are distinct stages; capabilities and determinism vary by algorithm. Do not equate one sparse format with one universally valid vendor path. |
| [Kokkos symbolic SpGEMM](https://kokkos.org/kokkos-kernels/docs/API/sparse/spgemm_symbolic.html) and [numeric SpGEMM](https://kokkos.org/kokkos-kernels/docs/API/sparse/spgemm_numeric.html) | Separate reusable structural analysis from numeric work. Keep this possible without putting vendor handles into the array representation. |
| [oneMKL matmat](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/oneapi-mkl-sparse-matmat.html) | Its staged API reinforces the same separation. Source inspection of oneAPI.jl, rather than this page alone, establishes the concrete migration requirement. |

The PR's strict left-fold duplicate-combination promise is stronger than SparseArrays' documented contract and would unnecessarily constrain future parallel assembly. Before v13, specify preserved relative operand order with unspecified parenthesization: custom combiners must be associative, not necessarily commutative; floating-point rounding may vary. This allows ordered segmented reductions while retaining meaningful noncommutative combiners. It explicitly excludes nonassociative custom combiners, and requires cancellation-aware numerical tests.

The independent research tracks disagreed about adding public plans and algorithm knobs immediately. I agree with Astra's final assessment: establish central selection and implement the concrete oneAPI resource owner now; defer a universal plan API until a portable repeated-use implementation and a vendor implementation establish its contract. A scratch workspace is not a symbolic-analysis cache.

## Performance foundations

Rebasing onto main materially changes the prerequisite story. GPUArrays 12 already implements dense sort, scan, compaction and reductions through AK. The PR's JLArrays host shims and claim that OpenCL needs its own implementations are obsolete. AK's stable `sortperm` can remove the second sort used to restore input order. Its current radix implementation does not implement permutation/key-value sorting, so merely requesting radix is not a solution.

The PR's algorithms are valid starting points, but several costs must remain replaceable:

- CSR↔CSC uses expansion, packed-key sorting and gathers. CSC SpMV and transposed CSR SpMV repeat this conversion. A reusable permutation must gather current values, not cache stale values.
- SpMV assigns one thread per row; SpMM assigns one thread per output entry. Skewed rows and dense-access layout limit performance. Workgroup/subgroup algorithms can replace these centrally through KA.
- SpGEMM materializes every candidate scalar product before sorting/deduplication. Memory follows candidate count, not final `nnz`. Checked counts and bounded-memory handling are release concerns; an efficient symbolic/numeric algorithm is the subsequent optimization.
- Sparse-vector broadcast currently allocates/work-scans by logical length rather than stored length. A merged structural union belongs in GPUArrays; reusable merge/segmented primitives belong in AK.
- Packed keys impose `m*n < typemax(Int64)` even for extremely sparse matrices. Retain a fast packed path with a checked lexicographic fallback; stable sorting removes the second-sort size limit, not this limit.

Measured on Apple M3 Pro, Julia 1.13.2, current-main dense code plus unchanged PR sparse sources, AK 0.5.8 / KA 0.10.1: a random 4096×4096 Float32 matrix with 65,777 stored entries took 0.350 ms for CSR SpMV, 1.721 ms for CSC SpMV, 1.563 ms for transposed CSR SpMV, and 1.512 ms for CSR→CSC. These are minima of seven synchronized warm calls after three warmups. They demonstrate conversion cost, not general performance or a controlled comparison with vendor libraries.

## Validation and limits

| Configuration | Result |
|---|---|
| Exact PR, JLArrays, Julia 1.13.2 / KA 0.9.45 | 1,724/1,724 existing focused Float32 assertions passed. Additional adversarial probes found the issues above. |
| Exact PR + Metal #982 snapshot `ce1ba502f`, M3 Pro | Basic generic SpMV and isbits adaptation work; nonfinite masking, device indexing and view-mutation failures reproduced. |
| Scratch current-main dense code + PR sparse sources, current local Metal snapshot `bf1f54660`, KA 0.10.1 | 31/31 portable checks passed across CSR/CSC/COO: construction, SpMV/SpMM/SpGEMM, transpose product, broadcast, reduction, filtering, duplicate assembly and structure checks. |
| Same scratch source on pegasus, CUDACore 6.4.2 / KA 0.10.1 | 31/31 portable checks passed using CuArray, without cuSPARSE. This proves generic portability, not CUDA.jl's vendor migration. |
| Same scratch source on pegasus, OpenCL 0.10.12 / PoCL 7.2.2 | 31/31 portable checks passed on the Ryzen 9950X CPU device. This is OpenCL CPU portability evidence, not a GPU result. |
| Same scratch source, OpenCL NVIDIA driver / RTX 5080 | Failed: 2 checks passed, 6 errored before completion. SpMM download reported `CL_INVALID_VALUE`; SpGEMM and broadcast kernels failed to compile. This path is not validated. |
| Scratch device-indexing fix on modern Metal | 15/15 checks passed: device lookup, host-backed arithmetic, permitted shared-memory reads and rejected PrivateStorage host scalar reads. Only the outer sparse `assertscalar` checks were removed. |

The original JLArrays environment pins KA 0.9 while current Metal requires KA 0.10, so those runs used separate environments. The scratch overlay is **not** a completed rebase or a v13 implementation. Remote backend source checkouts were only read; environments and modified GPUArrays sources are under `~/Julia/tmp/ga745-review`.

Buildkite build [1958](https://buildkite.com/julialang/gpuarrays-dot-jl/builds/1958) is green because all 16 downstream backend jobs soft-failed at installation with exit 3. No backend test suite passed there. Nightly Ubuntu's ambiguity failure is also hidden by allowed failure; macOS nightly additionally timed out. These checks are not release evidence.

OpenCL NVIDIA failures are unresolved portability evidence, not attributed PR bugs: the failing session also stopped handling simple dense operations, while a fresh daemon passed dense round-trip, sortperm and accumulate probes. Reproduce the sparse failures from a fresh context and minimize each failure against main before choosing GPUArrays, AK, OpenCL or driver fixes. Detailed logs are `~/Julia/tmp/ga745-review/opencl-nvidia-tests.log` and `opencl-baseline*.log` on pegasus. PoCL required loading `pocl_jll` before the first platform enumeration in a fresh daemon with system ICDs excluded.

No complete CUDA/cuSPARSE, AMDGPU, oneAPI, cross-context lifetime or full-version test matrix was run. The user's available hosts are pegasus for CUDA/OpenCL/AMDGPU and taurus for oneAPI/CUDA/OpenCL; both SSH connections were verified. The full matrix belongs to the implementation/release gates.

## Design discussion and evidence locations

Peer discussions are available with `peer log ga745-astra`, `ga745-history`, `ga745-julia`, `ga745-primitives` and `ga745-frameworks`. The initial Opus launches exited before producing research; retries completed. Findings used here were checked against sources, logs or direct probes. Astra changed its initial preference for public vendor overrides after examining the central-maintenance requirement, and recommended separate pointer storage before v13. Both conclusions are adopted.

Local scratch root: `/Users/tim/Julia/tmp/gpuarrays745-review`. Useful files: `probes.jl`, `more-probes.jl`, `boundary-probes.jl`, `semantic-smoke.jl`, `focused-tests.jl`, `metal-probes.jl`, `portable-probes.jl`, `kernel-fixed.jl`, `bench.jl`, their logs, `evidence/bk1958.json`, and the nightly logs. The large `metal-probes.log` contains the full InvalidIR diagnostic; search it rather than printing it all. The plan contains self-contained reproduction snippets so the next session is not dependent on these temporary files. Sol's handoff review (`peer log ga745-plan-review`) identified missing mixed-width round-trip and value-only-constructor rules; these are now explicit.
