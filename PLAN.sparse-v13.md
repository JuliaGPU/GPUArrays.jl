# Productionize generic GPU sparse arrays for GPUArrays 13

This is the implementation handoff for [GPUArrays #745](https://github.com/JuliaGPU/GPUArrays.jl/pull/745). Read [REPORT.sparse-v13.md](REPORT.sparse-v13.md) for the design assessment, confirmed failures, source references and measured limits. Implementation is underway on `tb/sparse-v13`; see [HANDOFF.sparse-v13.md](HANDOFF.sparse-v13.md) for exact commits, reviews, tests and remaining work. Representation, mutation preflight, arithmetic corrections and the initial selector are implemented. The selector is not frozen: vendor migrations and release gates remain open.

The objective is one coordinated backend migration, followed by portable sparse development and algorithm selection maintained in GPUArrays and its AK/KA dependencies. Keep backend-specific runtime and vendor bindings where they belong. New bindings or new hardware capabilities may require backend work; adding ordinary sparse operations or changing the preferred existing algorithm must not.

## Decisions to implement

### 1. Keep GPUArrays-owned immutable canonical containers

Keep sparse vector, CSR, CSC and row-major COO, under `AbstractGPUSparseArray <: AbstractSparseArray`. Sorted, unique, one-based coordinates are invariants; stored zeros are valid. Unassembled inputs and unfinished vendor outputs must remain private buffers/descriptors until canonicalization, including internal intermediates and results of qualified vendor APIs. This replaces CUDA's former noncanonical raw-storage contract; raw borrowing constructors must not silently sort caller buffers. Raw constructors check metadata; `check_structure` validates contents explicitly. Do not launch a full structural validation on every operation.

CSR/CSC become:

```julia
struct GPUSparseMatrixCSR{Tv,Ti<:Integer,
                          Vp<:AbstractVector{<:Integer},
                          Vi<:AbstractVector{Ti},
                          Vv<:AbstractVector{Tv}} <: AbstractGPUSparseMatrix{Tv,Ti}
    rowPtr::Vp
    colVal::Vi
    nzVal::Vv
    dims::Dims{2}
end
```

CSC has the analogous fields. Vector and COO keep `{Tv,Ti,Vi,Vv}`. This is an interface sketch; retain checked inner constructors. No descriptor, generation, algorithm, padded capacity or mutable pattern object belongs in these structs.

`indtype(A)` is the coordinate type; pointer arithmetic uses `eltype(compressed_ptr(A))`. Maintain these conversion rules:

- Raw compressed constructors infer both integer types from their buffers.
- `similar(A)`, `similar(A,Tv)`, value-type-only constructors such as `GPUSparseMatrixCSR{Tv}(A)`, `copy`, and compressed-to-compressed conversion preserve both widths, where representable. Do not implement `{Tv}` by forwarding to the explicitly uniform `{Tv,Ti}` constructor.
- Explicit `{Tv,Ti}` construction and `similar(A,Tv,Ti)` request uniform `Ti` for coordinates and pointers, with checked narrowing.
- Binary operations promote coordinate types and pointer types independently. Expansion to COO preserves coordinates but necessarily loses pointer-width metadata. With no compressed pointer prototype, a new compressed output uses `promote_type(Int,Ti)` for pointers; this is type-stable and accommodates host-addressable stored counts. Explicit `{Tv,Ti}` still requests uniform widths. Internal algorithms must carry their chosen output pointer type separately through COO intermediates. COO has no compressed-pointer representability limit: its stored count may exceed `typemax(Ti)` when both coordinates fit. Vector constructors likewise must allow `nnz == typemax(Ti)` because they have no terminal pointer. Test such cases with narrow coordinates and wide pointers.
- Keep the conservative requirement that both dimensions fit the coordinate type in v13. This makes transpose and format conversion predictable. A later relaxation for huge implicit major dimensions is possible.
- Output counts, `nnz+1`, size products and allocation byte sizes are checked before narrowing or allocation. Using Int64 without checking overflow is insufficient. For v13, retain the selected output widths and throw a clear representability error rather than silently wrapping; callers can explicitly widen beforehand.
- Pointer-buffer length is a host dimension, not a value that must fit its element type. Compression's logical end-search key must not wrap into a narrow coordinate type.

Update full-parameter aliases, constructor docs and dispatch examples together. Backends should constrain all buffer families while allowing their applicable memory/storage modes. Warn in migration notes that partially specified storage aliases are UnionAll types: downstream structs should parameterize on `typeof(A)` rather than assuming `CuSparseMatrixCSR{T,Ti}` is concrete.

### 2. Define storage semantics without pretending to enforce ownership

| Operation | Contract |
|---|---|
| Raw buffer constructor | Borrows supplied buffers. Caller supplies canonical compatible storage. |
| `copy`, `similar` with copied structure, allocating format constructors | Independent structure/value buffers as documented. |
| Identity `convert` | May return the input. |
| `adapt` | Preserves structure/format, adapts buffers; identity adaptation may share. GPU CSC/vector adapted to ordinary host vectors use SparseArrays types. For mixed-width CSC, SparseArrays requires a common integer type, so promote pointer and coordinate types, possibly allocating an index buffer. Adapting back preserves that unified type; the original width split is not recoverable. CSR/COO may remain generic format containers with host buffers. |
| `nonzeros(A)` updates | Values-only mutation; pattern remains valid. |
| Structural `copyto!`, filtering, sparse-output products/broadcast | Require supported resizable dense buffers and exclusive structural access; invalidate retained structural views/device representations/analysis. |

Support raw views for reads and values-only operations. Before structural mutation writes anything, validate dimensions, representability and **all** destination storage restrictions. Reject unsupported views centrally; skip resizing buffers already at the required length. For GPU storage these are dense GPU array wrappers represented on the host; ordinary CPU vectors are also resizable. A dense-buffer capability distinction, if needed, belongs in the general dense interface, not a new sparse hook that every backend implements.

Caller-guaranteed exclusivity remains necessary: buffers may be externally wrapped or shared even when their types look dense. Do not promise strong recovery from allocation failure. Do promise that known unsupported storage, invalid shapes and overflow do not leave a partially modified matrix. Tests must compare all buffers after rejected operations.

Define component-based `Base.dataids` and appropriate unalias copying. Audit in-place broadcast and overlapping `copyto!` against main's alias handling. Do not indiscriminately implement sparse `unsafe_free!`; preserve existing backend deallocation APIs only with an explicit ownership/alias contract. Internal temporaries can be released when exclusively owned and safe with asynchronous execution.

Remove the unconditional sparse host `assertscalar` from lookup. Host dense buffers already enforce their scalar-access policy; device buffers can then implement the lookup in kernels. Shared/host-accessible buffers may intentionally permit scalar access. Test implicit and stored zeros, bounds, direct kernel indexing, CSC `nzrange`, and host scalar rejection. Host-backed CSR/COO should remain usable through ordinary CPU array fallbacks; do not silently change format during `adapt(Array, A)` to avoid implementing that behavior.

Check compatible storage/device placement at host construction boundaries where it can be determined safely. Do not call host runtime queries when reconstructing device-buffer structs during adaptation or inside kernels. Same backend type alone is not proof of the same physical device/context. Add mismatched-buffer tests and use the stack's existing dense/runtime checks rather than introducing a sparse-specific device registry.

### 3. Make GPUArrays own the operation boundary

Use this call path:

```text
Julia public operation: mul!, *, constructor, sparse
  -> GPUArrays validation and wrapper normalization
  -> GPUArrays selection using operation/type/layout/device/library capabilities
  -> portable executor OR an already-selected vendor executor
```

Implement a small typed selection interface first for SpMV and SpMM, and apply the same boundary to regroup/conversion, assembly and SpGEMM as their vendor migrations land. Names can follow GPUArrays conventions, but the semantics must be concrete:

- Automatic selection may choose a supported portable or vendor algorithm.
- Forced portable execution bypasses vendor execution entirely.
- Forced vendor execution reports unsupported combinations before writing the output.
- Capability selection covers element and both index types, operand characters, layout/strides, device and library versions.
- Fall back because a capability is absent, not because an arbitrary vendor exception was caught after partial execution.
- Keep callable generic implementations for integrations and tests, but prevent recursive selection into the same public operation. Document whether subordinate operations may select implementations independently.

A scoped operation-specific entry point can expose algorithm/workspace choices without changing LinearAlgebra's standard `mul!` API. Do not make backend packages extend version-specific LinearAlgebra internals. Keep the Julia <1.13 forwarding adapters in GPUArrays.

Place high-level vendor integration in GPUArrays extensions. CUDA's extension should use cuSPARSE/CUDACore package boundaries; AMDGPU and oneAPI extensions can depend on their backend packages and invoke low-level vendor facilities. Remove or redirect old backend public methods that would bypass selection. Aliases and opinionated adaptors may remain backend conveniences, with their precision, format and index defaults preserved explicitly.

Do not invent a universal capability registry or a universal public planning API. The minimum proof is one real migrated vendor executor, forced-portable tests on the same buffers, and adding/changing a generic algorithm without editing that backend.

### 4. Separate transient scratch from persistent analysis

Reuse AK workspace facilities for temporary storage when their contracts fit. An AK workspace does not validate sparse pattern reuse.

Ordinary oneAPI operations need a concrete handle owner during migration, including originating context/queue, retained arrays, exception cleanup and deferred destruction. Existing optimized repeated operations need an explicit prepared object, not a fresh handle on every invocation that silently discards optimization. Document the transition from `sparse_optimize_*` entry points.

For prepared operations, choose and document a pattern lifetime: copy/snapshot the pattern, or borrow it while forbidding structural edits and resize. Holding an array reference does not pin its allocation. Buffer identity, pointer and length checks are useful diagnostics but cannot detect in-place index writes. Specify whether values can be updated in place or rebound. Bind independent concurrent executions to independent workspaces/descriptors unless the vendor explicitly supports sharing.

A future portable prepared conversion can retain an index permutation and gather **current** values. A future SpGEMM plan can retain symbolic structure. These can be added without changing sparse array fields. Do not require public APIs for every future plan before v13.

### 5. Define the supported Julia surface

Keep the PR's supported operations, subject to the corrections and assembly contract below. Do not claim generic SparseArrays inheritance makes everything work.

Before removing backend methods, preserve existing user-visible operations: constructors, transfer policies, `findnz`, linear slicing, supported wrappers, vendor products, triangular solves, factorizations/preconditioners and vendor-only formats. Use a per-package checklist; backend migration cannot mean losing previously accelerated solves until GPUArrays implements a portable replacement.

Add central value-based equality and alias-safe in-place broadcast for the advertised formats. A first implementation can materialize a canonical temporary and use checked `copyto!`; optimize values-only updates afterward. Handle zero-preserving and structure-changing cases explicitly. Never write a dense array through a scalar sparse loop. If nonzero fill or a dense broadcast into a sparse destination is intentionally unsupported, throw a precise error before mutation; don't quietly alter SparseArrays semantics.

Support common transpose/adjoint metadata, host conversion, reductions, broadcast and range slicing centrally, initially by explicit materialization where necessary. Complete `A[:]` via `vec`/copy with the documented ownership semantics. Equality must compare values rather than demand identical stored-zero patterns. Do not implement `hash` by hashing storage order; either implement array-compatible semantics or document it outside the initial surface.

Audit inherited paths including `hcat`, `vcat`, `blockdiag`, `UniformScaling`, `Diagonal`, views and mixed sparse/dense operations. Prefer straightforward central implementations where practical. Known unsupported paths must not silently download; give explicit errors and an accurately scoped operation table. Efficient sparse views, arbitrary index vectors, full concatenation and every structured wrapper need not all block v13.

Keep these deliberate differences explicit: non-zero-preserving broadcast and dimensional reductions may return dense GPU arrays; stored zeros may be retained; individual sparse insertion is unsupported; reshape currently copies. Avoid a broader claim of complete SparseArrays compatibility.

## Implementation work packages

Make each completed logical change reviewable and independently testable. Preserve the original author's commits where they still express the resulting design; avoid combining dependency reconciliation, interface changes and algorithm improvements into one commit.

| Step | Package(s) and change | Completion evidence |
|---|---|---|
| 1 | Reconcile #795/#796/#798 and #745 with current GPUArrays main. Preserve AK dependency/current compat, dense alias and wrapper fixes, and current tests. Target GPUArrays 13.0 and the appropriate new JLArrays version. | Correct merge base; range-diff; dense regression tests; Julia 1.10 resolution plus separate KA 0.9/0.10 environments as supported. |
| 2 | GPUArrays representation and contracts: separate `Vp`, checked integer arithmetic, raw borrowing docs, mutation preflight, `dataids`, device lookup. | Mixed Int64-pointer/Int32-coordinate tests; all four formats; unchanged buffers on rejected mutation; real GPU kernel lookup; host guards. |
| 3 | GPUArrays semantic corrections: symmetric/Hermitian filtering, compression and broadcast sentinels/counts, nightly copy ambiguity, robust norms. | Red/green repros below, nonfinite/complex tests, overflow before allocation/write, current nightly ambiguity test. Keep inherited numerical fixes identified as such. |
| 4 | GPUArrays Julia surface and fallback control: equality, in-place broadcast, common wrappers, linear slicing, explicit unsupported paths. | Scalar indexing disabled; output device/type assertions; aliasing tests; no implicit host result from concatenation/structured paths. |
| 5 | GPUArrays selection seam plus one actual CUDA/cuSPARSE product integration; move high-level selection to extensions. | Same input exercises automatic, forced portable and forced vendor paths; unsupported vendor case cleanly falls back only in automatic mode; no dispatch ambiguities or bypasses. |
| 6 | Remove obsolete sparse sorting workaround/JLArrays shims. Use current stable AK sorting, including a checked lexicographic alternative to packed-key overflow. | Duplicate order and empty cases on Metal, CUDA, OpenCL; no second sort; distinguish assembly combine semantics from reassociation permitted for products. |
| 7 | Backend migrations, each separately reviewable: CUDA family, AMDGPU, oneAPI, Metal; OpenCL integration/testing. | Existing backend sparse suites plus GPUArrays shared suite; preserved vendor-only operations; extension load order; oneAPI handle lifetime/reuse evidence. |
| 8 | Memory/performance and CI release gates; documentation, migration notes, PR text. | Hard-failing reverse CI; reproducible benchmark report; bounded SpGEMM behavior; registered dependency resolution; no outdated v12/OpenCL/Metal claims. |

Steps 2–4 can be developed alongside the selection prototype once step 1 establishes a trustworthy baseline. The interface is not frozen until the CUDA and oneAPI migration prototypes demonstrate it. AK enhancements below can proceed separately; do not make every optimization a release prerequisite.

### Backend-specific work in step 7

**CUDA / CUDACore / cuSPARSE / cuSOLVER**

Replace ordinary sparse formats with aliases plus compatible constructors. Update field users (`iPtr` to `nzInd`, `nnz` field to accessor), descriptors, `copyto!`, SpGEMM output mutation, type conversions, adaptation, precompilation and tests. Audit constructors taking explicit `nnz`/spare capacity: never silently treat extra buffer elements as stored entries. Preserve or explicitly deprecate those signatures with valid trimming/copy behavior.

Retain BSR and batched CSR as separate formats and supply conversions. Port their device representations to GPUArrays-owned facilities or keep narrowly owned backend representations; the PR deletes their old shared device structs too. Preserve cuSOLVER and preconditioner consumers. Test both Int32/Int64 uniform storage and mixed widths with explicit capability rejection where vendor support is narrower. Preserve `cu` policy unless deliberately changed in release notes.

**AMDGPU / rocSPARSE**

Migrate formats, accessors, device adaptation, precompile signatures and vendor wrappers. Preserve BSR, solves and factorization functionality. Test on pegasus's AMD GPU, not merely by loading the package. Treat integrated-GPU timings as noisy; correctness and dispatch coverage are still useful. The new generic paths must work for types outside rocSPARSE's supported set.

**oneAPI / oneMKL**

Migrate the handle/finalizer/optimization lifecycle before removing the old containers. Update descriptor binding and release behavior, `copyto!`, transpose/conversion overrides and test registrations. Preserve device/context affinity, including when the active context differs at destruction time. Exercise the portable path for element/index types not supported by oneMKL. Test on taurus, including empty matrices, same-length structural changes, changed `nnz`, GC, repeated optimized calls and nondefault tasks/contexts.

**Metal**

Rework #982 onto current main. Its old zero-length resize fix is already upstream; do not replay it as a new fix. Keep aliases/constructors and the explicit `mtl` precision/CSR/index policy, updating them for `Vp`. Remove temporary `[sources]` pins when releasing. Current KA 0.10 supports empty launches; retain guards only where still needed for the supported KA 0.9 path. Validate on this M3 Pro and another supported Apple generation in CI when available.

**OpenCL**

No vendor sparse layer is required. Test the generic implementation with current AK on pegasus and taurus devices. The investigation passed on PoCL CPU but failed on NVIDIA OpenCL with transfer and kernel-build errors; reproduce/minimize those failures from fresh contexts against main before attributing them to the sparse rewrite. Test both relevant OpenCL compilation routes/devices when available. PoCL CPU execution is valuable portability evidence but must not be reported as a GPU test. Registry/backend compatibility still needs coordination for v13 even when no sparse source changes are needed.

**Downstream users**

Audit users that subtype removed abstract types versus users that only dispatch on them. Deprecated abstract aliases preserve some dispatch but cannot preserve subtyping. Device-type aliases also have changed type parameters and storage predicates; do not add misleading host/device aliases simply to make imports succeed. Document the new kernel signatures and test representative consumers such as MLDataDevices, BSR/batched users and packages defining their own sparse formats. Version bumps for backend packages should reflect their public compatibility changes, not just GPUArrays' version number.

## AK and KA roadmap

Use the existing stable sort, scan, compaction, reduction and workspace facilities first. Keep matrix-specific formats, constructors, output policy and vendor selection in GPUArrays.

Prioritize these reusable AK additions based on measurements:

1. Stable radix permutation/key-value sorting, with workspace support, key-width/range handling and a portable merge-sort fallback. Current AK radix sort does not offer these paths.
2. Segmented reduction/scan and reduce-by-key, accepting compressed offsets or sorted keys. Define empty segments, output/count types, associativity, ordering and initialization. Use the ordered associative assembly contract below; do not accidentally promise the current strict sequential fold or permit arbitrary operand permutation.
3. Merge/union primitives useful for sparse-vector broadcast and work partitioning. Demonstrate at least the sparse consumer before generalizing an API.
4. Caller-provided output/count variants for compaction where fixed capacity or graph capture actually needs them. Dynamic exact-size constructors can legitimately synchronize for the resulting length.

Use KA/KernelInterface's existing subgroup, workgroup and atomic capability interfaces. Verify current released APIs rather than designing from the stale local KA checkout. Add a KA primitive only for a demonstrated missing portable kernel operation. Portable fallbacks must not assume 32-lane warps, independent subgroup progress, Float64 support or universal 64-bit floating atomics.

Subsequent GPUArrays work can then implement row-adaptive/workgroup SpMV, coalesced tiled SpMM, direct or prepared transposed products, faster regroup, merged sparse-vector broadcast, bounded symbolic/numeric SpGEMM, triangular solves, SDDMM and extra formats. These should not require backend sparse types or fresh allocation hooks.

Do not freeze bitwise reproducibility across versions/backends. Settle assembly before v13 with this contract:

> Duplicate values retain their original relative order and are combined using `combine`. Parenthesization is unspecified, so custom combiners must be associative; commutativity is not required. Floating-point addition is supported, but rounding may differ between algorithms, backends and versions.

This deliberately replaces the PR's strict left-fold promise. SparseArrays documents duplicate combination, while its current sequential implementation provides stronger observable ordering than the public contract. A nonassociative custom combiner is outside the GPU contract even if it happens to work on CPU. Stable sorting plus an order-preserving segmented tree is legal; arbitrary operand permutation is not. Test an exactly associative, noncommutative combiner and cancellation-sensitive floats using justified absolute/error-scale bounds, not only relative tolerance against the CPU fold. Deterministic product algorithms can later be selected explicitly.

## Validation and release gates

Use scratch environments and the julia-daemon workflow, with distinct daemon names and `--idle-timeout=2h`. For tests involving module or kernel redefinition, use fresh/no-Revise daemons or restart deliberately. Never resolve into the default environment or change the user's backend checkouts.

The shared suite must cover each generic format, Int32/Int64 coordinates, mixed pointer widths, supported value types including Bool/complex/Float16, zero dimensions, empty major slices, stored zeros, duplicate assembly, range boundaries, copy/adapt aliasing, values-only mutation, structural growth/shrinkage, rejected storage, wrappers and direct user kernels. Include small narrow-index tests to exercise limits without allocating billions of entries.

Run three distinct layers and report them separately:

1. **Generic semantics:** JLArrays on supported Julia releases and current nightly; preserve whole-suite dense tests affected by dispatch. Add sparse ambiguity checks with SparseArrays/LinearAlgebra and after loading each backend extension.
2. **Physical execution:** Metal locally; CUDA and AMDGPU on pegasus; oneAPI on taurus; OpenCL on both. The shared portable suite must also run with vendor execution explicitly disabled. Test actual migrated packages, not only CUDACore.
3. **Vendor/resource parity:** existing backend sparse/solver suites, descriptor lifetimes, context affinity, optimized repeated operations, unsupported capability fallback, extension load order and package resolution. A soft install failure is a release failure.

Benchmark warm and first-use execution separately, synchronizing correctly. Use diagonal/banded, uniform-random, power-law/skewed, empty-row and duplicate-heavy structures, plus representative SuiteSparse workloads. Include both CSR/CSC, transposes, complex adjoints, several SpMM widths and SpGEMM with heavy candidate expansion. Report allocation and peak temporary memory alongside time. Compare the old backend vendor path, the migrated vendor path and portable execution on the same hardware/input.

Before v13:

- Existing accelerated backend functionality remains available and the selector cannot silently choose a known large regression.
- SpGEMM checks candidate-count arithmetic before allocating. Provide a bounded-memory path or an explicit documented resource limit/error; do not let a tiny final result require uncontrolled candidate materialization.
- No known silent wrong result, structural corruption or implicit download remains in the advertised surface.
- Reverse CI actually installs and tests the coordinated revisions. Latest-nightly failures are triaged explicitly, not hidden behind a green allowed-failure job.
- Documentation and PR descriptions target v13, accurately describe synchronization, and distinguish supported operations from deferred work.
- Tag/release ordering is coordinated across GPUArrays/JLArrays and the migrated backend packages. Test unreleased revisions with temporary sources first, then verify registered resolution. Publishing, posting and merging are separate maintainer actions, not part of this investigation.

## Reproduction snippets for the next session

On exact PR head with JLArrays and scalar indexing disabled:

```julia
using GPUArrays, JLArrays, SparseArrays, LinearAlgebra, Adapt
using GPUArrays: GPUSparseMatrixCSC, GPUSparseVector
GPUArrays.allowscalar(false)

# Excluded symmetric terms must not manufacture NaNs.
A = adapt(JLArray, sparse([1. 2.; 4. 3.]))
Array(Symmetric(A, :U) * JLArray([Inf, 1.]))  # [NaN, NaN]; CPU: [Inf, Inf]

# Legal coordinate collides with broadcast's absent-entry sentinel.
x = GPUSparseVector(JLArray(Int8[127]), JLArray(Float32[2]), 127)
y = x .+ x
(nnz(y), Array(y)[end])                     # (0, 0), expected (1, 4)

# Compression's final search key wraps.
p = GPUArrays.compress(JLArray(Int8[127]), 127)
Array(p)[end]                              # 1, expected 2

# Accepted view storage is corrupted before resize! throws.
p = JLArray([1, 2, 3]); i = JLArray([1, 2])
v = JLArray([1., 7., 0., 9.])
A = GPUSparseMatrixCSC(p, i, view(v, 1:2:4), (2, 2))
try dropzeros!(A) catch err; @show typeof(err); end
(Array(A.colPtr), length(A.rowVal), length(A.nzVal)) # ([1,2,2], 1, 2)

# Inherited norm instability; compare with norm(nonzeros(A)).
S = sparse([1], [1], [1f30], 1, 1)
(norm(S), norm(adapt(JLArray, S)))           # (1f30, Inf)

# Host fallback must be handled or explicitly rejected.
A = adapt(JLArray, sparse(Float32[1 0; 2 3]))
typeof(hcat(A, A))                          # host SparseMatrixCSC
```

For device indexing, pass a sparse matrix to a KA kernel that assigns `out[i] = A[i,i]`. The unmodified PR fails compilation on Metal through `assertscalar`. A scratch prototype removing only the outer sparse checks preserved host scalar guards and enabled device lookup. Do not remove dense-buffer guards.

The exact-PR baseline, source overlay, scripts and detailed logs remain in `/Users/tim/Julia/tmp/gpuarrays745-review`; remote portability scratch is `/home/tim/Julia/tmp/ga745-review` on pegasus. These are evidence, not branches to merge. Start implementation from fresh current revisions and this plan, and rerun every release-relevant claim.
