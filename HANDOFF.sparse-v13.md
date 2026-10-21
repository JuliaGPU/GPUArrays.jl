# GPUArrays v13 implementation handoff

## Worktrees and branches

All implementation uses fresh isolated worktrees under `/Users/tim/Julia/tmp`:
`gpuarrays-v13`, `metal-v13`, `cuda-v13`, `amdgpu-v13`, `oneapi-v13`, `opencl-v13`.
Each branch is `tb/sparse-v13`. The investigation worktree and unrelated backend
working copies are untouched.

Fetched upstream on 2026-10-21: GPUArrays main `6fa6490`, PR #745 `2a50d28`;
Metal main `e4412098c`, PR #982 `ce1ba502f`; CUDA `44ac4ebb3`;
AMDGPU `284e54858`; oneAPI `705aeb0`; OpenCL `9c27048`.
Local backup refs `backup/sparse-v13-pr745-20261021` and
`backup/sparse-v13-pr982-20261021` preserve the previous PR tips.
The existing remote PR branches have not yet been rewritten.

GPUArrays retains all ten original contributor commits rebased onto main. Metal
retains its two integration commits, omitting the zero-resize fix already upstream.
GPUArrays is 13.0.0; JLArrays is 0.5.0. Existing AK dependency/compat and main's dense
alias, wrapper and reference-algorithm fixes are retained.

## Implemented and pushed

GPUArrays `2c797a8` is pushed to `JuliaGPU/GPUArrays.jl:tb/sparse-v13`:
- `9b116ed`: robust sparse norms through the scaled dense implementation.
- `52b2e7e`: explicit dense `supports_resize` capability, including JLArrays.
- `9b6b82b`: separate compressed-pointer storage, mixed-width conversions,
  borrowing/mutation/alias contracts, device lookup, checked counts and allocation
  sizes, stable assembly with lexicographic fallback, nonfinite filtering, and
  bounded ESC SpGEMM estimates.
- `d5e5050`: initial GPUArrays-owned SpMV/SpMM selection, value equality,
  in-place broadcast, common transpose metadata and explicit concat rejection.
- `2c797a8`: shared v13 contracts.

OpenCL `9526a11` is pushed to `JuliaGPU/OpenCL.jl:tb/sparse-v13`: v0.11,
GPUArrays 13 compat, resizing capability rejecting system-memory wrappers.
Physical execution and package release resolution are still pending.

Metal `f52178f83` is local: five-parameter aliases, resizing capability, checked
transfer counts, removal of the old dependency pin, and default-storage-aware tests.
No vendor code is needed for Metal. Do not claim complete migration until the shared
suite and load-order/ambiguity tests pass.

CUDA has an uncommitted direct-alias migration in `/Users/tim/Julia/tmp/cuda-v13`,
with matching GPUArrays `ext/CUSPARSEExt` selection/solves. Its split packages are
provisionally v7.0.0. Syntax parsing passes, but package loading and hardware parity
are still pending; do NOT call this a completed migration. AMDGPU and oneAPI still
only have uncommitted dense resize-capability declarations. Their sparse migrations
and oneAPI resource ownership remain unimplemented.

## Validation so far

Julia 1.13.2 / JLArrays / KA 0.9.45 / AK 0.5.8:
- Rebased baseline shared Float32 sparse suite passes.
- Focused red run: 1 pass, 12 failures, reproducing the report.
- Corrected focused suite: 72/72 passes, including mixed-width and mutation cases.
- New shared contract suite: 101/101 passes, with scalar indexing disabled.
- Existing shared Float32 sparse suite passes after the type-parameter updates.
- Relevant dense broadcasting/scan/predicate checks pass; full dense gates remain.
- Sparse ambiguity detection is empty on Julia 1.13.2.

Julia nightly 1.14.0-DEV.3743: sparse ambiguity detection with GPUArrays, JLArrays,
SparseArrays and LinearAlgebra passes (empty ambiguity list). Nightly full suite is
not yet run. The exact binary is
`~/.julia/juliaup/julia-nightly/Julia-1.14.app/Contents/Resources/julia/bin/julia`.

Metal M3 Pro / Julia 1.13.2 / KA 0.10.1 / AK 0.5.8:
50/50 backend constructor/transfer/densification checks pass. The shared contract
suite is being exercised with private storage. A constrained UnionAll array type
has no constructor; use a callable `x -> MtlArray{eltype(x),ndims(x),Metal.PrivateStorage}(x)` for
the focused helper, or configure the ordinary MtlArray default for the whole suite.
Main now defaults to SharedStorage; do not replay the old private-default assertion.

Validation used changing worktree snapshots. Rerun release-relevant tests after
final commits and report exact revisions. No CI stack validation has passed yet.

## Reviews and architectural reasoning

`peer log sparse-v13-architecture` (Astra) confirms central selection and recommends:
- Qualified cuSPARSE/rocSPARSE routines as executors, with GPUArrays-owned selection
  and normalization. Preserve CUDA #3416 integer/Bool input normalization.
- A synchronous explicit oneAPI owner retaining the actual SYCL queue, originating
  stream/context/device and arrays; integrate current managed-memory ownership and
  batch flushing, check Cint statuses, and dispose deterministically.
- Prepared reuse must be explicit, borrowing fixed storage with a prohibition on
  structural edits, resize, rebinding and free. Values may update between completed
  operations; cross-domain execution initially rejects before writes.
- oneAPI's installed support artifact must match the checked-in synchronous ABI.
  Existing complex-CSC wrappers mutate dense inputs; replace this with scratch.
- Canonical arrays may not represent unfinished/noncanonical vendor intermediates.
  CUDA column-sorted COO and sorted=false APIs require explicit migration notes.

`peer log sparse-v13-review` (Sol) has identified nine issues in two passes. They
were reproduced and corrected: instance resize capability; independent SpGEMM
pointer promotion; coordinate-aware expansion; default matrix pointers without a
compressed prototype; checked empty-pointer bytes; broadcast dispatch intersections;
mixed COO/compressed pointer promotion; direct portable subordinate conversions;
and wrapper/mixed concat guards. A refreshed third review is running. Verify all
findings independently and continue the conversation for later migrations.

## Running validation and remote scratch

Daemons use unique names and 2h idle timeouts:
`ga13-baseline`, `ga13-green`, `ga13-nightly` on GPUArrays test project;
`metal13` on Metal root and test project. Test daemons use --no-revise.
jld root/test separation matters; KernelAbstractions is a Metal weakdep and is
directly importable in its test project. Pkg.develop adds local sources/direct deps:
keep those changes out of production commits. Manifest files remain local.

Pegasus scratch checkout `~/Julia/tmp/gpuarrays-v13` is at pushed `2c797a8`.
Its origin is GitHub (a --shared clone initially pointed at the local source).
No existing remote ~/Julia/pkg working copy was modified. Both hosts became unreachable during implementation; see the latest checkpoint below.

## Next work and release gates

1. Finish refreshed core review, broader types, direct GPU lookup/host guards,
   overflow/resource-limit and dense regression coverage; run Julia 1.10.
2. Complete Metal shared tests and ambiguity/load-order checks; push migration.
3. Implement actual CUDA/cuSPARSE vendor selector and aliases, preserve BSR/batched
   representations and cuSOLVER consumers, canonicalize producing adapters, and run
   physical portable/vendor/solver suites. Legacy Cint ABI boundaries must check
   pointer width as well as coordinate width.
4. Migrate oneAPI explicit resource owner and prepared operations before removing
   old containers. Then AMDGPU and its solves/BSR/factorizations.
5. Test OpenCL NVIDIA failures from fresh contexts against main, PoCL CPU separately,
   and taurus routes/devices; never label CPU portability as GPU validation.
6. Hard-fail downstream install errors and nightly tests. Add exact coordinated
   temporary pins separately, including CUDA split-package test environments and
   explicit Julia 1.10 Pkg.develop setup. Inspect logs and exact tested revisions.
7. Performance/memory measurements and release-resolution checklist, local PR drafts,
   lease-checked updates of existing PR branches. Do not post GitHub discussions.

The selector is provisional until real CUDA and oneAPI integrations validate it.
Nothing here is a release-ready coordinated stack yet.

## Latest checkpoint (supersedes earlier validation counts)

GPUArrays pushed through `44b9e3b`; local central SpGEMM/open-format selection is
`3b72005`. New fixes include `7bba4e2` (CI installation/nightly hard failures),
`bcf3422` (wide equality and all-position concat guards), `7e23537` (checked values
and single-column conversions), `c09c1af` (real/complex value preflight and storage
placement), `1b6757e` (Metal count-loop workaround), `44b9e3b` (scalar policy tests).
Focused JLArrays regressions pass 103/103 on Julia 1.13.2. Full shared sparse runs
are ongoing against these changes; final exact-commit reruns remain required.

Metal's full private-storage suite exposed a genuine count-kernel miscompilation:
CSR/CSC with one slice [1,2,3,4] and an empty slice, Int32 pointers and Int64 counts,
can count 1 instead of 4 with an unchecked final store. Sol independently reproduced
it, including fixed-arity kernels. Removing that store's @inbounds restores the
counts; optimized LLVM loop bodies are identical except SSA names. The failing
stage beyond optimized LLVM is not established. Keep wide counts. The production
workaround retains a bounds-checked store and explicit layout tag; shared tests
cover both formats and Int32/Int64 pointers. The prior layout-tag-only snapshot
passed 200 repeated product/update checks and 69 focused SpGEMM assertions. Current
full private suite must pass before claiming this fixed across the stack.
Scratch repros: `/Users/tim/Julia/tmp/ga13-metal-{copy,iterator}-repro.jl`.

Pegasus fresh NVIDIA OpenCL context (RTX 5080, CUDA driver 13.4.96) passed the
101 shared v13 contracts using GPUArrays `2c797a8`, OpenCL `9526a11`, AK `b13f97e`,
Atomix `3486963`. The full suite stopped at scalar-rejection expectations because
CLArray's SVM storage deliberately permits host scalar access. `44b9e3b` makes the
check follow the component policy, not skip sparse results. Rerun full NVIDIA and
separate PoCL CPU suites after synchronizing final revisions; neither is a full
backend pass yet. No hardware equivalence is inferred from PoCL CPU.

AK `a915b41` and Atomix `1ffd4ae` are pushed to their `tb/sparse-v13` branches:
OpenCL 0.11 and provisional CUDA7/AMDGPU3/oneAPI3 compatibility. These are justified
by actual package-resolution failures against the new OpenCL major/minor. A split
CUDA scratch stack now resolves with explicit Pkg.develop paths in
`pegasus:~/Julia/tmp/ga13-cuda-env`; installation is not migration validation.
All remote work is in ~/Julia/tmp, with uniquely named, two-hour-idle jld daemons.

## Checkpoint after physical CUDA and refreshed Metal runs

GPUArrays latest committed extension/selection snapshot is `22df6fb`; central mask
slicing is `70fdfca`, triangular-wrapper preservation/component alias preflight is
`2f996a1`. JLArrays focused regressions now pass **142/142**, sparse ambiguities
empty on Julia 1.13.2. The entire shared sparse suite passed all default JLArray
value types before the mask/triangular additions. Julia **1.10.12** explicit
Pkg.develop resolution and strict precompilation pass (AK0.5.8, KA0.9.45,
Atomix1.6.0). Focused regressions142 pass there; full suite initially stopped because
the scratch harness lacked a direct KA dependency, being corrected.

Metal migration `a763c7fb6` is pushed to `tb/sparse-v13`. Full shared sparse tests
pass Float32/ComplexF32/Int32/Float16 against GPUArrays `3b72005` plus the extension
worktree snapshot (including checked count-store). Dense broadcast specialization
previously covered arbitrary AbstractArray destinations, creating a sparse copyto!
ambiguity; narrowing it to AnyGPUArray resolves this and preserves sparse errors.
Fresh current Metal tests: backend53 assertions, shared112 contracts, supported
dense broadcast groups, and sparse ambiguity check all pass. The ignored test
LocalPreferences.toml still selects private storage.

Pegasus CUDA stack (Julia1.13.2, runtime13.4, cuSPARSE12.8.6, RTX5080) physically
passed:
- Shared v13 contracts109 and forced portable/vendor product comparison60.
- Migrated cuSPARSE construction217 plus Float16 construction4.
- Existing BSR solve groups80 + 80 + 432; IC/ILU preconditioners20 + 20 + 16 + 24.
- Qualified SpMV/SpMM, dense conversions and ordinary triangular vendor groups
  passed in the generic suite. SpGEMM updates passed after replacing an obsolete
  different-pattern error expectation with the now-supported numerical result.
  The suite then exposed sparse-vector -> compressed default pointer widening in
  qualified gemv; fixed locally with explicit uniform output construction, not yet
  rerun. Batched/device suites have NOT run because generic.jl stopped first.
These used changing uncommitted CUDA/extension snapshots, not final commit hashes.

CUDA local history includes resizing `6b15bb1cf`, family-v7 declaration
`e530fab77`, descriptor storage retention `7435b2a08`, legacy both-width guards
`dcb0568af`. The large alias/output-builder migration remains uncommitted, as do
the latest physical-test follow-ups. Sol source review found and prompted fixes:
legacy Cint IC/ILU/coloring guards; descriptor-owned operand lifetimes; accumulator
precision eligibility; BSR component alias checks; shared Boolean masks; missing
SpMM version guards; destination-format conversion in qualified gemm!. New CUDA
sparse_v13.jl rejection/cancellation/alias regressions are not hardware validated.
The latest BSR metadata/copy preflight and scalar indexing corrections are also
not rerun. Do not describe all CUDA functionality as migrated.

oneAPI now has an uncommitted direct-alias/resource/qualified-operation rewrite,
plus GPUArrays ONEMKLExt (`22df6fb`). sparse_resources.jl owns captured actual SYCL
queue, stream/context/device/task and buffers; prepared objects borrow fixed
storage and constrain optimized-operation parameters. Scoped calls close
deterministically. All support statuses are checked. Complex CSC uses scratch;
effective RHS widths are used. Astra source-reviewed and required removal of the
blocking reclaim callback (could invert allocation locks), preservation of handles
on pre-release failure, full-resource quarantine after uncertain native release,
skipping already-closed finalizers, foreign-task drain without mkl_dirty changes,
and solve/scalar preflight. Those fixes are local. Exact RHS/output alias support
still needs vendor confirmation; stage transposed in-place TRSM until proven.
No oneAPI code has loaded or executed on hardware yet; installed ABI unverified.
AMDGPU migration is still unimplemented beyond resizing capability.

At approximately 12:25 local time, **both pegasus and taurus began returning
"No route to host" on SSH**, reproducible across several attempts. Taurus setup
disconnected during Pkg.resolve after jld bootstrap. Its new scratch trees:
~/Julia/tmp/{oneapi-v13,gpuarrays-v13,ak-v13,atomix-v13,ga13-oneapi-env}.
jld was absent there; installed JuliaDaemon from its actual app-manifest repository
https://github.com/KristofferC/JuliaDaemon.jl (master), then used unique two-hour
daemons. No ~/Julia/pkg edits. Ask user about LAN/VPN restoration; continue local
work. Pegasus latest source sync predates the descriptor-retention/ABI/precision/
mask follow-ups. Explicitly resynchronize and rerun after access returns.

Fresh NVIDIA-only ICD run reproduced full-suite failure: construction144 and
allocation97 pass, transfer16 then packed-key kernel compilation fails, followed
by CL_UNKNOWN_ERROR_CODE/finalizer errors. Fresh isolated packed_keys on three
entries passes; a separate fresh isolated sparse_transfer run passes20/20.
Therefore the failure is sequence/context dependent, not yet a proven bad
packed-key kernel. Full logs are in jld transcripts of ga13-nvidia-isolated,
ga13-nvidia-min and ga13-nvidia-adapt in pegasus ga13-opencl-env. The earlier
nonisolated run had CL_OUT_OF_HOST_MEMORY creating a context; explicitly creating
a new NVIDIA context subsequently worked. Keep Intel ICD enumeration warnings
separate from NVIDIA-only reproduction.

## Current checkpoint (supersedes incomplete status above)

LAN access works: Pegasus directly; Taurus with `ssh -J pegasus.maleadt.lan
taurus.maleadt.lan`. Work remains under `~/Julia/tmp`, never remote pkg trees.

GPUArrays HEAD is 7889d6c (pushed through f8e31c4); current uncommitted changes
include ROCSPARSEExt, constructor/shape compatibility, diagonal arithmetic,
trace overflow semantics and Kronecker test rounding. Fresh Julia 1.13.2 strict
precompilation and focused JLArrays regressions pass 303/303; no sparse
ambiguities. This does not establish a globally ambiguity-free package: eight
unrelated dense ambiguities remain. Julia 1.10.12 full shared sparse tests passed
Float32/ComplexF32/Int32/Float16 before the newest compatibility additions.

CUDA RTX5080: qualified generic cuSPARSE suite, batched53, canonical device14,
CSC column views3, ABI16, cancellation36, BSR alias6, cuSOLVER reorder12/numeric100
and factorization36 pass. Remaining actual suite stops in conversion.jl on old
constructor-identity expectations and missing typed BSR conversions. Array490,
duplicate9, utility32, batch7, conversion52, half12, Bool29 passed in that run.
The full shared suite exposed integer trace widening and exact complex Kronecker
rounding expectations; fixes are local, full rerun pending. Large CUDA migration
is uncommitted; pushed branch only contains earlier preparatory commits.

oneAPI Arc A770: full actual oneMKL suite passes746 sparse assertions with24
pre-existing broken CSC triangular cases; supported dense BLAS/LAPACK also pass.
Full shared sparse suite passes Float32/ComplexF32/Int32/Float16. Resource and
prepared-value-refresh tests pass119. Latest source HEAD1f0cb7c is not pushed
(remote f93f70b). Optimized bindings use independent bit-exact value snapshots,
re-preparing when source values change; PLAN records Astra's contract review.
Transposed dense SpMM is rejected before writes and automatic selection uses
portable execution. An unhandled native oneMKL exception previously aborted the
process; C++ generator/epilogue/generated catches now contain all std::exception
and unknown exceptions. A task-owned support build was actually loaded and
native error containment passes2. Regenerated final C++ source still needs
synchronizing locally/rebuilding/full resource rerun. Official support artifact
does NOT contain this fix: coordinated artifact release remains mandatory.

AMDGPU gfx1036: migration loads and strict-precompiles; shared contracts109,
forced portable/vendor/solve43 pass. Existing generic/vendor solve groups,
preconditioners24, triangular solves288 and conversion groups pass. Rectangular
SpMM54 layouts across three policies passes165, ABI preflight9. COO transpose
descriptor normalization fixes unsorted vendor interpretation and retains its
central canonical temporary. Remaining interfaces suite failures concern BSR
arithmetic normalization, rectangular UniformScaling, scalar wrapped arithmetic
and scaled-product precision. Entire AMD migration remains uncommitted.

Sol reviews through turn13 (`peer log sparse-v13-review`) verified descriptor,
ABI, placement and geometry fixes. Latest remaining finding: qualified AMD
gemm! must reject zero-based index before central delegation and mutation.
Astra reviews through turn11 are in sparse-v13-architecture. No GitHub discussion
was posted. Metal remains a763c7fb6, OpenCL9526a11, AKa915b41, Atomix1ffd4ae.

No coordinated CI stack has run yet; neither existing PR branch has been
rewritten. Need exact dependency pins (including Julia1.10 explicit develop and
CUDA split workspace), remote-tip backups/concurrency checks, commit all logical
changes, push, then inspect real job logs and revisions. Full dense regressions,
final extension/load-order checks, representative performance/memory measurements,
OpenCL sequence minimization and hardware coverage remain incomplete. These are
not release-ready results or completed backend migrations.
