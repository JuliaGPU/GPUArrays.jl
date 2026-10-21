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

CUDA, AMDGPU and oneAPI only have uncommitted dense resize-capability declarations.
Their sparse containers, solves, factorizations, BSR/batched paths, vendor integration
and lifetimes have NOT been migrated.

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
has no constructor; use a callable `x -> MtlArray(x; storage=PrivateStorage)` for
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
No existing remote ~/Julia/pkg working copy was modified. Both hosts are reachable.

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
