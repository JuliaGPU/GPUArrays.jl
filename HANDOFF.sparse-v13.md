# GPUArrays v13 implementation handoff

## Current state (Oct21, final source validation)

This section supersedes every historical checkpoint below. GPUArrays local4083b78
contains four reviewed fixes after pushed ef7659a: mixed sparse broadcast styles and
preallocation rejection, logical negative-order norms, CPU-MIME-independent display
tests, and scalar-magnitude nonfinite complex norm references. Runtime source at
4083b78 equals frozen64ea2d8; only the final regression reference differs. Final
central pins are CUDAa3e0c60, AMDf5b6976, oneAPId1cb45d, Metal1aaa653 and OpenCLe0a09f4,
plus AKa915b41/Atomix1ffd4ae. Pin/helper/timeout changes will be folded into the single
TEMP CI commit, not pushed as another series of debug commits. Preserve tested source
SHAs and the old remote tip before that rewrite; do not relabel historical tests as
literal tests of a new commit hash.

Exact clean-source results:

- GA64ea2d8/OpenCLe0a09f4: actual root Pkg.test sparse/SVM suites pass11,759 on
  EACH host, Pegasus RTX5080 and Taurus RTX6000Ada. Shared sparse11,744 across all
  ten supported types, SVM15, normal GC/synchronization, one worker. An empty
  OCL_ICD_VENDORS directory plus native OCL_ICD_FILENAMES constrained enumeration
  to NVIDIA only. Both logs: ga13-opencl-svm-env/actual-final-entry-full.log.
  This is actual migrated GPU coverage, not PoCL evidence or a whole-backend pass.
- GA64ea2d8/oneAPId1cb45d: actual checked Pkg.test, allow_reresolve=false, passes
  13,138 plus24 pre-existing broken checks: sparse ABI96, shared11,641, native
  oneMKL1279+24broken, resources122. Taurus ga13-oneapi-env/final-marked-stack.log.
  The loaded rebuilt support SHA256 is
  c19915c2965bc4f7153e4c16d240deb107ecfe0a902f07936445d5f9969932c9 at
  ~/Julia/tmp/gpuarrays-v13-stack-lyFPFO/support-scratch/8f75cd03-7ff8-4ecb-9b8f-daf728133b1b/deps/lib/liboneapi_support.so.
  This is a source-built stack, not the registered artifact. Actual published0.11.0
  is rejected for its absent version marker; the new library reports2. Dense oneAPI
  still executes with sparse eligibility disabled; real-artifact probes6/6 pass.
- GA64ea2d8 full shared JL sparse on Julia1.15-DEV61 passes14,957; ambiguity1.
  Expanded regressions initially1890pass/4fail exposed CPU BLAS versus scalar complex
  Inf/NaN reference differences, not a runtime sparse failure. Corrected GA4083b78
  actual nightly Pkg.test passes1894regressions+ambiguity1. Pegasus
  ga13-nightly-final-env/{nightly-logical-final,nightly-norm-final-regressions}.log.
- Local Julia1.13.2, runtime4083b78: regressions1894 and contracts154 pass;
  focused display52, linalg195, mixed broadcast275, ambiguity1 pass. The broadcast
  regression was independently red9/9 then green9/9. Negative orders now include
  implicit zeros on every Julia version, preserving NaN magnitudes. Dense operands
  may broadcast-expand; sparse expansion/mixed sparse ranks reject before allocation.
- GA64ea2d8/AMDf5b6976: GPUArrays-first contracts151+ambiguity1 pass; earlier backend-
  first ef7659a/f5b6976 contracts109+ambiguity1 pass. Pegasus
  ga13-amdgpu-loadorder-final-env/core64-gpuarrays-first.log.
- Metal1aaa653/GAef7659a: fresh actual Pkg.test spot53/53, --jobs=1 and explicit
  --threads=1. Local ga13-metal-final-spot-prefix.log. No complete local Metal suite
  and no physical Metal retest of the final central broadcast/norm changes claimed.
  The user's limit remains single-threaded spot testing, not full local testing.
- CUDAa3e0c60/GA64ea2d8 refreshed focused contracts/linalg/broadcast run is pending;
  Pegasus ga13-cuda-env/final-core-focused.log. Retain older full automatic/portable
  and native solver/factorization/BSR/batch evidence below at its exact revisions.

The inherited NVIDIA dense-allocation churn remains independently reproducible on
OpenCL main9c27048/registered GA12.1.1, including strict NVIDIA-only discovery. A
native C shim observes NULL(size44,alignment4), alloc/fail/free=(15,1,3) at failure
and(15,1,11) after unwind, after ten checks; baseline-strict-nvidia.log. The root
sparse suites above do not reproduce it. Cause unresolved; neither a v13 regression
nor a driver defect is established. The independent NULL guard and vendor-driver
test initialization fix are production changes, not alleged causal fixes. See
validation/REPORT.opencl-svm.md and its dense-only reproducer.

Actual CI run39625043230 tested synthetic merge32c670f409c65fd095df6f0cdff9a541cc3ac788
(main6fa6490 + ef7659a), NOT the local fixes above. Ubuntu/Windows1.10–1.13 and docs
pass; Windows1.12 actual35,416 assertions. Both nightly jobs fail21 (18 implicit-zero
norm assertions,3 CPU display-format references), now addressed by the frozen-source
tests above but awaiting new CI. OpenCL hit its exact60-minute timeout while both
IL/C sparse suites and dense jobs were still running; no completed aggregate pass.
Logs were inspected. The final isolated CI layer gives this unshortened two-route
suite180minutes, rather than skipping tests or allowing failure. macOS jobs remain
queued/running. Buildkite2082 canceled and2083 all-broken are NOT backend validation.

oneAPI's native artifact gate probes wrapper version2 plus wide CSC using one loaded
handle. All87 raw sparse wrappers gate before argument conversion/native entry;
96 ABI checks verify this behavior. Regeneration retains unrelated generated files.
The official0.11.0+0 library lacks this marker despite exporting wide CSC, so must
not be accepted as safe. Sparse support fails closed; dense behavior remains usable.
Yggdrasil topic maleadt/Yggdrasil:tb/oneapi-sparse-v13-support is pushed at31c79455b,
on latest master0defdb4d8. Separate e0dc775 fixes an existing extra-parenthesis parse
error. Version0.11.1 retains upstream source5093529 and Intel2026.1, carrying only
the focused native cpp/header patch; patched files cmp equal tested oneAPId1cb45d.
Runtime dependency floors are the validated Loader1.32/OpenCL2024.10.24/Hwloc2.15,
not claimed minimal ABI requirements. Julia1.13.2 root-project BinaryBuilder metadata
generation passes for x86_64-linux-gnu-cxx11. Actual crossbuild/generated-JLL test is
pending on Pegasus (ga13-ygg-build.log), with task-owned JULIA_PKG_DEVDIR. Metadata
and the source-built Arc result are not a generated artifact validation.

Release/maintainer checklist (temporary pins are not release-ready resolution):

1. Finish refreshed exact-stack CI and inspect checkout + STACK_REVISION lines,
   including CUDA's root/all13 split packages and their test environments. Use the
   explicit develop/setup path on Julia1.10; [sources] alone is ineffective there.
2. Build/test/publish and register oneAPI_Support0.11.1 with wrapper contract2;
   test the generated artifact WITHOUT the source-build preference. Raise oneAPI's
   support minimum to the actual registered version, retaining the runtime gate.
3. Release compatibility-only AK0.5.9/Atomix1.6.1, GPUArrays13/JLArrays0.5 (Core stays
   0.2.1), then coordinated CUDA7 (all14 packages), AMDGPU3, oneAPI3, OpenCL0.11,
   Metal1.12 as their weak-dependency ordering permits. Verify every test project.
4. Land downstream compatibility/API updates recorded in
   validation/REPORT.sparse-v13-downstream.md; whole consumers were not tested.
5. Remove the isolated TEMP CI/bootstrap pins when registered resolution works;
   rerun normal Pkg.test/docs on Julia1.10 and supported modern/nightly versions.
   Before then describe results as exact pinned-stack validation only.
6. Keep the inherited OpenCL dense NULL reproducer open; old cuSPARSE algorithm
   branch performance, additional GPU generations and full local Metal remain
   untested. CUDA-only measured performance/memory is in perf/REPORT.sparse-v13.md.

Astra architectural decisions through23 and Sol implementation review through76 are
in peer conversations sparse-v13-architecture and sparse-v13-review. Findings were
checked and fixes rerun. No GitHub discussions or PR descriptions changed yet. The
user authorizes succinct existing PR745/Metal982 description updates once work is
finished; new backend/dependency PR descriptions remain local drafts, not posted PRs.

## Historical checkpoints

## Latest checkpoint (Oct21, CI history cleanup and final native gates)

This section supersedes the chronological checkpoints below. GPUArrays ef7659a pins
CUDAa3e0c60ee, AMDf5b69764c, oneAPI4858db9, Metal89143a6ed and OpenCL0760462.
All topic and existing PR branches are pushed. Per the user's concern,14 incremental
TEMP CI commits were consolidated into ONE isolated CI commit. Production commits
and contributor authorship remain; the full tree before repinning was byte-identical
to fc9bda1. Old remote65ff6ec and full fc9bda1 are preserved in local/remote backup refs
backup/sparse-v13-{ci-debug,before-ci-cleanup}-20261021. Backend bootstraps pin cleaned
dd0ee48, retained under backup/sparse-v13-bootstrap-20261021; new central pins are the
only tree difference from that cleanup. Backend previous tips have their own local/
remote backup/sparse-v13-before-bootstrap-repin-20261021. All rewrites checked remote
heads and used explicit leases/refspecs. This remains stack validation, not normal
registered dependency resolution.

Every backend's CI now explicitly resolves the pinned stack rather than letting an
early test-plugin instantiate registered incompatible versions. Existing toolkit,
allocator/memory, sanitizer, multi-GPU, minimum-dependency, Enzyme, debug/validation,
OpenCL driver/memory/translation and docs/performance variants are retained. Nightly
and dependency installation failures are hard failures. CUDA's memory matrix now
writes CUDACore's preference (not obsolete CUDA's), and the helper asserts the loaded
memory mode. Custom dependency jobs change the disposable root environment and keep
their selections through the checked runner. Preferences retain recursive __clear__
semantics and transitive UUID/name bindings via extras. Binding, actual Pkg.test
sandbox propagation and preference-clearing probes pass on Julia1.10.12 and1.13.2.
Sol turns28-30 reviewed these changes; two verified wiring issues were corrected.
YAML parsing and diff checks pass. Actual checked Pkg.test runner ecbe532 with CUDA
42d9ec7 passes228 in confirmed unified memory, allow_reresolve=false/check-bounds=yes;
AMD47a61c2 passes3189 native sparse+dense solver; oneAPI80a88b3 passes1401 native/
resource checks plus24 pre-existing broken CSC checks. Their runtime files are unchanged
by the repin; do not label the new bootstrap SHAs hardware-tested yet. Logs Pegasus
ga13-cuda-ci-entry/self-unified-final-green.log, ga13-amdgpu-env/self-final-green.log;
Taurus ga13-oneapi-env/self-final-green.log. oneAPI rebuilt support SHA256
6d0ee79ec8b9a91034b0442e8e9d7e54e493019aaf36db76376fa4241be13e55 at
/tmp/gpuarrays-v13-stack-sttpHy/support-scratch/8f75cd03-7ff8-4ecb-9b8f-daf728133b1b/deps/lib/liboneapi_support.so.
Initial invocations accidentally appended daemon startup ARGS; reruns cleared ARGS
before the checked runner. That was a harness error, not a production CI failure.
No further full local Metal runs: single-worker/thread spot tests only.

Clean frozen GA0f2d80f / CUDAe37f85d19 full shared six-type suites now pass automatic
7712 and forced-portable7713 (2004 portable selections), with an empty sparse ambiguity
check each. Clean GA0f2d80f / AMDe3a3a604c shared four-type suites pass automatic5540
and forced-portable5541 (1557 selections), ambiguity1 each. Pegasus logs:
ga13-{cuda,amdgpu}-env/shared-0f2-{auto,portable}.log. Clean GA c8dd580 /
oneAPI26fb776 forced-portable shared four-type run passes5517 (1560 selections),
ambiguity1; Taurus ga13-oneapi-env/portable-final.log. These supplement the native
backend suites and oneAPI12904+24 pre-existing broken checks recorded below.

New deprecated CSR/CSC/COO dispatch names (Astra16/Sol31) alias canonical containers,
not abstract extension points. BSR/device aliases and dense_array_type remain removed.
CPU precision-policy and initialized dense-copy fixtures pass:858 on Julia1.10/1.13;
clean frozen781a0b2 actual Julia1.15 Pkg.test passes859 including ambiguity. Log Pegasus
ga13-nightly-final-env/nightly-final.log. History-cleaned equivalent is3ceeb6e.
Julia1.10 docs makedocs also passes locally (ga13-docs-build-only110.log); merely
including docs/make.jl in interactive jld does NOT call main. No deployment performed.
See validation/REPORT.sparse-v13-downstream.md for exact consumer audit revisions;
MLDataDevices, QuantumToolboxCore and SparseMatrixColorings still need coordinated
compatibility/API releases. No whole downstream-package pass claimed.
CUDAcb5389b13 / AMD8c0ff4687 remove four no-longer-included legacy integration files;
their history remains recoverable and the active centralized methods are unchanged.

OpenCL Pegasus clean781a0b2/88d88e8: PoCL CPU forced-portable four-type full shared
suite5544 passes (1554 selections), ambiguity1; forced OpenCL-C contracts112+ambiguity1
pass. NVIDIA RTX5080 fresh contracts112+ambiguity1 pass, but broader sparse run fails
first at native SVM upload(-30); separately the requested Float16 is unsupported there.
The driver now records resolved IL/C route, capability-tests Float16 rejection when
absent and omits it only from default types; explicit type selections remain honored.
No complete NVIDIA sparse suite pass. Logs Pegasus ga13-opencl-final-env.

Minimizer ga13-opencl-dense-triplet-churn.jl uses ONLY dense CLArray buffers and CPU
SparseArrays reconstruction: current stack fails872 or4 checks in fresh contexts;
matched minimal v13 environment also fails952. Same-host OpenCL main9c27048 with
registered GA12.1.1 passes960 twice. Other simple dense/GC probes pass. This narrows
the failing stack but does NOT establish GPUArrays or driver causality. GPUToolbox
3.4.1 matches both Pegasus envs; AK/Atomix src/ext match their registered baseline
trees exactly. Default preferences match. Guarded noinline uploader passes once;
Native-blocking and ordinary-Ptr-return diagnostics also fail. Do not enable GC-off or
blocking synchronization as a production fix without causal evidence.

Subsequent controls supersede that baseline correlation: fresh same-host OpenCL
main9c27048 / registered GA12.1.1 also fails10+1 on this minimizer, first native
SVM memcpy CL_INVALID_VALUE(-30), log ga13-opencl-baseline-env/baseline-long.log.
Classify the intermittent allocation/transfer failure as reproduced on baseline
and v13, cause unresolved; this does not establish a driver or v13 regression.
The ordinary-Ptr native-return diagnostic also fails824+1, so changing the custom
CLPtr return type is not a sufficient fix. A same-call native C shim confirms actual
NULL on the baseline, with the native out-parameter matching Julia's return. With
OCL_ICD_FILENAMES=libnvidia-opencl.so.1,344 checks pass before size36/alignment4
returns NULL: alloc/fail/free=(516,1,237) at failure and (516,1,514) after unwind.
That environment variable adds ICDs, not exclusive discovery. The allocating task's
finalizer-inhibition counter was0 on the default pool. No fixed resource cap is
established. See validation/REPORT.opencl-svm.md and its dense-only reproducer.

Independent OpenCL029274e is pushed: correctly detects CL_NULL (not cross-address-
space C_NULL), retries through existing bounded reclamation, preserves the Context
across GC and throws before retaining its native context or publishing storage.
Mocked retry/error/argument checks and actual native context-reference checks pass15.
The new error reports native NULL, not an unproven OOM cause. This does NOT fix the
underlying intermittent native failure. Astra20/Sol43 reviewed the focused patch;
post-initialization cleanup was deliberately not added because failed completion
cannot justify freeing pending memory. Existing Intel initialization cleanup is a
separate audit. Actual clean root Pkg.test with GAef7659a/OpenCL029274e crashed
before tests in versioninfo, mixing JLL/system PoCL libraries: the runner imported
pocl_jll even for --platform=cuda. Independent runner fix6ba2cf4 conditionally
imports CPU drivers in all three runner contexts, leaving target/test selection
unchanged. The matching GA setup fix remains uncommitted in the single CI layer.
Actual root GAef7659a/OpenCL6ba2cf4 on both hosts now passes the15 SVM checks and is
running the entire shared sparse group, one worker. Logs on both hosts:
ga13-opencl-svm-env/actual-driver-entry-full.log. Complete result pending.

Upstream Metal/OpenCL main advanced during validation. Rebased/pushed Metal
1aaa6534c9770c28403a46d8c4006cfbaac5edb4 onto e0dd041 (cooperative execution test
hardening) and OpenCL e0a09f4994faedfaa1606cb9e4f5ff4913e4da41 onto c0ee73d
(versioninfo subgroup-ballot display). Metal's four CI iterations are consolidated
into one isolated CI commit. Previous tips89143a6 and6ba2cf4 remain local/remote
backup/sparse-v13-before-main-refresh-20261021; pushes used explicit leases/refspecs.
Runtime sparse implementations are unchanged by these upstream deltas. The current
root runs remain tied to frozen6ba2cf4, not the rebased OpenCL SHA. A fresh Metal
53-check spot run is pending, --jobs=1 and explicit --threads=1; no full local suite.
All other upstream main refs were rechecked and unchanged.

Astra17/Sol33 independently confirmed an existing OpenCL callback lifetime defect:
both Context constructors drop runtime Base.CFunction handles. Wrapper-only rooting
would not cover native context lifetime, and foreign-thread callbacks need an audit.
Keep this separate: not shown to cause the upload failure, not patched or posted.

GitHub reports a PR/Actions/webhook delay incident starting14:52UTC Oct21; PR heads
lagged verified git refs. GHA39625043230 documentation passes, actual checkout
32c670f409c65fd095df6f0cdff9a541cc3ac788 (synthetic merge of main6fa6490 and
topic ef7659a), not the topic SHA alone. Logs verify exact AK/Atomix pins, actual
doctests and CheckDocument; preview deployment declined. Windows Julia1.12 also
passes35416 actual assertions on that same merge checkout, with exact AK/Atomix
pins verified in ga13-ci-windows112-ef7659a.log. Other jobs are queued or running,
no whole-stack CI pass. Buildkite2082 on ef7659a was auto-canceled with
build_skipping after delayed older webhooks;2083 on old781a0b2 says passed but ALL
backend jobs are broken/never executed. This is NOT backend validation.
Inspect exact head SHAs/logs after recovery; earlier Ubuntu1.13 35017 and Metal53 spot
passes retain their recorded older revisions. Do not cancel unrelated busy GPU jobs.
Official oneAPI support JLL still lacks the native exception-boundary change: a new
artifact plus coordinated releases remain mandatory. User now authorizes updating
existing PR descriptions ONCE EVERYTHING IS DONE, clearly and extremely succinctly;
draft descriptions for new migration PRs too. Do not post premature completion claims.
No PR descriptions/discussion text have been changed yet. Succinct drafts for GA,
all five backends, AK and Atomix are /Users/tim/Julia/tmp/sparse-v13-*-description.md;
Sol57 identified and corrected an overclaim about cuSPARSE direct publication:
it avoids redundant canonical reassembly, not vendor candidate expansion.

## Latest checkpoint (Oct21, policy ownership and final-stack checks)

This section supersedes older chronological checkpoints below. Latest pushed GPUArrays
is1234787 on both topic/PR branches; production0f23249 is committed locally and awaits the
coordinated dependency-pin push. CUDAe37f85d19 and AMDe3a3a604c are pushed to tb/sparse-v13;
Metalb8dc6bf1f is pushed to both branches and bootstraps GA1234787. oneAPI26fb776,
OpenCL9526a11, AKa915b41 and Atomix1ffd4ae unchanged. PR745 targets main and is detached
from its old stack; the other three PRs' bases and heads were verified unchanged.

Production0f23249 moves dense sparse(A) policy fully into GPUArrays. Two scoped helpers
preserve CUDA/AMD uniform Cint defaults and existing BSR constructor/block-size behavior;
explicit canonical constructors and coordinate assembly retain their width rules. Removed
backend public overrides in CUDAbcf4a222a/AMDc5ae5ef37. Astra turn14/Sol turn24 approve.
Shared tests now check the declared default including pointer width; independent backend
tests require Cint and BSR parity, invalid-format tests use unknown_format. No tests skipped.
Candidate physical conversions pass600 shared+8 findnz each on CUDA/AMD, plus each native
conversion suite and12 dedicated policy checks. JL reference208 checks pass. These are
candidate-worktree results, not yet clean final-commit full shared runs.

Independent explicit SparseMatricesCSR upload fix in CUDAe37f85d19/AMDe3a3a604c:
typed CSC/COO now construct the requested uniform-width CSR directly instead of first
narrowing through the backend's default Cint adaptor. Old CUDA8a9379519 red repro rejects
explicit Int64 COO with dimensions(1,2147483648) despite one stored entry. New physical
12 regressions each pass, plus CUDA48/AMD53 existing extension checks. No giant allocation.

Clean frozen GA c8dd580 / oneAPI26fb776 Pkg.test on Taurus ArcA770 passes12904,
with24 pre-existing broken CSC triangular checks: shared sparse11503, resources122,
oneMKL1279. Custom support path verified and native exception containment included.
AKa915b41 and Atomix1ffd4ae source trees verified clean. Official support artifact still
lacks the C++ catch fix, so this remains stack validation, not normal release resolution.
Log Taurus: ga13-oneapi-env/frozen-root.log.

Clean frozen GA c8dd580 / AMD08277ef7c actual Pkg.test on Pegasus passes all selected
native sparse and dense solver groups: sparse_v13 208, solver1089, interfaces625,
generic596, preconditioners312, broadcast65, linalg168, SparseMatricesCSR53,
conversions34, device15. Log ga13-amdgpu-env/frozen-native.log. This precedes the latest
policy/upload commits; final rerun remains required. CUDA clean c8dd580/8a9379519 shared
six-type runs expose ONLY24 dense policy expectation failures: automatic7652 passing,
forced-portable7653 passing. The policy correction above addresses those exact failures;
do not call those whole runs passing.

TEMP CI1234787 (reviewed by Sol turn23) resolves each actual CUDA library test project
with the same pins; Julia<1.12's generated root-test Manifest is moved into scratch before
Pkg.test because old Pkg unconditionally rejects duplicate developed UUIDs even at the
same path. Library manifests stay for runner project switches. Actual Julia1.10.12 CUDA
entrypoint SparseMatricesCSR test passes48, log ga13-cuda-ci-entry/entry110-libs2.log.
Self-backend checkouts are cloned clean into disposable scratch (no stale manifests or
source worktree mutations). oneAPI source build is unconditional build_local, with supplied
preferences retained through the build and propagated to tests. Metal own spot tests53
pass one-worker, log /Users/tim/Julia/tmp/ga13-metal-spot-clean.log; daemon stopped.
Local Metal testing must remain focused and single-worker per user's request.

GHA39608465677 / Buildkite2067 test exact GA1234787, not local0f23249. At last check all17
GHA jobs queued; Buildkite uploader passed, backend jobs scheduled. Earlier GHA7d2123d
Ubuntu1.13 passed35017. New pins/commits will need revision verification and real logs.
GPU agents are online but occupied; do not cancel unrelated jobs. Other backend self-CI
still needs coordinated bootstrap wiring; Metal is wired. No CI stack pass claimed.

Performance checkpoint0cc9461 commits the harness,115 native-case comparisons, raw data,
and observed Nsight device memory. Exact GA c8dd580 / CUDA8a9379519 native SpGEMM is
within1% of old CUDA44ac4ebb3, same203605548-byte incremental peak. See perf/REPORT.sparse-v13.md.

Fresh NVIDIA OpenCL Taurus reproduction: frozen GA c8dd580/OpenCL9526a11, RTX6000Ada,
GPUToolbox3.4.2. Normal GC/cooperative synchronization gives82 passing/3 errors; first
-9999 occurs after tiny fused CSR broadcast, then context is poisoned. Same full contracts
with GC disabled pass112 + ambiguity1; blocking preference (normal GC) also passes112+1.
Isolated tiny broadcast with explicit synchronization+GC passes3 repetitions. These variants
are diagnostics, not acceptable test skips/fixes. Sol turn25 found no confirmed missing
kernel root; cooperative finalizer marker path and timing need dense-only minimization
against OpenCL main9c27048. Taurus logs ga13-opencl-frozen-env/{contracts-nvidia4.log,
contracts-nvidia-nogc.log,min-csr32.log}; blocking variant ga13-opencl-blocking-env.
Dense pending-input-GC minimizer ga13-opencl-dense-gc.jl is running; baseline environment
ga13-opencl-baseline-env and clean main source opencl-v13-main-baseline are prepared.
Do not attribute the failure to sparse code or treat PoCL CPU as GPU coverage.

Validation driver validation/sparse-v13-backend.jl is committed in5c477a3.
It supports full automatic/portable shared runs and fixed contract fixtures, two public
import scenarios (not impossible dependency-order reversals), extension assertions,
source revision/dirty metadata, and ambiguity filtering including split extension files.
No permanent global algorithm switch; portable instrumentation is process-local only.
Finish clean exact-revision suites, OpenCL minimization/both-host coverage, oneAPI prepared
resource tests with forced-portable shared products, backend self-CI wiring and CI logs.
Keep the release/artifact checklist precise. No PR discussion text has been posted.

## Current checkpoint (Oct21, after PR retarget)

The sections below are chronological checkpoints; this section supersedes stale state
claims in them. PR745 now targets main and is no longer stacked, with the user's explicit
approval. GitHub only supports unstacking the whole stack: saved stack797 to
`/Users/tim/Julia/tmp/ga13-stack797-before.json`, dissolved it, recreated unchanged
PRs795/796/798 as stack832, and retargeted745. Verified their base relationships and
head SHAs unchanged. No other branches, PR descriptions or discussion comments changed.
PR745 is mergeable. Its previous tip and Metal982's previous tip are preserved in both
local and remote backup refs documented below. Both PR branches have been safely rewritten.

Latest pushed GPUArrays31bd874 is on tb/sparse-v13 and PR branch tb/sparse. Metal f8b9125
is on both corresponding branches. CUDA8a9379519 adds direct publication of completed
canonical SpGEMM output; 168 physical RTX5080 assertions pass across CSR/CSC and
default/ALG1/ALG2/ALG3, including empty products, cancellation, retained-storage GC and
check_structure. Older cuSPARSE library versions retain assembly (not hardware-tested).
CUDA213e11fff and oneAPI26fb776 also remove obsolete TestSuite.sparse_types registration
hooks that prevented the actual root runners from starting. The real focused CUDA Pkg.test
entrypoint passes221 at CUDA213e11fff / GPUArrays31bd874, clean detached checkouts.
Other pushed tips: AMD08277ef7c, OpenCL9526a11, AKa915b41, Atomix1ffd4ae.

GitHub Actions39597816964 tests exact GPUArrays7d2123d, not the newest tips. Ubuntu1.13
passed35017 assertions including dense and sparse. Documentation failed because setup-only
mktempdir dependencies vanished at process exit;31bd874 fixes retention but needs a new
run. Other jobs were running/queued at the last check. Buildkite2065 at31bd874 had only
the uploader passed; all8 backend jobs scheduled. Authenticated `bk api` / `bk build view`
works with pipeline julialang/gpuarrays-dot-jl; public web JSON omits jobs. Obsolete builds
2060..2064 and2058 were automatically canceled. Do not report scheduled jobs as coverage.

Local full Metal Pkg.test (--all,96 groups,5 workers) was interrupted at the user's request
to reduce local load. Source stack: GA c0c2482 / Metal a763c7fb6 / KA0.10.1. Many dense,
native and own sparse tests passed, kernelabstractions failed (details not emitted before
interruption), and the shared sparse group had not finished. Log:
`/Users/tim/Julia/tmp/ga13-metal-stack-full.log`. This is NOT a passing full suite.
Use only focused spot tests locally, one worker (--jobs=1, Julia threads1); leave full
Metal suites in CI. The full daemon and workers are stopped. Prior focused Metal results
remain valid only for their recorded revisions.

Performance harness is currently untracked at perf/sparse-v13-cuda.jl; old/new warm timings
and Nsight peak-memory traces exist in Pegasus ga13-cuda-{baseline,benchmark}-env.
Pre-optimization heavy SpGEMM: old vendor0.589ms, migrated vendor0.853ms, portable7.391ms;
incremental peak device allocation203605548 bytes for both native paths,553648304 portable.
These are exploratory changing-stack results, not the final report. Rebenchmark8a9379519
serially against frozen baseline before recording conclusions. Record cumulative GPU
allocation separately from observed live peak. New dependency pins must be pushed and
their exact tested revisions verified. Official oneAPI support still needs a new artifact.

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

## CI and parity checkpoint (Oct21, after first coordinated push)

GPUArrays PR745 is now safely rewritten onto current main history: old tip2a50d28
is preserved remotely as backup/sparse-v13-pr745-20261021. Both tb/sparse and
tb/sparse-v13 include temporary exact-stack CI commit7d2123d. The PR itself still
targets tb/permutedims-empty and is conflicted: `gh pr edit --base main` fails
with "Cannot change the base branch because the pull request is part of a stack."
An asynchronous question asks the user to detach it or retain stacking. Do not
modify the old base branch. No PR description/comment was posted or edited.

Direct workflow dispatch avoids the conflicted merge ref: GitHub run39597816964
tests exact GPUArrays7d2123d. All17 jobs were queued at the last check. Buildkite
2058 also tests7d2123d, with five backend steps scheduled (nine jobs including
matrix expansion). Public build JSON statistics work, but its jobs/steps arrays
are empty; retrieve actual job logs before accepting results. These are NOT
passing CI results. Backend pins in .buildkite/sparse-v13-stack.toml are exact;
follow-ups must update pins and verify tested revisions. The helper performs
named Pkg.test from a separate environment, explicitly develops CUDA's complete
split workspace (including test-only libraries), runs --all library tests and
asserts runtime/vendor functionality. oneAPI builds support in an isolated module
and task-owned scratch with cleaned ARGS, propagates the artifact preference and
tests the native exception boundary only after verifying the expected library.

Latest pushed production backend tips at this checkpoint: CUDAcdca56af1,
AMDGPU08277ef7c, oneAPI01b71ad (Metal/OpenCL/AK/Atomix unchanged). AMD actual suite8
and suite9 completed successfully, including43 preflight/empty checks,165 forced
rectangular products, all native generic/solver groups, preconditioners24,
triangular288, interfaces620, SparseArrays5, device15 and SparseMatricesCSR53.
This is changing-worktree evidence, not final fresh-checkout commit validation.

CUDA suites through conversions, broadcast, operations, reductions and linalg
passed. interfaces/mul then found missing canonical +/- Symmetric/Hermitian
materialization; centrally fixed, focused601 assertions + sparse ambiguity check
pass. cuda-rest17.log is the current fresh physical rerun. Earlier tests also
found missing generalized-dot dimensions (silently accepted trailing x entries),
now fixed both in sparse dot(x,A,y) and generic dense dot. Typed conversion now
checks requested widths/values before regrouping or wrapper materialization;
the huge-COO narrowing regression previously attempted32GiB allocation.
Diagonal extraction handles transposes, narrow offsets and extreme empty results;
explicit host Array(Diagonal(GPUVector)) transfers only its diagonal first.
495 focused + all JLArray linalg/diagonal groups passed that independent fix.

Full JLArrays shared sparse default-type suite completed in
ga13-parity-shared-final.log before the latest compatibility additions. Full
Julia1.10 CPU+dense+sparse Pkg.test is currently running under ga13-ci-stack110b
from the new explicit pinned setup; log /Users/tim/Julia/tmp/ga13-ci-stack110b.log.
Its first run exposed a real Test import omission at the runner entrypoint,
fixed separately in ebda95b. The initial setup experiment rewrote an ignored
root manifest with Julia1.10 stdlibs; it was moved recoverably to
/Users/tim/Julia/tmp/ga13-root-manifest-ci110.saved.toml and fresh1.13 strict
precompilation passes. Production Project.toml was restored; the final helper no
longer develops into it. Do not confuse dependency setup with completed tests.

Sol review through turn20: dense diagonal transfer20 local checks pass; wrapper
preflight, empty-BSR guards, CI functionality checks and build namespace fixes
verified. Source claims about a missing AMD logical-dimension guard were checked
and withdrawn: require_cint_storage already checks it. Astra turn12 approves the
explicit scale-and-promote numerical policy with documented scope/limitations.

Still required: finish CUDA interface/solver parity, final exact-revision hardware
runs (including portable-only shared products), fresh Metal, OpenCL GPU sequence
minimization and both-host coverage, load-order/ambiguities, performance/peak-temp
benchmarks, actual CI logs, backend self-CI wiring/release coordination, and Metal
PR982 update. Official oneAPI support artifact still lacks the C++ exception fix;
new artifact release and normal registered dependency resolution are release gates.
Local draft PR text: /Users/tim/Julia/tmp/sparse-v13-pr-description.md.
