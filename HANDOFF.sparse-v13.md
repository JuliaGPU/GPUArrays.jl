# GPUArrays v13 implementation handoff

## Current state

Implementation and backend migrations are committed. Final reviewed runtime is
GPUArrays34d125e0ce852c823c8aabcaf0c6bc88424bfe86, retained remotely as
backup/sparse-v13-all-dimensions-runtime-20261021. The final central CI head will
differ only in dependency pins/documentation; verify source equivalence instead
of relabeling tests as literal tests of a different commit.

Production ownership is settled: immutable canonical vector/CSR/CSC/row-major COO
with separate compressed-pointer/coordinate storage; checked widths and mutation;
borrowed raw buffers; ordered associative duplicates; centralized portable/vendor
selection and high-level vendor integration. CUDA/AMD native BSR, batch, solve,
factorization and preconditioner paths remain. oneAPI resources own context/queue/
buffers and prepared analysis outside canonical containers. No scratch prototype
was merged. See PLAN.sparse-v13.md, docs/src/interface.md and the implementation.

All ordinary broadcast dimensions now retain sparse validation: supported dense
vector/matrix expansion works; mixed sparse ranks, sparse expansion, GPU0 operands,
and higher-dimensional output explicitly reject before allocation/scalar capture.
Host-backed sparse inputs previously escaped into CPU Array output for 3D fusion;
this was reproduced and fixed. Negative-order norms include logical implicit zeros
on every Julia version, preserving NaN magnitudes. Nonfinite complex references use
scalar magnitudes rather than BLAS-dependent Inf/NaN handling.

## Branches and CI pin graph

All branches are topic branches, never main. Final intended pins:

| Repository | Branch | Tip |
| --- | --- | --- |
| GPUArrays | tb/sparse and tb/sparse-v13 | final CI head; runtime34d125e |
| CUDA | tb/sparse-v13 | a7336fd03673a9978f6cdd3bb2838da8801cac37 |
| AMDGPU | tb/sparse-v13 | b518d77d41d35f916fc378cd02f0a22f698b8875 |
| oneAPI | tb/sparse-v13 | 06f2f77021c105ab27c6c5e35af705426aaad730 |
| Metal | tb/sparse and tb/sparse-v13 | 4e837cb7db64ebd7315369a1521226f45eda80b3 |
| OpenCL | tb/sparse-v13 | 94fae12aa6b88e3e3a39f7f5933ebb66fa9e6d29 |
| AcceleratedKernels | tb/sparse-v13 | a915b419728e3abd258bce78ea3d1d3255b88ed7 |
| Atomix | tb/sparse-v13 | 1ffd4ae4746fbc9463882c68563c6bdb81e0d9c0 |
| maleadt/Yggdrasil | tb/oneapi-sparse-v13-support | 31c79455bc9f0b415a75ceb9faac59a26148e8f4 |

Final backend changes above only repin their TEMP bootstrap to frozen34d125e;
runtime/tests equal CUDAa3e0c60, AMDf5b6976, oneAPId1cb45d, Metal1aaa653 and
OpenCLe0a09f4 respectively. Central CI pins the resulting backend tips. This is a
finite graph: stop repinning for documentation/CI-only changes. All old tips are
preserved locally/remotely; rewrites check concurrent changes and use explicit
refspecs/leases. Useful contributor commits/authorship remain. Each repository has
ONE isolated TEMP CI layer, not an incremental debug-commit series.

Exact setup includes all14 CUDA workspace packages and library test projects;
Julia1.10 uses explicit Pkg.develop because it ignores [sources]. Existing backend
CI variants/preferences remain. Installation and nightly failures are hard failures.
Custom min-dependency/Enzyme variants retain their selected dependency graphs.
The oneAPI CI override builds and verifies native support from source; this is not
published-artifact resolution. Remove temporary plumbing after coordinated releases.

## Exact validation evidence

Do not equate source equivalence, setup success, a focused test, and a full migration
suite. Older exact revisions remain useful evidence, not literal final-head passes.

- Final34d125e local Julia1.13.2:2058 regressions,247 shared contracts,52 display,
  195 linalg,275 mixed broadcast,ambiguity1 pass. Logs
  /Users/tim/Julia/tmp/ga13-broadcast-rank-green2.log. First added destination checks
  expected ArgumentError instead of the existing pre-mutation DimensionMismatch;
  corrected the test expectation and reran green. Initial 3D CPU fallback and GPU0
  wrong-route reproductions remain red evidence.
- Final34d125e Julia1.10.12:2058regressions,52display,195linalg,275broadcast and
  ambiguity1 pass, local ga13-final34d110-focused.log. Julia1.15-DEV61 actual Pkg.test
  passes2058regressions+ambiguity1, Pegasus ga13-nightly-final-env/final34d-regressions.log.
  CUDAa3e0c60 and AMDf5b6976 final shared contracts244+ambiguity1 pass each,
  Pegasus ga13-{cuda,amdgpu-loadorder-final}-env/final34d-contracts.log. Final
  oneAPI/OpenCL/Metal guard spot checks remain pending separately.
- GA4e20de2/CUDAa3e0c60 and AMDf5b6976:181contracts+ambiguity1 each pass,
  GPUArrays-first; GA64ea2d8/A3 focused151contracts+195linalg+275broadcast+ambiguity1.
  Frozen GA0f2d80f/CUDAe37f85d full shared six-type automatic7712/portable7713
  (2004 portable selections); GA0f2d80f/AMDe3a3a604 shared four-type automatic5540/
  portable5541 (1557 selections), ambiguity1 each. Earlier actual native CUDA/AMD
  sparse/solver/factorization/BSR/batch suites pass at revisions in HISTORY.
- GA4e20de2/oneAPId1cb45d, actual generated oneAPI_Support0.11.1+0 artifact on Arc A770:
  Julia1.13.2 checked Pkg.test passes13167+24pre-existing broken (shared11671,
  native1279,resources121,ABI96). Julia1.10.12 native/resources/ABI passes1496+24broken;
  identity/marker/device checks3 each. Loaded artifact tree876c51345ae21089df99bd35dfc29cf292c2a4ce,
  support SHA256d5e977efdfe577477c74827cd9b827b61d14f49fa2c8466fb3387a3e3c933fd7.
  No source-build preference. Taurus ga13-oneapi-generated-jll{,110}-env/test.log.
  See validation/REPORT.oneapi-support.md for recipe, tarball and audit limitations.
- Separately source-built oneAPId1cb45d/GA64ea2d8 checked Pkg.test passes13138+24broken,
  including resources122 because an override-path check was enabled. Loaded support
  SHA256c19915c2965bc4f7153e4c16d240deb107ecfe0a902f07936445d5f9969932c9.
  Earlier oneAPI forced-portable shared suite5517/1560selections+ambiguity1 at
  GAc8dd580/oneAPI26fb776. These are separate from generated-artifact evidence.
- GA4e20de2/OpenCLe0a09f4 actual root sparse/SVM11789 pass on Pegasus RTX5080
  (shared11774+SVM15). Prior GA64ea2d8/E0a root sparse/SVM11759 passes on EACH
  Pegasus RTX5080 and Taurus RTX6000Ada, all ten supported types, normal GC/sync.
  Strict NVIDIA-only discovery uses an empty OCL_ICD_VENDORS directory plus native
  OCL_ICD_FILENAMES; actual enumeration is recorded. Logs both hosts
  ga13-opencl-svm-env/actual-final-entry-full.log; latest Pegasus final4e-entry-full.log.
  GA64edd6d/E0a actual final helper native array/memory/context/queue/event/sync155
  passes, final-native64.log. Not whole OpenCL suites. PoCL CPU IL/C portable tests
  remain separate, not GPU correctness evidence.
- GA4e20de2/Metal1aaa653 actual root spot53 passes, one worker and explicit
  --threads=1; /Users/tim/Julia/tmp/ga13-metal-final4e-spot.log. Daemon stopped.
  No full local Metal and no physical final34d guard check claimed yet.
- GA64ea2d8 Julia1.15-DEV61 full shared JL sparse14957+ambiguity1 passes. Corrected
  nonfinite reference at4083b78 actual nightly Pkg.test1894+ambiguity1 passes.
  Earlier full Julia1.10/1.13 dense+sparse runs and native test groups are in HISTORY.
- CUDA-only benchmark115 native cases: median migrated/baseline ratio1.029,
  max1.171; measured SpGEMM578.21→583.60us, portable7369.04us; native device peak
  203605548 bytes both, portable553648304. Duplicate assembly1292.82→199.90us,
  peak2.7→4.3MB. Exact revisions, methodology and limits: perf/REPORT.sparse-v13.md.
  No performance claim for older cuSPARSE branch or other hardware.

## Remaining gates

1. Finish final exact-stack GitHub/Buildkite CI and inspect actual checkout and
   STACK_REVISION lines, not PR head alone or a green pipeline-upload step. Current
   older099665d run39643990450 and Buildkite2089 predate the all-dimensional guard;
   new final-pin runs are required. All queried matching GPU agents were busy on
   unrelated jobs; do not cancel them. GHA macOS remains heavily queued.
   Earlier ef7659a merge32c670f CI passed Ubuntu/Windows1.10–1.13 and docs, but both
   nightlies failed21 (18 logical norms,3 CPU display references), fixed and tested
   locally afterward. OpenCL hit60min while progressing; final unshortened IL/C
   suite has180min. Never count canceled/broken jobs as a suite pass.
2. Resolve or explicitly review inherited BinaryBuilder audit-tool failures before
   publishing support0.11.1: old binutils2.25 objdump aborts and standalone MKL probes
   warn on baseline and patched builds. Normal generated-JLL native/resource checks
   pass separately; interrupted audits did not pass. Test the published artifact,
   then raise oneAPI's minimum to the actual registered version; retain marker2 gate.
3. Release compatibility-only AK0.5.9/Atomix1.6.1, GA13/JLArrays0.5 (Core0.2.1), then
   coordinated CUDA7(all14), AMDGPU3, oneAPI3, OpenCL0.11, Metal1.12 as weak-dependency
   ordering permits. Test every package/test project, especially Julia1.10.
   Remove TEMP pins/bootstrap only once normal registered Pkg.test/docs resolve.
4. Complete downstream compatibility/API releases recorded in
   validation/REPORT.sparse-v13-downstream.md (MLDataDevices, QuantumToolboxCore,
   SparseMatrixColorings); no whole-consumer pass is claimed.
5. Keep the inherited NVIDIA dense-only SVM churn issue open. It fails on OpenCL
   main9c27048/registered GA12.1.1 and v13; a native shim confirms NULL, including
   strict NVIDIA-only discovery. Cause unresolved, not established v13 regression
   or driver defect. Independent NULL/error guard passes15; do not mask with GC-off
   or blocking defaults. See validation/REPORT.opencl-svm.md/reproducer.
6. Additional GPU generations, old-cuSPARSE algorithm performance and complete local
   Metal are untested. Follow the user's Metal limit: single-threaded spot tests.

## Continuation and review

Local isolated repositories: /Users/tim/Julia/tmp/{gpuarrays,cuda,amdgpu,oneapi,metal,opencl,ak,atomix}-v13.
Remote scratch: ~/Julia/tmp only; existing ~/Julia/pkg checkouts untouched.
Pegasus direct SSH; Taurus via ssh -J pegasus.maleadt.lan taurus.maleadt.lan.
Use unique jld names, --no-revise, --idle-timeout=2h; clear ARGS before test runners.
Some user/untracked preferences and generated coverage files in Metal were left alone.

Astra decisions through23 and Sol reviews through86 are in peer conversations
sparse-v13-architecture and sparse-v13-review; findings were independently checked.
Historical checkpoints and older exact native-suite evidence were moved intact to
validation/HISTORY.sparse-v13.md. PLAN/REPORT are the design and original investigation,
not implementations to merge.

Existing PR745 and Metal982 are draft, detached onto main, with authorized concise
description updates to perform once independent work/validation is settled. No
GitHub discussion/comment or PR body has been changed yet. New backend/dependency
PR descriptions are local /Users/tim/Julia/tmp/sparse-v13-*-pr-description.md drafts;
do not post new PRs or discussions without authorization.
