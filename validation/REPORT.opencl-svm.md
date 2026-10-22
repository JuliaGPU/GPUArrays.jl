# NVIDIA OpenCL allocation/transfer failure

Status: reproduced on baseline and v13; cause unresolved. This is not established
as a sparse redesign regression or a driver defect. PoCL CPU results do not validate
this GPU path.

`opencl-svm-transfer-churn.jl` uploads three ordinary dense buffers and reconstructs
a CPU SparseMatrixCSC after downloading them. It uses no GPU sparse containers or
kernels. Run through a fresh uniquely named jld daemon with normal GC and the default
synchronization policy. Forty repetitions contain 960 checks; request more with
`OPENCL_TRIPLET_REPEATS`. A passing short run is not sufficient: fresh baseline runs
initially passed 960 twice, then failed after ten checks.

## Same-host baseline control

Pegasus, RTX 5080, NVIDIA 615.78.08, Julia 1.13.2:

- OpenCL main `9c27048a45501f0022dbeff47a24305db33efeb6`.
- Registered GPUArrays 12.1.1, tree `f72b034a94bf1578dcc6e5410990f257b2091ada`.
- GPUToolbox 3.4.1; AK 0.5.8 and Atomix 1.6.0 implementations match the migration
  branches' source/extension trees. Those branches change compatibility metadata.
- Baseline direct run: ten checks then native SVM memcpy `CL_INVALID_VALUE` (-30).
- Baseline C-shim run with `OCL_ICD_FILENAMES=libnvidia-opencl.so.1`: 344 checks,
  then native `clSVMAlloc` returned NULL for size 36, alignment 4. A native out-parameter
  matched the pointer returned to Julia, excluding return-value loss in that call.
  Completed allocation/failure/free counts were (516,1,237) at failure and
  (516,1,514) after unwinding. No fixed live-allocation threshold is established.
- Strict NVIDIA-only discovery (empty OCL_ICD_VENDORS directory plus the native
  NVIDIA ICD filename) also fails after ten checks: native NULL for size44,
  alignment4, counts(15,1,3) at failure and(15,1,11) after unwind. The actual
  enumeration contains only NVIDIA CUDA; baseline-strict-nvidia.log.

Logs and the process-local C instrumentation are under Pegasus
`~/Julia/tmp/ga13-opencl-baseline-env/{baseline-long,baseline-shim,baseline-nvidia-only}.log`
and `~/Julia/tmp/ga13-opencl-alloc-probe.{c,jl,so}`. The ICD filename setting adds an
ICD; it does not promise exclusive system-ICD discovery.

## Independent correction

OpenCL `9e55fc3` (rebased successor of `029274e6b517464084f00248e3df2631191633a5`)
rejects native NULL before
publishing nonempty storage or retaining a native context. The old cross-address-
space comparison with host C_NULL missed CLPtr NULL. The fix uses existing bounded
reclamation, preserves the context across GC and reports NULL without inventing a
precise native error cause. Fifteen focused mocked/native-reference checks pass.
This corrects error handling; it does not resolve the allocation failure.

An ordinary-Ptr return declaration and native blocking-copy diagnostics also fail;
neither is a sufficient fix. The allocating task's finalizer-inhibition counter was
zero on the default pool. GC-off and blocking synchronization passes remain
diagnostics, not production fixes or completed migration results.

## Separate runner initialization failure

Actual Pkg.test on GA `ef7659a` / OpenCL `029274e` crashed before tests in
`versioninfo`: JLL libpocl, system libpocl and the system pthread device module were
mixed while enumerating platforms. The runner imported pocl_jll even for an explicit
vendor request. OpenCL `6ba2cf4` imports CPU drivers only for default/CPU requests;
the target generation and test bodies are unchanged. The corresponding temporary
GPUArrays setup follows the same rule. This removes injected CPU drivers, not all
installed system ICDs.

## Actual migrated GPU suite

Fresh actual root Pkg.test at GPUArrays64ea2d8/OpenCLe0a09f4 passes11,759 checks on
EACH host: Pegasus RTX5080 and Taurus RTX6000Ada. Shared sparse11,744 across all ten
supported types plus15 SVM checks, one worker, normal GC/synchronization, default
program route. Strict ICD discovery enumerates only NVIDIA. Both logs are
~/Julia/tmp/ga13-opencl-svm-env/actual-final-entry-full.log. These are actual migrated
GPU sparse/SVM results, not PoCL CPU or whole-backend CI results. They do not resolve
or conceal the independently reproduced baseline dense-allocation failure above.

The final exact pinned helper at GPUArrays9a9126f/OpenCL94fae12 also passes11,852
on Pegasus: shared11,837 plus15 SVM, including the final all-dimensional broadcast
guards. Clean sources and exact pins are logged in
~/Julia/tmp/ga13-opencl-svm-env/final9a-root.log. This is actual checked Pkg.test
stack validation on a test host, not a hosted-CI or normal registered-resolution pass.

Existing Context callback-handle lifetime and Intel SVM initialization-failure
cleanup need separate audits. Neither has been shown to cause this failure, and
neither was silently changed as part of the NULL guard.
