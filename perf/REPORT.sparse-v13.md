# Sparse v13 performance checkpoint

Measured Oct21,2026 on Pegasus, RTX5080 (16GB), Ryzen9950X, Julia1.13.2,
CUDA runtime13.4.0, driver615.78.8, cuSPARSE12.8.6. This is a CUDA performance
checkpoint, not evidence for other hardware, Julia versions or older cuSPARSE.

Baseline: clean CUDA44ac4ebb3794013be79b607aec0c8aee4321d71d (6.4.2), registry
GPUArrays12.1.1 (tree f72b034a94bf1578dcc6e5410990f257b2091ada).
Migrated: clean CUDA8a9379519f03586ca41d1a472d0ddf63eb9ba7be (7.0.0),
GPUArraysc8dd580afebcebf86d87b01243f9bb773176c371 (13.0.0),
AKa915b419728e3abd258bce78ea3d1d3255b88ed7, Atomix1ffd4ae4746fbc9463882c68563c6bdb81e0d9c0,
KA0.10.1. Both use the same installed CUDA libraries. Baseline and migrated grids
ran serially, without other test activity on the GPU. The harness checks every
result against CPU sparse mathematics. Raw measurements are in results/*.tsv.

The grid includes diagonal, nine-diagonal banded, uniform-random, skewed and
empty-row matrices; HB1138_bus and bcsstk17; CSR/CSC; normal/transpose SpMV;
SpMM widths1/8/64; complex adjoints; duplicate assembly; and heavy-candidate SpGEMM.
Three warmups precede eleven synchronized samples; tables use warm medians.
`first_seconds` is the first call for that case in that process, not a fully cold
package/compiler measurement: earlier cases can already have compiled its methods.
Automatic precedes forced vendor and portable calls; do not compare their first
times as independent cold starts. CPU references and result downloads are outside
timing. Host and GPU cumulative requested allocations are recorded separately.

## Results

Times below are microseconds. Across115 matched native cases, migrated/old warm
time has median1.029 and maximum1.171 (uniform CSR transposed SpMV,16.50→19.33µs).
The small-operation validation/selection cost is visible; this is not a claim that
every case is faster. No hardware-specific automatic-selection threshold was added.

| Case | Old vendor | Migrated vendor | Forced portable |
|---|---:|---:|---:|
| bcsstk17 CSR SpMV | 15.52 | 16.37 | 26.69 |
| bcsstk17 CSC SpMV | 17.00 | 17.40 | 271.62 |
| bcsstk17 CSR SpMM width64 | 86.64 | 85.57 | 710.97 |
| bcsstk17 CSC SpMM width64 | 152.82 | 146.07 | 953.48 |
| Duplicate-heavy CSR assembly | 1292.82 | 199.90 | same central assembly |
| Candidate-heavy CSR SpGEMM | 578.21 | 583.60 | 7369.04 |

SpGEMM has8388608 candidate products but262144 final entries. Before CUDA8a9379519,
redundant canonical reassembly cost about0.85ms versus0.59ms baseline. Direct
publication after the vendor's completed canonical output restores parity without
weakening raw constructors or other vendor-output normalization. All four native
algorithms have dedicated canonicality/alias/empty/cancellation/GC tests. The older
library branch still assembles and has not been performance-tested on old hardware.
Automatic selection retains the supported native path. Portable transposed/CSC
regrouping and candidate expansion remain explicit optimization opportunities.

## Observed device memory

Separate Nsight system-wide CUDA/NVTX traces recorded device allocation/free events
inside one warmed, pool-purged call per range. These are observed incremental live
device peaks, including the new result, relative to range entry; not process RSS,
not solely temporary workspace, and not cumulative requested allocation. Inputs
and preallocated dense outputs already exist at range entry. They do not establish
a universal bound on driver-internal or future allocator behavior.

| Case | Old peak bytes | Migrated vendor peak bytes | Portable peak bytes |
|---|---:|---:|---:|
| Banded CSR SpMM width8 | 1416 | 1416 | 0 |
| bcsstk17 CSC SpMV | 680 | 680 | 21861168 |
| Duplicate CSR assembly | 2723912 | 4320016 | same central assembly |
| Candidate-heavy CSR SpGEMM | 203605548 | 203605548 | 553648304 |

SpGEMM cumulative GPU requests are203803492 bytes for either native stack and
569723060 portable. Portable assembly is much faster in the duplicate-heavy case
but has a larger measured peak. The portable1MiB candidate-estimate limit rejects
the heavy product before candidate materialization. This is an explicit estimate
gate, not a promise that arbitrary vendor workspace obeys that limit.

## Reproduction

Use separate task-owned environments under ~/Julia/tmp for the old and new stacks.
Develop the CUDA repository's CUDACore/CUDATools and library workspace packages
explicitly; use the exact revisions above, and develop GPUArrays plus its core,
AK and Atomix for the migrated environment. Add NVTX, SHA, Test, Statistics and
the script's standard-library imports as direct dependencies. Do not activate the
default environment. Preserve each resolved Manifest with the run logs.

Cache these NIST copies of the SuiteSparse Harwell-Boeing workloads:

- https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/psadmit/1138_bus.mtx.gz
  SHA2560544e52ed96893ae82c5a7f9051d662e04a1c27fe57768578de09932381fd963
- https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/bcsstruc2/bcsstk17.mtx.gz
  SHA256105425dc922f5aaa4f2a7b52d707e946cbc3cc309accea11f5365a30528f49a5

Set SPARSE_BENCH_DATA to that cache. From each environment run:

```sh
jld --name=unique-sparse-bench --idle-timeout=2h --no-revise --timeout=1800 run /path/to/perf/sparse-v13-cuda.jl
```

For memory profiling, start Nsight in system-wide mode before starting a fresh
daemon, then run the harness with CUDA_INJECTION_SHM_ALLOWED=TRUE,
SPARSE_BENCH_TRACE_MEMORY=1, SPARSE_BENCH_REPETITIONS=1 and a relevant
SPARSE_BENCH_FILTER regex. Profiling the jld launcher with fork tracing stalled
locally; the system-wide approach captures the actual daemon process.

```sh
nsys profile --trace=cuda,nvtx --cuda-trace-scope=system-wide --cuda-memory-usage=true --sample=none --cpuctxsw=none --wait=primary --duration=50 --output=memory-final /bin/sleep 50
nsys export --type=sqlite --output=memory-final.sqlite memory-final.nsys-rep
sqlite3 -header memory-final.sqlite < /path/to/perf/sparse-v13-memory.sql
```

No device-allocation events means zero incremental device allocation for that
range (the SQL has no row). Pegasus logs and reports remain under
ga13-cuda-{baseline,benchmark}-env/{benchmark-final-grid.log,memory-final*}.
These measurements do not replace functional backend suites or CI/release gates.
