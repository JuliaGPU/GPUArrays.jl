# GPUArrays v13 implementation handoff

## Current state

Implementation is in `/Users/tim/Julia/tmp/gpuarrays-v13`, branch `tb/sparse-v13`.
The original investigation worktree is untouched. Read PLAN.sparse-v13.md and
REPORT.sparse-v13.md for the agreed architecture and release gates.

Fetched GPUArrays origin on 2026-10-21: main `6fa6490`, PR #745 `2a50d28`.
Local backup `backup/sparse-v13-pr745-20261021` preserves the old PR tip.
Rebased its ten contributor commits onto main, retaining author metadata.
Resolved conflicts by keeping main's dense implementation, memory-location
alias fix, closure-box check, and the new sparse suite. The obsolete
`coo_type` regression no longer applies after removing that interface.
GPUArrays is 13.0.0 and JLArrays 0.5.0. Removed the PR's obsolete JLArrays
algorithm shims in favor of main's existing reference implementations.

## Reviews and validation

Architecture discussion: `peer log sparse-v13-architecture` (Astra).
Baseline daemon: `ga13-baseline`, project `test`, no Revise, idle timeout 2h.
No production tests, backend migrations, pushes or CI results yet.

## Remaining work

All implementation packages in PLAN.sparse-v13.md remain open beyond the
initial rebase. In particular the five-parameter compressed representation,
mutation preflight, semantic regressions, selector, vendor extensions and
backend parity must be implemented and tested before claiming migration.
Preserve remote topic tips and check leases before updating PR branches.
Do not post GitHub comments or edit PR descriptions; draft text locally.
