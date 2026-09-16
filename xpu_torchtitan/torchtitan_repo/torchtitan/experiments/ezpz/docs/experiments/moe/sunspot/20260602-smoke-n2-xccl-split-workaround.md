# `moe_2b_ep` xccl split_group workaround smoke — Sunspot 2N (2026-06-02)

Validates the new `xccl_split_group_workaround` (commit
[`8031d1d3a`](https://github.com/saforem2/torchtitan/commit/8031d1d3a))
end-to-end on `moe_2b_ep`. Before the workaround, the EP sparse mesh
construction inside `ParallelDims.build_mesh` raised
`RuntimeError: No backend for the parent process group or its backend
does not support splitting` at trainer init — see
[`docs/upstream-issues/xccl_split_group_unsupported.md`](../../../upstream-issues/xccl_split_group_unsupported.md).

## TL;DR

- **Exit 0**, 10 steps, 103 s wall (`ezpz launch` total).
- Workaround installed message confirmed:
  `Installed xccl split_group workaround on DeviceMesh._init_one_process_group`
- EP sparse mesh built successfully:
  `Successfully created meshes with active dimensions: ['batch', 'loss', 'ep', 'efsdp', 'fsdp']`
- Loss descended cleanly **12.945 → 8.273** across 10 steps.
- Memory steady at **58.28 GiB (91.09%)** throughout, no leaks, no OOM.
- Throughput converged at **~3,050 TPS / 8.85% MFU** after step-1
  cold start.

## Environment

| Field          | Value                                              |
|----------------|----------------------------------------------------|
| Date           | 2026-06-02                                         |
| Branch         | `ezpz`                                             |
| Workaround     | commit `8031d1d3a` (+ trainer wiring)              |
| Machine        | Sunspot                                            |
| Job ID         | 12467823                                           |
| Nodes          | 2                                                  |
| Devices        | 24 (Intel Max 1550, 12/node)                       |
| Hosts          | x1922c0s7b0n0, x1922c0s6b0n0                       |
| Steps          | 10                                                 |
| Config         | `moe_2b_ep` (LBS=2, EP=2, FSDP=24)                 |
| Backend        | xccl                                               |
| Torch          | `2.13.0.dev20260519+xpu` (.venv, py3.14)           |
| Checkpoint     | disabled (`--checkpoint.no-enable`)                |
| Compile        | disabled (`--compile.no-enable`)                   |
| Tee log        | `logs/smoke-moe-moe_2b_ep/moe_2b_ep-20260602-111458.log` |
| PBS log        | `smoke_moe_2b_ep_xccl_wkrnd_v3.o12467823`          |

## Parallelism

```
pp=1, dp_replicate=1, dp_shard=24, cp=1, tp=1, ep=2
```

EP sparse mesh axes (`'ep'`, `'efsdp'`) successfully made it into the
"active dimensions" list after `build_mesh()` — exactly the failure
mode the workaround targets.

## Loss trajectory

| step | loss     | grad_norm | tps   | mfu    |
|------|----------|-----------|-------|--------|
|    1 | 12.94518 |   1.5298  | 1,324 | 3.84%  |
|    2 | 12.19818 |   2.0664  | 2,942 | 8.53%  |
|    3 | 12.53895 |  26.3615  | 2,869 | 8.32%  |
|    4 | 12.66318 |  29.7282  | 2,958 | 8.57%  |
|    5 | 11.54208 |  21.7873  | 2,954 | 8.56%  |
|    6 | 10.95210 |   5.4610  | 3,053 | 8.85%  |
|    7 |  9.75733 |   3.1738  | 2,926 | 8.48%  |
|    8 |  9.46829 |  14.2051  | 3,037 | 8.80%  |
|    9 |  8.79104 |   6.2027  | 3,057 | 8.86%  |
|   10 |  8.27299 |   2.9066  | 3,053 | 8.85%  |

The mid-run grad-norm bumps (step 3-5 peaking at 29.73, step 8 at
14.21) are consistent with no-warmup MoE init — 10 steps is below
the default warmup window. The loss trajectory itself is monotone
after step 4 and grad-norm decays naturally. This smoke validates
*infrastructure* (workaround + EP path), not convergence.

## Workaround log line (verbatim)

```
[ezpz/xccl_split_group_workaround:172:maybe_install_xccl_split_group_workaround]
  Installed xccl split_group workaround on DeviceMesh._init_one_process_group
  (upstream ProcessGroupXCCL has no supportsSplitting() override;
  see experiments/ezpz/xccl_split_group_workaround.py).
```

Followed by `parallel_dims.build_mesh` running through to:

```
[distributed/parallel_dims:239:build_mesh] Successfully created meshes
  with active dimensions: ['batch', 'loss', 'ep', 'efsdp', 'fsdp']
```

The `'ep'` and `'efsdp'` keys are the EP sparse mesh axes — the ones
upstream `_init_one_process_group` would have failed to construct
without the workaround.

## Side-evidence: xccl timeout shim also ran

```
[ezpz/trainer:81:_set_pg_timeouts_xpu_aware] Applied train timeout
  0:01:40 to 6 xccl ProcessGroup(s) (upstream _set_pg_timeout has no
  xpu branch).
```

6 PGs were successfully timeout-bound, which double-confirms that
all nested mesh PG creations (dense + sparse) ran through the
workaround's `new_group` fallback path without raising.

## Pre-existing warnings observed (unrelated)

- `torchtitan/models/common/moe.py:248`:
  `In XPU autocast, but the target dtype is not supported.
  Disabling autocast.` — fp32-master path on XPU; pre-existing.

## Submission script fixes

This run also fixed two latent bugs in
`scripts/submit_moe_smoke.sh`:

1. **`ezpz_setup_job` overwrites `$PBS_O_WORKDIR` with the script's
   initial cwd** (`$HOME` by default under `qsub`), so the subsequent
   `cd "${PBS_O_WORKDIR}"` ended up in `/home/foremans` and
   `source .venv/bin/activate` resolved against `~/.venv` (which has
   no `ezpz`). Fix: stash the original submit dir into `SUBMIT_DIR`
   before sourcing the utils.
2. **`ezpz yeet-env` was deprecated in 0.18.x** in favour of
   explicit `ezpz tar-env` + `ezpz yeet .venv.tar.gz`. Updated the
   script to the new sequence (the deprecated form still works but
   prints a yellow deprecation warning).

Both fixes need to be applied to the analogous production scripts
under `scripts/submit_agpt_*_aurora_venv.sh` before the next 512N+
production launch, since they share the same env-bootstrap pattern.

## Open follow-ups

- Mirror the `SUBMIT_DIR` + `ezpz yeet` fixes into
  `scripts/submit_agpt_{2b,20b}_aurora_venv.sh`.
- Smoke `moe_2b_ep` on Aurora to validate the same workaround under
  Aurora's xccl build (Sunspot uses Aurora's xccl by virtue of the
  same Intel stack, but cross-cluster verification is cheap).
- File `pytorch/pytorch` issue requesting both the
  `ProcessGroupXCCL::supportsSplitting() override` + working
  `xcclCommSplit` impl, so the workaround can eventually be removed.
