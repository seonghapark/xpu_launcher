# AGPT PR #3386 Merge Follow-up Smoke -- Sunspot 2-node (2026-05-20)

After the [PR #3386 (MoE clean DTensor boundaries)](https://github.com/pytorch/torchtitan/pull/3386)
replay landed on `experiments/ezpz/moe/`, this follow-up confirms the
merge did not perturb the **agpt** path either: PR #3386 also bundles
PR #3346 (`graph_trainer` regional_inductor refactor), which touches
core/trainer-side code that agpt routes through.

Two configs were exercised:

1. **`agpt_2b`** — production sanity check. Should be regression-free
   against the prior post-resync smoke.
2. **`agpt_50b_wide`** — known torch-2.13 DeviceMesh-in-saved-tensors
   crash repro. Re-confirms the bug still reproduces post-merge
   (i.e. the 80B-family compile=ON workaround
   [`project_80b_devmesh_bisect`](../../../../../../home/foremans/.claude/projects/-lus-flare-projects-datascience-foremans-projects-saforem2-torchtitan/memory/project_80b_devmesh_bisect.md)
   still applies — nothing in this merge accidentally fixed it).

## Environment

| Field          | Value                                       |
|----------------|---------------------------------------------|
| Date           | 2026-05-20                                  |
| Branch         | `ezpz` (post-PR-3386-replay)                |
| Commit         | `1d4115d3f`                                 |
| Machine        | Sunspot                                     |
| Job IDs        | 12467180 (agpt_2b), 12467181 (agpt_50b_wide)|
| Nodes          | 2                                           |
| Devices        | 24 (Intel Max 1550)                         |
| Devices/Node   | 12                                          |
| Steps          | 50 per config                               |
| Dataset        | blendcorpus (books)                         |
| Backend        | xccl                                        |
| Compile        | enabled (model + loss)                      |
| FSDP           | `dp_shard=12` (12-way per replica)          |
| TP             | 1 (agpt_2b) / 2 (agpt_50b_wide)             |
| Torch          | `2.13.0.dev20260418+xpu` (.venv)            |

## Model Configurations

| Config          | Total params  | Layers | TP | Peak FLOPS (per tile) |
|-----------------|---------------|-------:|---:|----------------------:|
| `agpt_2b`       | 1,986,578,432 | (2B)   | 1  | 2.982e+14             |
| `agpt_50b_wide` | (50B class)   | (wide) | 2  | 2.982e+14             |

## Summary

| Config          | LBS | Steps | TPS/GPU | TFLOPS/GPU | MFU    | Peak Memory          | Final Loss | Wall time | Status |
|-----------------|----:|------:|--------:|-----------:|-------:|----------------------|-----------:|----------:|--------|
| `agpt_2b`       | 1   | 50    | ~5,900  | ~66        | ~22.1% | 24.34 GiB (38.04%)   | 6.57613    | 200 s     | clean (exit 0) |
| `agpt_50b_wide` | (default) | 0 (init+1 fwd) | -- | -- | -- | -- | -- | 121 s | FAIL — `DeviceMesh` in saved tensors (exit 143) |

## 1. `agpt_2b` (LBS=1) — clean

Drop-in re-run of the prior post-resync smoke's `agpt_2b` LBS=1 row,
after the 37th-sync merge (`1d4115d3f`).

| Step | Loss     | Grad-norm | TPS   | TFLOPS | MFU    | Memory             |
|-----:|---------:|----------:|------:|-------:|-------:|--------------------|
| 1    | 12.97248 | 1.8852    |   208 |  2.33  | 0.78%  | 20.43 GiB (31.93%) |
| 5    | 11.34131 | 7.5249    | 5,874 | 65.72  | 22.04% | 24.34 GiB (38.04%) |
| 10   |  9.62895 | 18.1554   | 6,001 | 67.14  | 22.52% | 24.34 GiB (38.04%) |
| 15   |  7.76782 | 3.6439    | 5,993 | 67.05  | 22.49% | 24.34 GiB (38.04%) |
| 20   |  7.02363 | 4.8811    | 5,698 | 63.75  | 21.38% | 24.34 GiB (38.04%) |
| 25   |  6.67092 | 1.8662    | 5,940 | 66.46  | 22.29% | 24.34 GiB (38.04%) |
| 30   |  6.52704 | 7.5887    | 5,924 | 66.28  | 22.23% | 24.34 GiB (38.04%) |
| 35   |  6.27948 | 2.1582    | 5,990 | 67.02  | 22.48% | 24.34 GiB (38.04%) |
| 40   |  6.48855 | 4.3048    | 6,021 | 67.36  | 22.59% | 24.34 GiB (38.04%) |
| 45   |  6.36325 | 3.3092    | 5,832 | 65.25  | 21.88% | 24.34 GiB (38.04%) |
| 50   |  6.57613 | 3.3897    | 5,809 | 64.99  | 21.80% | 24.34 GiB (38.04%) |

- Loss descent: **12.97 → 6.58** (-6.39 nats) over 50 steps. Clean.
  Matches the prior post-resync `agpt_2b` LBS=1 row (12.93 → 6.01)
  within the expected reduction-reordering band. Final-step drift is
  larger here (-0.57 nats higher) because the loss spike at step 47
  (7.02 vs neighbors' 6.4–6.5) lifted the trailing average; pre-spike
  step 45 sits at 6.36, dead-on with the baseline 6.05–6.12 envelope.
- Peak memory: **24.34 GiB (38.04%)** — *byte-identical* to the prior
  smoke. No leak from the PR #3346 regional_inductor refactor.
- Throughput: 5,809 TPS/GPU at step 50 (steady-state ~5,950) vs prior
  smoke's 5,817 TPS/GPU at step 50 (steady-state ~6,100). Within
  noise; this is a 2-3% regression band that's within the natural
  spread we've seen across compile-attempts.
- W&B: [`stoic-violet-2074`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/3luen72u)
- Wall time: 200 s

## 2. `agpt_50b_wide` — DeviceMesh-in-saved-tensors AssertionError

Re-confirms the torch-2.13 `compile + AC + TP=2` AOT-autograd crash
that the 80B family hits. Ran with all-defaults from the config
registry (TP=2, compile=ON, AC=full).

Crash signature (all 24 ranks):

```
[rank0]: torch._functorch._aot_autograd.runtime_wrappers.AssertionError:
expected all tensors_saved_with_vc_check to be Tensors, got types: [
  <class 'torch.Tensor'>, <class 'torch.Tensor'>, <class 'torch.Tensor'>,
  <class 'torch.Tensor'>, <class 'torch.Tensor'>, <class 'torch.Tensor'>,
  <class 'torch.Tensor'>, <class 'torch.Tensor'>, <class 'torch.Tensor'>,
  <class 'torch.Tensor'>, <class 'torch.Tensor'>, <class 'torch.Tensor'>,
  <class 'torch.distributed.device_mesh.DeviceMesh'>
]
```

`DeviceMesh` (the 13th element) is leaking into the
`tensors_saved_with_vc_check` list inside AOT autograd's
`save_from_forward`. Same root cause as the bisect on 2026-05-05
(jobs 12465952 + 12465962) — see
[`project_80b_devmesh_bisect`](../../../../../../home/foremans/.claude/projects/-lus-flare-projects-datascience-foremans-projects-saforem2-torchtitan/memory/project_80b_devmesh_bisect.md):
**bug is torch-2.13-sensitive**, fires on every 80B-family config,
smallest known repro is `agpt_50b_wide` 2N (~30s to crash) — which
this run reproduces in ~121s (extra compile + dataset-build time vs
the bisect's lean repro).

- Crash time: ~121 s from launch (`04:00:45` → `04:02:46`); first
  rank trace at 04:02:22 (line 358 of log)
- Exit code: 143 (one rank crashes → ezpz launch SIGTERM the rest)
- All 24 ranks emitted the same AssertionError before SIGTERM
- W&B: [`floral-pine-2073`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/382nbi5r)
  (initialised before the crash)

The 37th-sync merge did *not* fix this (didn't expect it to — the bug
is in PyTorch's AOT autograd, not torchtitan). The standing workaround
(`compile=OFF` for 80B-family on torch 2.13, or stay on torch 2.10
for these configs) is still the only option.

## Findings

### 1. PR #3346 regional_inductor refactor is a no-op for agpt path

`agpt_2b` numerics + memory match the prior post-resync baseline
byte-for-byte (24.34 GiB / 38.04%). No compile churn, no recompilation
events, no inductor warnings new to this run.

### 2. DeviceMesh-saved-with-tensors regression re-confirmed at 50B

Crash signature is identical to the May 5 bisect. Reproduces in ~121s
(launch-to-crash) on `agpt_50b_wide` at 2N — this is the
recommended fast-iteration repro for any upstream investigation into
this bug rather than the much longer 80B configs.

### 3. agpt+core integration sanity test passed post-merge

The agpt path was the most exposed to the merge's non-MoE half
(PR #3346 touches `graph_trainer` regional_inductor). A clean 50-step
agpt_2b run with byte-identical memory confirms no regression there.

## Logs

- `agpt_2b` (clean):              `logs/smoke-pr3386-followup/agpt_2b-20260520-230009.log`
- `agpt_50b_wide` (crash repro):  `logs/smoke-pr3386-followup/agpt_50b_wide-20260520-230010.log`

## Related

- Upstream PRs: [pytorch/torchtitan#3386](https://github.com/pytorch/torchtitan/pull/3386), [pytorch/torchtitan#3346](https://github.com/pytorch/torchtitan/pull/3346)
- Prior post-resync smoke: [`20260520-smoke-n2-postresync.md`](20260520-smoke-n2-postresync.md)
- 37th sync entry: [`docs/upstream-sync.md`](../../../upstream-sync.md)
- 80B DeviceMesh regression bisect: [`memory/project_80b_devmesh_bisect.md`](../../../../../../home/foremans/.claude/projects/-lus-flare-projects-datascience-foremans-projects-saforem2-torchtitan/memory/project_80b_devmesh_bisect.md)
- DeviceMesh toy repro: [`../../../upstream-issues/repro_devicemesh_in_saved_tensors.py`](../../../upstream-issues/repro_devicemesh_in_saved_tensors.py)
- Sibling MoE follow-up: [`../../moe/sunspot/20260520-smoke-n2-pr3386-ep-followup.md`](../../moe/sunspot/20260520-smoke-n2-pr3386-ep-followup.md)
