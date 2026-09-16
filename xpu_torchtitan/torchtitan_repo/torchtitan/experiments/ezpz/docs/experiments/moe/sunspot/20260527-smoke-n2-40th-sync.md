# 40th Upstream-Sync Smoke — Sunspot 2N (2026-05-27)

Routine post-merge smoke after the 40th upstream sync
(`19c567f76..af33f7638`, 7 commits). Validates that `agpt_2b` and
`moe_2b_ep` still train cleanly on `experiments/ezpz/`, and tests
whether PR #3146 actually unblocks `--debug.deterministic` on MoE+XPU.

## TL;DR

- **`agpt_2b`** clean: 50 steps, exit 0, 194 s, peak 24.34 GiB,
  ~5,900 TPS, ~22% MFU, loss step 50 = 6.42. Matches the 2026-05-22
  baseline exactly on memory (24.34 GiB).
- **`moe_2b_ep` at new LBS=2 default** clean: 50 steps, exit 0,
  327 s, peak 27.09 GiB (42.33%), ~3,200 TPS, 9.4% MFU, loss step
  50 = 6.12. Numerically identical to the 2026-05-22 baseline
  (27.08 GiB, ~3,200 TPS, 9.4% MFU, loss 6.15). LBS=2 registry pin
  ([`59354e43f`](https://github.com/saforem2/torchtitan/commit/59354e43f))
  holds correctly post-merge.
- **`moe_2b_ep` with `--debug.deterministic`** — **still fails** with
  the same `_histc_xpu does not have a deterministic implementation`
  error from 2026-05-21. PR #3146 was advertised to fix this but
  **its actual diff only adds `aten.topk.default` to the SAC save
  list** — the promised `histc → bincount` swap in
  `torchtitan/models/common/moe.py:262` is missing from the commit.
  Verified against the upstream PR diff via GitHub API. Worth filing
  upstream.

## Environment

| Field        | Value                                             |
|--------------|---------------------------------------------------|
| Date         | 2026-05-27                                        |
| Branch       | `ezpz` (post-40th-sync merge + PR #3398 replay)   |
| Commit       | `b052f29e4`                                       |
| Machine      | Sunspot                                           |
| Job ID       | 12467455                                          |
| Nodes        | 2                                                 |
| Devices      | 24 (Intel Max 1550, 12/node)                      |
| Steps        | 50 per config                                     |
| Dataset      | blendcorpus                                       |
| Backend      | xccl                                              |
| Torch        | `2.13.0.dev20260519+xpu` (.venv)                  |
| Checkpoint   | disabled (`--checkpoint.no-enable`)               |

## Results

### `agpt_2b` (sanity)

- Log: `logs/smoke-40th-sync/agpt_2b-20260527-134002.log`
- 50 steps, **exit 0**, 194 s wall.
- Loss 12.97 → 6.42, peak memory **24.34 GiB (38.04%)**, ~5,900 TPS,
  ~22% MFU.
- Matches the 2026-05-22 baseline (also 24.34 GiB peak) — confirms
  the PR #3398 nn_modules import-path replay
  ([`b052f29e4`](https://github.com/saforem2/torchtitan/commit/b052f29e4))
  didn't perturb the agpt path.

### `moe_2b_ep` at LBS=2 (new registry default)

- Log: `logs/smoke-40th-sync/moe_2b_ep-20260527-134352.log`
- 50 steps, **exit 0**, 327 s wall.
- Loss 12.93 → 6.12, peak memory **27.09 GiB (42.33%)**, ~3,200 TPS,
  9.4% MFU.
- Numerically identical to the 2026-05-22 baseline at the same
  config (27.08 GiB, ~3,200 TPS, 9.4% MFU, loss 6.15). The MoE
  refactor in PR #3423 (3D tensors through MoE) inherits cleanly via
  the dispatcher path; no perturbation on the
  `LocalTokenDispatcher` + for_loop backend ezpz uses.

### `moe_2b_ep` with `--debug.deterministic` (PR #3146 test) — **still fails**

- Log: `logs/smoke-40th-sync/moe_2b_ep-deterministic-20260527-135015.log`
- Exit 143 (SIGTERM after rank failures), 53 s wall.
- Every rank crashes inside `torchtitan/models/common/moe.py:262`:

  ```
  num_tokens_per_expert = torch.histc(
      selected_experts_indices.view(-1),
      bins=self.num_experts,
      min=0,
      max=self.num_experts,
  )
  RuntimeError: _histc_xpu does not have a deterministic implementation,
  but you set 'torch.use_deterministic_algorithms(True)'.
  ```

- **Same error as 2026-05-21.** PR #3146 (which was supposed to fix
  this) actually only landed the `aten.topk.default` save-list
  addition; the promised `histc → bincount` swap in `common/moe.py`
  is missing from the diff.

#### How we verified the upstream gap

GitHub API query for PR #3146's actual file changes:

```
$ gh api repos/pytorch/torchtitan/pulls/3146/files --jq '.[] | .filename'
torchtitan/distributed/activation_checkpoint.py
```

Only one file. The PR body claims `histc → bincount (×2)` in
`torchtitan/models/common/moe/moe.py`, plus tests in
`tests/unit_tests/test_moe_routing.py` — neither is in the merged
diff. The file path in the PR body even references a non-existent
`models/common/moe/moe.py` subdirectory (we have
`models/common/moe.py`; the `common/moe/` directory exists as a
stale empty dir from PR #3386).

## Action items

1. **File upstream issue on `pytorch/torchtitan`** flagging that
   PR #3146's merged diff is missing the `histc → bincount` change
   the commit message describes. Either the PR author needs to land
   a follow-up, or the description should be corrected to match the
   actual scope.
2. Continue using the existing workaround on ezpz — skip
   `--debug.deterministic` for MoE+XPU runs (or set
   `warn_only=True` if a non-bit-exact regression gate is enough).

## Artifacts

- Smoke logs under `logs/smoke-40th-sync/`:
  - `agpt_2b-20260527-134002.log`
  - `moe_2b_ep-20260527-134352.log`
  - `moe_2b_ep-deterministic-20260527-135015.log`
