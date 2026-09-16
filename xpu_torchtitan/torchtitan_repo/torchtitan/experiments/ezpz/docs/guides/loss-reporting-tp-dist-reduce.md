# Loss reporting was off by `dp_world_size` on TP > 1 — fixed upstream 2026-05-18

**Status:** Resolved on upstream `main` as of commit
[`d64eabcce`](https://github.com/pytorch/torchtitan/commit/d64eabcce)
([PR #3159](https://github.com/pytorch/torchtitan/pull/3159), merged
2026-05-18). Our local ezpz workaround was removed in commit `TBD`
once the fix had been in our `ezpz` branch via upstream sync long
enough to be smoke-validated.

**Affected window:** All torchtitan runs with `tensor_parallel_degree > 1`
between upstream commits `1786292d` (2026-04-27) and `d64eabcce`
(2026-05-18). Both training and validation loss reporting were wrong
by a factor of `dp_world_size`. **Gradients and optimizer steps were
unaffected** — only the `loss:` field in stdout / W&B was wrong.

## Symptom (historical)

Step-1 loss for our `agpt_*` configs at vocab=256128 should be roughly
`ln(256128) ≈ 12.45 nats` (uniform softmax). On TP=1 runs this matched:

| Run | TP | Step 1 loss |
|---|---|---|
| `agpt_2b` (Apr 25) | 1 | **12.95** ✓ |

After the upstream bug landed, TP > 1 runs showed much smaller values:

| Run | TP | dp_size | Step 1 loss | True loss |
|---|---|---|---|---|
| `agpt_50b_wide` (May 3) | 2 | 12 | **1.07** ✗ | 12.84 (= 1.07 × 12) |
| `agpt_2b` TP=2 (May 3, no fix) | 2 | 12 | (would be) 1.08 | 12.94 |
| `agpt_2b` TP=2 (May 3, ezpz workaround) | 2 | 12 | **12.94** ✓ | 12.94 |

Reported `loss = true_loss / dp_world_size` for the entire window.

## Root cause (historical)

`torchtitan/distributed/utils.py:_dist_reduce` short-circuited DTensor
inputs:

```python
if isinstance(x, DTensor):
    # ... full_tensor() reduces over the DTensor's own mesh ...
    return float(x.full_tensor().item())
```

This was correct only if the DTensor's mesh equaled the requested
reduction mesh. But every call site of `dist_sum`/`dist_max` passed
`batch_mesh` or `loss_mesh` (= batch × cp), while the loss was a
Replicated DTensor on the **TP mesh** — orthogonal to the requested
mesh. `full_tensor()` on a Replicated DTensor is a no-op, so the
requested cross-batch reduction was silently dropped. Every batch
rank ended up reporting its own local per-token NLL with no sum
across batches.

The loss arrives Replicated-on-TP because the trainer enables
`loss_parallel()` whenever `tp_enabled` is True; `cross_entropy_loss`
inside `loss_parallel()` returns a Replicated DTensor on the TP mesh
and the subsequent `loss / global_valid_tokens` preserves the
DTensor-ness.

## Upstream fix

Filed as
[pytorch/torchtitan#3204](https://github.com/pytorch/torchtitan/pull/3204)
(2026-05-03). That PR proposed `full_tensor()` + a mesh-overlap helper
that detects whether the DTensor's mesh shares axes with the requested
reduction mesh, only skipping the explicit reduction in the shared-axes
case.

The actually-landed fix took a simpler shape, proposed by @fegin during
the PR review: always `to_local()` the DTensor, then run the requested
mesh `all_reduce` on the resulting plain tensor unconditionally. That
landed as part of
[PR #3159](https://github.com/pytorch/torchtitan/pull/3159) (commit
`d64eabcce`, merged 2026-05-18) — the [Full DTensor] config-based
Llama3 work needed the fix anyway, since the same code path under-reports
loss for non-full DTensors. From that PR's issue list:

> Our `_dist_reduce` under-reports loss for non full tensor:
> `full_tensor()` reduces only over the DTensor's own mesh (TP-only
> in legacy), skipping DP. Loss was under-reported by
> `dp_shard × cp`. Switched to `to_local()` then mesh all_reduce
> over loss_mesh.

PR #3204 was closed as superseded on 2026-06-12.

## Implications for historical production dashboards

- **2B / 20B production W&B dashboards: correct.** Those configs use
  `tensor_parallel_degree=1`, so loss never entered the buggy DTensor
  branch.
- **80B production W&B dashboards (Apr 27 – May 18, before upstream
  sync brought in the `to_local()` fix): under-reported by a factor of
  `dp_world_size`.** A 256N run (`world=3072`, TP=2) reported loss
  values 1536× too small. Multiply reported losses from that window
  by `dp_world_size = world_size / tp_degree` to recover true NLL.

## Verification

The 2026-05-03 2B-TP=2-no-compile smoke (with the ezpz workaround in
place) showed step-1 loss = 12.94, matching the known-good 2B-TP=1
baseline of 12.95. Upstream's `to_local()` fix produces the same result
via a different mechanism (`to_local()` on a Replicate DTensor returns
the local replica; subsequent `funcol.all_reduce` over the loss mesh
sums across batch ranks — equivalent end-state to our `full_tensor()`
workaround).
