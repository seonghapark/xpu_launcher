# `_dist_reduce` skips mesh all_reduce on DTensor inputs, breaking loss reporting on TP > 1

**Status:** **Resolved upstream** as of commit
[`d64eabcce`](https://github.com/pytorch/torchtitan/commit/d64eabcce)
([PR #3159](https://github.com/pytorch/torchtitan/pull/3159), merged
2026-05-18). The fix switched the DTensor branch of `_dist_reduce`
from `full_tensor()` to `x = x.to_local()` followed by the
unconditional mesh `all_reduce` on the resulting plain tensor.

Our originally-filed
[PR #3204](https://github.com/pytorch/torchtitan/pull/3204) proposed a
mesh-overlap-helper approach; @fegin requested the simpler `to_local()`
shape during review, which landed via #3159 (alongside other Full
DTensor work that needed the same fix). PR #3204 was closed as
superseded on 2026-06-12.

The local ezpz workarounds (`loss.full_tensor()` in `trainer.py` +
`EzpzValidator.validate()` doing the same) were removed once the
upstream fix had been in our `ezpz` branch via sync long enough to
be smoke-validated.

The remainder of this doc is preserved as historical context.

**Affected:** torchtitan window 2026-04-27 (commit `1786292d`,
"[Module][Full DTensor] Config-based sharding infrastructure") through
2026-05-18 (commit `d64eabcce`).

## Summary

After commit `1786292d`, `torchtitan/distributed/utils.py:_dist_reduce`
short-circuits DTensor inputs by calling `full_tensor()` and skipping
the requested mesh `all_reduce`. This is silently incorrect when the
DTensor's mesh and the requested reduction mesh are orthogonal — which
is exactly the case for the trainer and validator's loss reduction:
the loss is a Replicated DTensor on the TP mesh, but the reduction is
requested across `loss_mesh` (= batch × cp).

The result: **all training and validation loss reporting on TP > 1 is
off by a factor of `dp_world_size`**. Gradients and optimizer steps are
unaffected; only the `loss:` field that prints to stdout / W&B is wrong.

## Reproducer

Any `tensor_parallel_degree > 1` run shows the bug. A clean
side-by-side from the same model:

| Config | TP | dp_world_size | Step 1 loss |
|---|---|---|---|
| llama3 / agpt 2B | 1 | 24 | ~12.95 (≈ ln(vocab) for vocab≈256k) |
| llama3 / agpt 2B | 2 | 12 | ~1.08 (= true ÷ 12) |

The shape `reported = true / dp_world_size` is exact (within rounding).

## Root cause

In `_dist_reduce`:

```python
if isinstance(x, DTensor):
    # DTensor path: ``full_tensor()`` already performs the mesh reduction
    # for Partial placements and is a no-op for Replicate. Skipping the
    # subsequent mesh all-reduce is required to avoid double-counting.
    ...
    return float(x.full_tensor().item())
```

The "double-counting" concern is valid only when the DTensor's mesh
overlaps the requested reduction mesh. The trainer's loss path passes
orthogonal meshes:

- DTensor mesh: TP (loss is Replicated on TP because of `loss_parallel`)
- Requested mesh: `loss_mesh = batch × cp`

`full_tensor()` on a Replicated DTensor is a no-op (returns the same
local value). Returning here without doing the explicit `all_reduce`
on `loss_mesh` drops the cross-batch sum.

## Affected call sites

All loss reductions in `torchtitan/trainer.py:780-783`,
`torchtitan/components/validate.py:320`, and the same patterns
in `experiments/forge/`, `experiments/ft/`, `experiments/ezpz/`.

## Suggested fix

Do `full_tensor()` first to materialize the value, then run the
requested mesh `all_reduce` regardless. For a Replicated DTensor on a
mesh orthogonal to the requested mesh, this is exactly what we want
(reduce across the requested axis). For a Partial DTensor on a mesh
that equals the requested mesh, `full_tensor()` reduces and the
subsequent `all_reduce` would double-count — so a check is needed:

```python
if isinstance(x, DTensor):
    # full_tensor reduces over the DTensor's mesh axes
    x = x.full_tensor()
    # If the DTensor's mesh covers the requested mesh, we're done
    if mesh is None or _mesh_covers(x_dtensor_mesh, mesh):
        return float(x.item())
    # Otherwise, do the explicit reduction on the requested mesh
```

A simpler conservative fix is to require callers to pass plain tensors
(via `.full_tensor()`) when they want a cross-mesh reduction, and add
an assertion in `_dist_reduce` that DTensor inputs only happen when
their mesh equals the requested mesh.

## Workaround (downstream)

Convert loss to plain tensor before `dist_sum`/`dist_max`:

```python
from torch.distributed.tensor import DTensor

if isinstance(loss, DTensor):
    loss = loss.full_tensor()
global_avg_loss = dist_utils.dist_sum(loss, loss_mesh, ft_pg)
```

We have applied this workaround in `experiments/ezpz/trainer.py` and
in a `Validator` subclass in `experiments/ezpz/validator.py`.

## How we found it

A `agpt_50b_wide` smoke test on Sunspot at TP=2 reported step-1 loss =
1.07 — too small for random init (expected ≈ 12.45 for vocab=256128).
A side-by-side TP=1 baseline showed loss = 12.95 at the same model
shape. The 12× ratio (= `dp_world_size`) pointed at the cross-batch
reduction, which led to the DTensor short-circuit added in `1786292d`.

The 2026-05-03 fix-verification run (`agpt_2b` at TP=2 with the
workaround in place, no compile, 3 steps) confirmed reported loss
matches the TP=1 baseline.
