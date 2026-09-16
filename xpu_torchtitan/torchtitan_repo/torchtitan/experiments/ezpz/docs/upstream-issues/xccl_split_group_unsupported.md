# `ProcessGroupXCCL` never overrides `supportsSplitting()`

> **Filed upstream as [pytorch/pytorch#186548](https://github.com/pytorch/pytorch/issues/186548)** (2026-06-07). Issue body
> mirrors this doc + adds a verified Sunspot repro log + `collect_env`
> block. Sibling: [pytorch/pytorch#171938](https://github.com/pytorch/pytorch/issues/171938).
> Workaround module:
> [`xccl_split_group_workaround.py`](../../xccl_split_group_workaround.py).
> Removal criteria: see the bottom of this doc.

## TL;DR

`ProcessGroupXCCL` inherits `Backend::supportsSplitting()` from
[`Backend.hpp`](https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/Backend.hpp),
which defaults to `false`. `ProcessGroupNCCL` overrides it to `true` (see
[`ProcessGroupNCCL.hpp`](https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp)).
xccl does not.

Combined with `DeviceMesh._init_one_process_group`'s gate (torch 2.13,
`device_mesh.py:550-562`), this means every nested mesh creation on XPU
under the eager-init pathway raises:

```
RuntimeError: No backend for the parent process group or its backend does
not support splitting
```

This kills `moe_2b_ep` (the EP-flavored sparse mesh is built
unconditionally inside `ParallelDims.build_mesh`), and any future EP /
multi-dim mesh config on XPU.

## Why the gate fires for xccl

`DeviceMesh._init_one_process_group` (torch 2.13):

```python
# device_mesh.py:550-562
if (
    (
        getattr(default_group, "bound_device_id", None) is not None
        or dist_config.use_torchcomms
    )
    and torch.accelerator.is_available()
    and (
        backend is None
        or default_group._get_backend(
            torch.accelerator.current_accelerator()
        ).name() == backend
    )
):
    dim_group = split_group(...)   # <-- xccl explodes inside here
    ...
```

`split_group` then does:

```python
# distributed_c10d.py:5565-5570
if (
    not parent_backend or not parent_backend.supports_splitting
) and not _use_torchcomms_enabled():
    raise RuntimeError(
        "No backend for the parent process group or its backend does not "
        "support splitting"
    )
```

`parent_backend.supports_splitting` for xccl is `False` (inherited from
`Backend`), and we don't enable torchcomms — so the RuntimeError fires
**before** any xccl-level `split` call is attempted.

The gate only checks `bound_device_id` / `torch.accelerator.is_available()` /
backend-name match. None of those rule out xccl. The eager init path
that ezpz uses on XPU sets `bound_device_id`, so we always take the
broken branch.

## Trigger inside torchtitan

`torchtitan/distributed/parallel_dims.py:200-204` builds the EP sparse
mesh unconditionally as part of `build_mesh()`:

```python
full_sparse_mesh = unflatten_mesh(
    self._world_mesh,
    ("pp", "dp_replicate", "efsdp", "ep"),
    (self.pp, self.dp_replicate, efsdp, self.ep),
)
```

The unflatten calls `DeviceMesh._unflatten` → `_init_process_groups`
→ `_init_one_process_group`, which hits the broken branch for every
named axis. Failure is at trainer init, before any forward pass.

## What needs to land upstream

Two fixes are required:

1. **`ProcessGroupXCCL.hpp` needs a `supportsSplitting() override`** that
   matches NCCL's:

   ```cpp
   bool supportsSplitting() const override {
     return true;
   }
   ```

2. **`ProcessGroupXCCL::split` (or equivalent) needs to be implemented
   and exercised** by the `test_c10d_xccl.py` suite. Setting
   `supportsSplitting()` to `true` without a working `split()`
   implementation just moves the failure from a Python `RuntimeError`
   to an `xcclCommSplit`-shaped C++ crash.

Both should be filed against `pytorch/pytorch` once a working
`xcclCommSplit` shim is in hand.

## Local workaround in ezpz

[`torchtitan/experiments/ezpz/xccl_split_group_workaround.py`](../../xccl_split_group_workaround.py)
monkey-patches `DeviceMesh._init_one_process_group` at import time
(installed lazily inside `FaultTolerantTrainer.init_distributed`,
[`trainer.py:485`](../../trainer.py)).

The wrapper:

1. Detects xccl/XPU at module load (`torch.distributed.is_xccl_available()
   and torch.xpu.is_available()`); on cuda/cpu it's a no-op.
2. At each `_init_one_process_group` call, checks the default group's
   per-accelerator backend's `supports_splitting`. If it's `True` (NCCL),
   calls upstream verbatim.
3. If `supports_splitting` is `False` (xccl), temporarily clears
   `bound_device_id` on the default group so the upstream gate's first
   clause goes `False` and we fall through to the `new_group` loop
   (same code path xccl already uses successfully in single-mesh
   configs). Restores `bound_device_id` afterwards so eager-init
   bookkeeping is preserved for other code paths.

The patch is idempotent. It deliberately uses the same `new_group`
fallback that the upstream code already includes — we're just
extending the condition under which that fallback is taken.

## Removal criteria

Delete `xccl_split_group_workaround.py` and the import in `trainer.py`
once both upstream conditions hold:

1. `ProcessGroupXCCL` declares `supportsSplitting() override { return
   true; }`.
2. `ProcessGroupXCCL::split` is implemented and a CI test exercises
   nested mesh creation under xccl.

## Related

- [`train_timeout_xpu_silent_noop.md`](train_timeout_xpu_silent_noop.md)
  — sibling xccl Python-dispatch gap (no `xpu` branch in
  `_set_pg_timeout`); same shape of "upstream Python codepath doesn't
  know about xccl yet".
