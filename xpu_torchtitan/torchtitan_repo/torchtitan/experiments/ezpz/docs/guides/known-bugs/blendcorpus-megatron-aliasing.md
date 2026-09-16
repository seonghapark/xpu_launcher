# `BlendCorpusDataLoader` aliases torchtitan parallelism axes onto Megatron knobs

> **Status:** Latent (all current production runs use pure FSDP — no
> PP, no CP, no Megatron-style TP path). Will silently misbehave the
> moment any of those features are turned on with this dataloader.
>
> **Tracked in:** [`docs/TODO.md` §6](../../TODO.md).
>
> Discovered: 2026-05-05 while debugging the `agpt_2b --validator.enable`
> trace.

## Summary

`torchtitan/experiments/ezpz/blendcorpus/blendcorpus_builder.py:__init__`
constructs a `bc_cfg` for the Megatron-derived `blendcorpus` library by
mapping torchtitan's `parallel_dims` axes onto Megatron's parallelism
knobs. Several mappings are **name collisions, not semantic
equivalents**, and one (the `dist.barrier` monkey-patch) was being
applied even on torch versions that don't need it.

The two active items have been fixed (gloo monkey-patch gated on
`torch < 2.13`; `_train_ds` renamed to `_served_ds` so the attribute
matches its actual contents). The remaining items are **latent
foot-bullets** — they only matter if/when we enable PP, CP, or real
Megatron-TP communication on this dataloader. Each is enumerated
below.

## The mapping today

```python
# blendcorpus_builder.py
parallel_dims = kwargs.get("parallel_dims")
tp_degree = getattr(parallel_dims, "tp", 1)
pp_degree = getattr(parallel_dims, "pp", 1)
cp_degree = getattr(parallel_dims, "cp", 1)
...
bc_cfg = SimpleNamespace(
    ...
    tensor_model_parallel_size=int(tp_degree),
    pipeline_model_parallel_size=int(pp_degree),
    sequence_parallel_size=int(cp_degree),       # ← CP aliased to SP
    ...
)
...
bc_mpu.initialize_model_parallel(
    tensor_model_parallel_size=bc_cfg.tensor_model_parallel_size,
    pipeline_model_parallel_size=bc_cfg.pipeline_model_parallel_size,
    sequence_parallel_size=bc_cfg.sequence_parallel_size,
)
```

## Issues, by severity

### 1. PP semantics mismatch — silently underfeeds PP

Megatron's `pipeline_model_parallel_size` controls how the data
**sampler** partitions samples across PP ranks. In Megatron PP, only
the first PP stage receives data and the rest get empty tensors;
`pipeline_model_parallel_size > 1` divides the global batch by `pp`.

torchtitan's PP doesn't work that way — the *trainer* decides which
ranks read inputs (the first PP stage), and the model passes hidden
states between stages. The dataloader doesn't need to know about PP
at all in the torchtitan flow.

**Effect:** Passing `pp_degree=N > 1` to the Megatron sampler will
silently give us `1/N` of the per-rank batch we asked for.

**Fix:** Set `pipeline_model_parallel_size=1` unconditionally in
`bc_cfg`, since the blendcorpus sampler should not be driving PP
partitioning in our flow.

### 2. CP → SP aliasing — wrong sharding axis

torchtitan's **CP (context parallel)** shards the *sequence dimension*
across CP ranks during attention. CP wants **one full sample
replicated across the CP group**, then sharded inside the model layer.

Megatron's **`sequence_parallel`** is a *TP variant* that shards
activations along the sequence dimension only inside transformer
layers, and is bundled with TP — not a separate parallelism axis.
The blendcorpus sampler treats `sequence_parallel_size > 1` as a
divisor of the global batch.

**Effect:** With `cp > 1`, we'd see the global batch divided by
`cp_degree` instead of replicated across the CP group. Torchtitan's
attention layer would then receive distinct samples on every CP rank
when it expects identical samples that it can shard along seq itself.

**Fix:** Set `sequence_parallel_size=1` unconditionally in `bc_cfg`
and let the model handle CP at the layer level. CP-aware data
loading (replicate-then-shard) belongs in torchtitan's CP plumbing,
not in the blendcorpus sampler.

### 3. `dp_world_size` double-source

Two independent computations of "data parallel world":

- We pass `dp_world_size` from the torchtitan `batch_mesh`
  (= DP × CP) to the loader's `__init__`.
- Megatron's sampler internally derives DP from
  `world_size / (TP × PP × SP)`.

With CP > 1 and the SP aliasing in #2, the two disagree, and the
`requested_global_batch_size = local_batch_size * dp_world_size`
computation we use for `bc_cfg.global_batch_size` no longer matches
what Megatron's sampler will partition by.

**Effect:** Off-by-`cp_degree` `global_batch_size` whenever CP > 1.

**Fix:** Naturally falls out of #2 — once `sequence_parallel_size=1`
in `bc_cfg`, Megatron's DP computation matches our `dp_world_size`.

### 4. Two parallel-state systems

```python
bc_mpu.initialize_model_parallel(
    tensor_model_parallel_size=...,
    pipeline_model_parallel_size=...,
    sequence_parallel_size=...,
)
```

This builds Megatron's *own* TP/PP/SP groups via
`torch.distributed.new_group`, independent of torchtitan's
`DeviceMesh`. Now both libraries think they own model-parallel state.

**Effect today:** mostly inert because (a) we use
`comm_backend="standard"` in MoE configs (so we don't actually
exercise Megatron's TP collectives), (b) TP/PP/SP are usually 1 in
the bc_cfg path, and (c) blendcorpus only uses `bc_mpu` to look up
its own DP rank for sample partitioning, which we wouldn't notice
disagreeing as long as the *total* world matches.

**Effect if anyone wires Megatron-TP communication:** two parallel
TP groups using different ranks → silently incorrect collectives.

**Fix:** Either skip `initialize_model_parallel` entirely when
`tp == pp == cp == 1` (which is always, today), or rebuild Megatron's
groups from torchtitan's existing meshes rather than duplicating.

### 5. Cosmetic Megatron-only knobs

Two leaks of Megatron internals into our `Config`:

- **`data_impl="mmap"`** — hard-coded inside `bc_cfg`, not exposed as
  a `Config` field. No way to override without editing this file.
- **`dataloader_type: str = "single"`** — Megatron's sampler
  selector with values `"single"` / `"cyclic"` / `"external"`. None
  of those map to anything torchtitan exposes; it's dead config from
  torchtitan's perspective. Either drop entirely or document why we
  keep `"single"`.

These don't affect correctness; they just make the `Config` look
larger than it is.

## What's already fixed (2026-05-05)

- **#5: Gloo barrier monkey-patch was over-applied on torch ≥ 2.13.**
  The `_xccl_needs_barrier_fix` predicate now gates on the torch
  version (`< (2, 13)`) before checking the device-bound state, so
  the upstream-fixed XCCL barrier on 2.13 is no longer silently
  shadowed by a CPU/gloo replacement.
- **#9: `self._train_ds` actually held the `valid_ds` when
  `serve_validation=True`.** Renamed to `self._served_ds` so the
  name matches what it stores; the consumers in
  `set_consumed_by_global_step` and `load_state_dict` use the new
  name too.

## Why this matters even though it's latent

Every one of these items will produce **silently wrong training**
the day someone enables PP, CP, or real Megatron-TP on a blendcorpus
run. The errors won't be assertion crashes — they'll be off-by-N
batch sizes, double-counted samples, or replicated parallel groups
diverging from torchtitan's. Documenting them here so the trap is
visible the next time someone tries.
