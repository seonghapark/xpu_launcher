# Known Issues and Operational Notes

Troubleshooting reference for running ezpz experiments across ALCF machines.

## `--checkpoint.async-mode=async` kills the cluster at 20B 512N+

**Symptoms:** 20B 512N runs reach a `step-N00` boundary, the async
ckpt save fires, and within ~15-20 min the entire 6,144-rank cluster
gets a gloo `Connection closed by peer` cascade. The save dir
(`outputs/checkpoints/.../step-N00/`) is created but never gets a
`.metadata` file — it's an empty / incomplete partial save. Resume
falls back to the previous complete ckpt and re-trains those steps.

**Root cause (diagnosed 2026-05-23):** every 20B 512N save on disk
between step-200 and step-800 happened on 2026-05-01 + 2026-05-03,
**before** the submit script added `--checkpoint.async-mode=async`.
After that flag landed (sometime between May 3 and May 11), nothing
has persisted past step-100 on the chain. The flag streams the 244 GB
6,144-shard ckpt to flare in the background concurrently with gloo
training-step heartbeat. At 6,144+ ranks, the concurrent write
pressure on flare + gloo backs up, a peer times out, cascade.

**Affects:** 20B 512N (confirmed). Likely 80B at any production scale
(244 GB+ ckpts on flare-shared trees). Does **not** affect 2B at any
scale (3,073 files / smaller ckpt → flare handles it), nor 20B 256N
where step-200/300 saved fine before the async switch and may save
fine again.

**Workaround:** Submit with `CHECKPOINT_ASYNC_MODE=disabled`:

```bash
qsub -l select=522 \
    -v NHOSTS_TRAIN=512,FAILOVER_MAX_RETRIES=2,CHECKPOINT_ASYNC_MODE=disabled \
    scripts/submit_agpt_20b_aurora_venv_failover.sh
```

Trade-off: sync saves *block* training. At 244 GB / 6,144 shards,
expect 5-15 min of save time at interval=100. ~10% MFU hit, but
that beats 0% on-disk persistence.

**Investigation status:** sync-mode submits queued 2026-05-23 evening
(`8505258` 20B 512N, `8505255-57` 20B 256N). Live test of the
hypothesis is pending dispatch.

## `training.dtype = bfloat16` silently freezes RMSNorm weights

**Symptoms:** RMSNorm `weight` parameters stay exactly at their `1.0`
init for every step of training. Loss/grad_norm curves look normal,
optimizer state shows non-zero `exp_avg` / `hessian`, but the on-disk
parameter values never move.

**Root cause:** `training.dtype = bfloat16` puts the master copy in
bf16 (no fp32 master). The bf16 ULP at scale 1.0 is `7.8e-3`; the
per-step optimizer update for norms is `~1.6e-5`, so updates round
to zero forever. Linear layers initialize at much smaller scales
(0.005-0.02) and update fine.

**Affects:** Every agpt and moe production run launched before
2026-04-29 (when the default in `agpt/config_registry.py` and
`moe/config_registry.py` was flipped from `bfloat16` to `float32`).

**Fix:** Use `training.dtype = float32`. With FSDP
`MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)`, the
master is fp32, forward all-gather is bf16, gradient reduce is fp32.
Memory cost: ~1 GB extra at 2B, ~10 GB at 20B.

**See full writeup:** [`training-dtype-bf16-norm-freeze.md`](training-dtype-bf16-norm-freeze.md).

## torch.compile SYCL compilation time at high rank counts

**Symptoms:** `torch.compile` takes hours to complete at 512+ nodes (6144+
ranks). The SYCL/XPU inductor backend generates and compiles C++ kernels
independently on each rank, causing filesystem contention on `/tmp`. At
extreme rank counts, compilation never finishes within walltime.

**Observed scaling:**

| Nodes | Ranks | 2B compile | 80B compile |
|-------|-------|------------|-------------|
| 2     | 24    | ~3 min     | ~2 min      |
| 4     | 48    | —          | ~4.5 min    |
| 128   | 1536  | ~3.5 min   | ~5 min      |
| 256   | 3072  | ~4.5 min   | pending     |
| 512   | 6144  | 12+ hours  | pending     |

**Root cause:** Each rank compiles kernels independently to `/tmp`. At 6144
ranks across 512 nodes, the parallel SYCL C++ compilation creates massive
filesystem I/O contention. The 80B model may fare better because it has
fewer unique kernel shapes (84 identical transformer blocks) while the
small 2B model (12 blocks) triggers more diverse compilation paths
relative to its compute time.

**At 512 nodes (80B):** Compile OOMs on **CPU memory** (`MemoryError:
std::bad_alloc`) within 2 minutes. The inductor backend exhausts host RAM
generating SYCL kernels for 84 transformer blocks across 6144 ranks.

**Workaround:** Use `--compile.no-enable` at 512+ nodes. For the 80B
model, compile works at 128N (5 min) but fails at 512N (CPU OOM). The
2B model compiles at 256N (4.5 min) but takes 12+ hours at 512N.

**See also:**
[Scaling and production runs report](../production/scaling-performance.md)

## torch.compile + AC + TP crashes on torch 2.12+ (DeviceMesh assertion)

**Symptoms:** `AssertionError: expected all tensors_saved_with_vc_check to be
Tensors, got types: [..., <class 'torch.distributed.device_mesh.DeviceMesh'>]`
during the first training step. Crashes in `runtime_wrappers.py:save_from_forward`.

**Root cause:** AOT autograd traces the compiled + activation-checkpointed forward
pass and saves intermediate values for the backward pass. On torch 2.12+, the
`DeviceMesh` object from the TP mesh leaks into the saved tensors list. The
`save_from_forward` assertion rejects it because `DeviceMesh` is not a
`torch.Tensor`.

**Affects:** All dense agpt configs with `compile + AC + TP > 1`. Confirmed on
torch 2.12.0.dev20260415+xpu and 2.13.0.dev20260418+xpu. Does not affect
TP=1 configs (no TP mesh to leak) or no-compile configs (no AOT autograd).

**Tested configurations:**
- `compile + AC=full + TP=2` → DeviceMesh assertion (CRASH)
- `compile + AC=none + TP=2` → OOM (61.18/63.98 GiB, AC required)
- `compile + AC=none + TP=2 + gc_freq=1` → same OOM
- `no-compile + AC=full + TP=2` → **OK** (85 tps, 15.6% MFU, 95% memory)

**Workaround:** `--compile.no-enable` for models that require TP (50B+). Eager
mode with AC achieves comparable throughput to compiled mode on torch 2.10
(85 vs 83 tps for 80B) since compile provided minimal benefit with TP anyway.

**Status:** Not fixed in torch 2.13.0.dev20260418+xpu. Upstream PyTorch bug
in `torch._functorch._aot_autograd.runtime_wrappers`.

## torch.compile + loss compilation crashes with TP (upstream PR #2741)

**Symptoms:** `InductorError: PendingUnbackedSymbolNotFound: Pending unbacked
symbols {zuf0, zuf1}` during the first training step. The crash occurs in the
compiled loss function, not in the model forward pass. Only triggers when
`loss_parallel` is active (i.e. TP > 1) and loss compilation is enabled.

**Root cause:** Upstream PR #2741 ("Enable per-layer compile with or without
MoE") consolidated `apply_compile_dense` and `apply_compile_sparse` into a
single `apply_compile` that unconditionally sets
`torch._dynamo.config.capture_scalar_outputs = True`. This flag is needed for
MoE dynamic shapes (expert routing produces data-dependent scalars), but it
breaks dense models when the loss function is separately compiled. With the
flag set, `F.cross_entropy` with `ignore_index` + `reduction='sum'` produces
unbacked symbols from internal token masking that aren't bound to any output,
causing `compute_unbacked_bindings` to raise.

**Fix:** Reset the flag after `apply_compile` in the dense agpt parallelize
path (`ezpz/agpt/parallelize.py`, commit `e8cbb8ef`):

```python
if model_compile_enabled:
    apply_compile(model, compile_config)
    torch._dynamo.config.capture_scalar_outputs = False
```

**Workaround:** `--compile.components=model` (skip loss compilation).

**Affects:** All dense agpt configs with TP > 1 and compile enabled (80B
variants at TP=2+). Does not affect MoE configs (which need the flag) or
dense configs at TP=1 (where `loss_parallel` is inactive).

## torch.compile + SDPA `set_priority=True` (PyTorch 2.11)

**Symptoms:** `RuntimeError('Invalid backend')` during `torch.compile` tracing
of `F.scaled_dot_product_attention`. Crashes on all ranks at model init, before
any training steps run.

**Root cause:** `torch._dynamo` bug — `SDPAKernelVariable.enter()` in
`ctx_manager.py` creates FX graph nodes that pass proxy objects to `int()`
when `set_priority=True`, producing invalid backend IDs for
`_set_sdp_priority_order()`. Only triggers in the full model context
(activation checkpointing + DTensor + distributed), not in standalone tests.

**Fix:** `EzpzScaledDotProductAttention` in `experiments/ezpz/agpt/__init__.py`
overrides `forward()` to call `sdpa_kernel(self.sdpa_backends)` without
`set_priority=True`. Use `_default_inner_attention()` in configs — it
auto-selects the right subclass based on device (XPU vs CUDA).

**Affects:** All model configs using SDPA with `torch.compile` enabled.

## Blendcorpus data cache race condition

**Symptoms:** `FileNotFoundError` or `EOFError` on
`.cache/blendcorpus/*_index.npy` when launching with a new combination of
`seq_len` and `global_batch_size`.

**Root cause:** The blendcorpus data loader caches index files keyed by a hash
of (seq_len, global_batch_size, dataset). On the first run with a new
combination, rank 0 builds the cache while other ranks race to read it.
The hash changes with `global_batch_size`, which depends on parallelism
settings (e.g. TP=4 FSDP=2 LBS=1 gives GBS=2, different from FSDP=8 LBS=2
which gives GBS=16).

**Workaround:** Just retry — rank 0 builds the cache on the first attempt, so
the second run will find the files. Kill stale processes first:

```bash
ssh <node> 'ps aux | grep torchtitan | grep -v grep | awk "{print \$2}" | xargs -r kill'
```

## CXI resource leak after killing processes (Polaris)

**Symptoms:** `mpiexec` fails instantly with:

```
CXI alloc failed on cxi0: request exceeds PTEs, TXQs, TGQs, EQs, CTs, LEs, ACs limits
```

**Root cause:** When training processes are killed (via SIGTERM, TaskStop, or
SSH disconnect), the Slingshot CXI network resources are not released by the
palsd daemon.

**Fix:**

1. Kill ALL related processes on ALL nodes in the allocation
2. Wait a few seconds for CXI cleanup
3. If still failing, submit a new PBS job (fresh nodes)

```bash
# Kill on both nodes
ssh <node1> 'ps aux | grep torchtitan | grep -v grep | awk "{print \$2}" | xargs -r kill'
ssh <node2> 'ps aux | grep torchtitan | grep -v grep | awk "{print \$2}" | xargs -r kill'
```

## Memory limits by machine

### Polaris (8x NVIDIA A100-SXM4-40GB per 2-node job)

| Config | Status | Memory | Parallelism |
|--------|--------|--------|-------------|
| agpt debugmodel | PASS | 2.23 GiB (6%) | FSDP=8 |
| agpt 2B | PASS | 24.98 GiB (63%) | FSDP=8 |
| agpt 7B | PASS | 12.33 GiB (31%) | FSDP=8 |
| agpt 20B | PASS | 35.07 GiB (89%) | FSDP=8, LBS=1, seq=4096 |
| agpt 50B | OOM | 36/39.5 GiB | TP=2, FSDP=4, LBS=1, seq=4096 |
| agpt 80B | OOM | 38.2/39.5 GiB | TP=4, FSDP=2, LBS=1, seq=4096 |
| moe debugmodel-4B | PASS | <54% | FSDP=8 |
| moe 7B | PASS | 35.09 GiB (89%) | FSDP=8, AC=none |
| moe 10B_2B_sdpa | PASS | 30.22 GiB (77%) | FSDP=8, AC=none, LBS=1 |

50B+ agpt models need 4+ Polaris nodes or A100-80GB GPUs.

### Aurora / Sunspot (24x Intel Max 1550 per 2-node job)

- 64 GiB per tile, 12 tiles per node
- 80B fits at TP=2 on Aurora again as of 2026-04-18 (88 TPS, 16% MFU)
- Previously broken 2026-04-12 through 2026-04-17 (OOM by 60 MiB); fixed by
  removing `import intel_extension_for_pytorch` (IPEX allocator overhead)
- 80B variants (alt, wide, deep) run reliably at TP=3+ on both Aurora and Sunspot
- Best 80B throughput: 80B TP=2 compile = 88 TPS, 16.05% MFU (Aurora, 2026-04-18)

## Tokenizer compatibility

The `agpt_8B` config uses `vocab_size=128256` (Llama 3 architecture) which
requires a Llama 3 tokenizer. The available tokenizer on ALCF systems is
Gemma (`vocab_size=32000`), causing a CUDA assert on embedding index out
of bounds.

**Affected:** `agpt_8B` only. All other agpt configs use `vocab_size=32000`
or `vocab_size=256128` (compatible with Gemma tokenizer).

## MoE activation checkpointing incompatibility

**Symptoms:** `CheckpointError` during AC recomputation for MoE configs 7B+.

**Root cause:** MoE routing is non-deterministic (expert dispatch varies
between forward passes). Activation checkpointing recomputes the forward
pass during backward, but the routing decisions differ from the original
forward, causing mismatched tensor shapes.

**Fix:** Add `--activation_checkpoint.mode none` for MoE configs 7B and larger
(7B, 10B_2B, 10B_2B_sdpa, 16B, small). Smaller configs (debugmodel, 500M,
2B, 4B) are not affected in short runs.

**Trade-off:** Disabling AC increases memory usage significantly. The moe 7B
config uses 35 GiB (89%) with AC=none on Polaris A100-40GB.

## MoE + Tensor Parallelism (TP > 1)

**Symptoms:** `AssertionError: q, k, v must have the same placements, but got
q=(Shard(dim=2),), k=(Shard(dim=2),), v=(Replicate())`

**Affected:** All MoE configs with `--parallelism.tensor_parallel_degree > 1`.

**Root cause:** The MoE MLA (Multi-head Latent Attention) implementation uses
LoRA-based KV projection with asymmetric sharding. When TP shards q and k
across heads, the v tensor from `wkv_b` remains replicated because its
projection shape doesn't match the TP sharding pattern.

**Workaround:** Use TP=1 for MoE models. Expert parallelism (EP>1) is also
blocked on `aurora_frameworks-2025.3.1` (missing `ShardPlacementResult`).
MoE scaling requires a newer PyTorch version.

## Context Parallelism (CP > 1) on agpt

**Symptoms:** With compile: `TorchRuntimeError: Dynamo failed to run FX node
with fake tensors: call_function scaled_dot_product_attention`. Without
compile: `RuntimeError: aten.add.Tensor got mixed torch.Tensor and DTensor`.

**Affected:** All agpt configs with `--parallelism.context_parallel_degree > 1`.

**Root cause:** The agpt attention implementation doesn't convert all tensors
(e.g. RoPE embeddings) to DTensors on the CP mesh. When CP shards Q/K/V
along the sequence dimension, the non-sharded tensors remain as regular
`torch.Tensor`, causing DTensor/Tensor mixing errors.

**Workaround:** Use CP=1 (default). For longer sequences, increase TP instead.

## 80B TP=2 on Aurora (regression 2026-04-12, resolved 2026-04-18)

**Status: RESOLVED** — 80B TP=2 works again as of 2026-04-18. See
[restoration report](../experiments/agpt/aurora/20260418-80b-tp2-restored.md).

**Symptoms (when broken):** `torch.OutOfMemoryError` or
`UR_RESULT_ERROR_OUT_OF_RESOURCES` on step 2. Step 1 completes at
60.75 GiB (94.94%) but step 2 needs 5.25 GiB with only 5.19 GiB free
(missed by 60 MiB).

**Affected period:** 2026-04-12 through 2026-04-17. Tested on 4 different
node pairs (`x4216c5s*`, `x4704c1s*`, `x4219c2s*`, `x4310c3s*`), all
failed identically.

**Resolution:** Removing `import intel_extension_for_pytorch` (IPEX). IPEX
registered XPU allocator hooks that added ~60 MiB overhead per rank —
exactly the margin between fitting and OOM at 93.49% utilization. The
framework is still `torch==2.10` (`aurora_frameworks-2025.3.1`).

On 2026-04-18, 80B TP=2 runs at 59.82 GiB (93.49%), 88 TPS, 16.05% MFU —
identical to the original April 4 benchmark (89 TPS, 16.24% MFU).
