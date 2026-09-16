# TODO

## 1. Docs Restructure

### Problem
Scaling results, benchmarks, and throughput data are scattered across
`docs/benchmarks.md`, `docs/benchmark-80B.md`, `docs/scaling-study.md`,
`docs/scaling-study-torch213.md`, `docs/production-training/scaling-performance.md`,
and per-experiment logs in `docs/experiments/{agpt,moe}/{aurora,sunspot,polaris}/`.
Finding the latest numbers for a given model/machine requires checking 5+ files.

### Proposed structure
```
docs/
├── README.md                          # overview + links
├── configs/
│   ├── dense.md                       # agpt model configs (from dense-configs.md)
│   └── moe.md                         # moe model configs (from moe-configs.md)
├── guides/
│   ├── running-with-newer-pytorch.md
│   └── known-issues.md
├── scaling/
│   ├── README.md                      # consolidated scaling summary table
│   ├── agpt-2b.md                     # all 2B results across machines/torch versions
│   ├── agpt-20b.md
│   ├── agpt-80b.md                    # merge benchmark-80B.md + throughput leaderboard
│   └── moe-7b.md
├── production/
│   └── scaling-performance.md         # production run tracker
└── experiments/                       # raw per-run logs (unchanged)
    ├── agpt/{aurora,sunspot,polaris}/
    └── moe/{aurora,sunspot}/
```

### Key changes
- `scaling/README.md` becomes the single place to find latest numbers
  for any model/machine/torch version, with links to raw experiment logs
- `configs/` and `guides/` consolidate reference material
- Top-level clutter (`benchmarks.md`, `scaling-study.md`, etc.) moves
  into the appropriate subdirectory
- `experiments/` stays as-is (raw per-run logs)

## 2. 80B compile + AC on torch 2.13

### Problem
80B TP=2 with compile + activation checkpointing crashes on torch 2.13
with `AssertionError: expected all tensors_saved_with_vc_check to be Tensors,
got types: [..., DeviceMesh]`. The AOT autograd tracer saves DeviceMesh
objects from DTensor-parallelized modules into the autograd graph, and
the AC version check rejects non-Tensor objects.

### Status
- Not patchable from our side — the DeviceMesh is structurally embedded
  in the autograd saved state
- Workaround: use `--compile.no-enable` on torch 2.13 (49 tflops / 16.4% MFU)
- The 80B works with compile on torch 2.10 + IPEX (22.5 tflops / 7.5% MFU)
- Needs upstream PyTorch fix in `torch/_functorch/_aot_autograd/runtime_wrappers.py`

## 3. MoE Throughput Optimization

### Problem
MoE models currently get 4-11% MFU on Aurora. torch.compile hurts MoE (-35%
for 500m). The 7b+ configs can't use activation checkpointing (routing
non-determinism causes CheckpointError).

### Knobs to try
- **Expert parallelism (EP)**: Split experts across ranks. Try
  `--parallelism.expert_parallel_degree 12` (one expert group per tile).
  Reduces per-rank expert count, may improve memory and compute balance.
- **Tensor parallelism for MoE**: Try TP=2 or TP=4 for the larger MoE
  configs (7b, 10b). Currently all MoE runs use TP=1 (FSDP only).
- **Context parallelism (CP)**: CP=2 or CP=4 to split the sequence
  dimension. Halves activation memory per rank.
- **Per-block compile**: The MoE parallelize.py already does per-block
  compile instead of fullgraph. Test if `compile.components=["loss"]`
  alone helps (skip model compile, just compile loss).
- **DeepEP / HybridEP backends**: Try
  `--parallelism.expert_parallel_comm_backend deepep` or `hybridep`
  for optimized expert dispatch communication.
- **Float8 quantization**: `Float8LinearConverter` or
  `Float8GroupedMMConverter` for expert layers — cuts compute/memory.
- **Larger batch sizes**: MoE models use less memory than dense.
  Try LBS=2 or LBS=4 for the smaller configs (500m, 2b).

### Experiments to run
```
# EP sweep for 10b
for ep in 1 2 4 12; do
  ezpz launch ... --config moe_10b_2b_sdpa \
    --parallelism.expert_parallel_degree $ep \
    --activation_checkpoint.mode none --compile.no-enable
done

# TP sweep for 7b
for tp in 1 2 4; do
  ezpz launch ... --config moe_7b \
    --parallelism.tensor_parallel_degree $tp \
    --activation_checkpoint.mode none --compile.no-enable
done

# LBS sweep for 2b
for lbs in 1 2 4 8; do
  ezpz launch ... --config moe_2b \
    --training.local_batch_size $lbs --compile.no-enable
done
```

## 4. Aurora Scaling Study

### Goal
Reproduce the Sunspot scaling study (2-128 nodes) on Aurora to measure
weak and strong scaling efficiency.

### Plan
1. **Configs**: Use the best configs from throughput benchmarks:
   - agpt_20b: TP=2, compile=on
   - agpt_80b_wide: TP=4, compile=on
   - moe_2b: no-compile
   - moe_10b_2b_sdpa: no-compile, AC=none

2. **Node counts**: 2, 4, 8, 16, 32, 64, 128 nodes

3. **Metrics**: TPS, TFLOPS, MFU, memory per node, weak scaling
   efficiency (TPS * N_nodes / TPS_1node)

4. **Script**: Extend `scripts/run_scaling_study.sh` or create a new
   one that submits jobs at each node count and collects results.

5. **Output**: Scaling curves (TPS vs nodes), efficiency table,
   report in `docs/experiments/agpt/aurora/` and `moe/aurora/`

### Challenges
- Need large allocations (64-128 nodes on capacity queue)
- May need to adjust TP/DP ratios at different scales
- MoE expert parallelism becomes relevant at higher node counts

## 5. Production Multi-Stage Training Plan

### Goal
Design configs for a production training run combining optimal LRs
(from LR finder) with optimal throughput configs (from benchmarks).

### Status

**agpt_20b -- IN PROGRESS**

- 2026-04-14: 2-node verify run completed 146 steps (loss 12.92 -> 10.50)
  before hitting 2h walltime. Config validated end-to-end.
- 2026-04-14: 512-node production job queued (Job 8436463), waiting for nodes.
- 2026-04-08: 1024-node attempt crashed at init (Job 8423904, exit 143).
- Full report: [20B Production (n512)](experiments/agpt/aurora/20260414-production-20b-n512.md)

### Recommended configs

**agpt_20b production (SUBMITTED):**
```
Model:      agpt_20b
TP:         1
Compile:    on (model + loss)
seq_len:    8192
LBS:        1
AC:         full
LR:         2.28e-5 (SophiaG, from LR finder)
Warmup:     200 steps
Decay:      cosine, min_lr_factor=0.1
Optimizer:  SophiaG
Tokens:     4.67T (olmo-mix-1124)
Steps:      92,859 (at GBS=6144 on 512 nodes)
Expected:   346 TPS per 2-node, ~17.3% MFU
Submit:     submit/aurora/submit_agpt_20b_n512.sh
```

**agpt_80b_wide production:**
```
Model:      agpt_80b_wide
TP:         4
Compile:    on
seq_len:    8192
LBS:        1
AC:         full
LR:         ~1e-4 (AdamW, extrapolated)
Warmup:     200 steps
Decay:      cosine, min_lr_factor=0.1
Optimizer:  AdamW
Expected:   68 TPS per 2-node, ~11.8% MFU
```

**moe_2b production:**
```
Model:      moe_2b
TP:         1 (FSDP only)
Compile:    off
seq_len:    4096
LBS:        2
AC:         full
LR:         8e-4
Expected:   6,600 TPS per 2-node, ~11.1% MFU
```

### Validation plan
1. Run each config for 1000 steps with the recommended LR
2. Verify loss convergence (should decrease monotonically after warmup)
3. Compare final loss with baseline (default LR 8e-4)
4. Run for 10,000 steps to verify stability
5. Enable checkpointing every 500 steps for fault tolerance

### Multi-stage training
For large-scale runs, consider:
- Stage 1: Short context (2048) for fast initial convergence
- Stage 2: Full context (8192) for long-range capability
- Stage 3: Cooldown with reduced LR for final quality

## 6. Debug 80B TP=2 Aurora OOM

### Problem
80B at TP=2 OOMs on Aurora with `aurora_frameworks-2025.3.1` at model
init or step 2. The same config worked on Sunspot (93.49% memory,
85 TPS). The error is `UR_RESULT_ERROR_OUT_OF_RESOURCES` from the
Level Zero driver, not a PyTorch OOM.

### Evidence
- All 80B variants (80B, 80B_alt, 80B_wide) OOM at TP=2
- 80B_wide gets through step 1 at 94.94% (60.75 GiB) but OOMs on
  step 2 (needs 5.25 GiB, only 5.18 GiB free — missed by 70 MiB)
- Even with cpu_offload, the driver OOMs
- Same code at same commit (c6ff706) OOMs — not a code regression
- Benchmark was run on Sunspot nodes, not Aurora nodes

### Debugging plan

1. **Verify on Sunspot**: Run 80B TP=2 on Sunspot to confirm it still
   works there. If it does, the issue is Aurora-specific.

2. **Check driver versions**: Compare Level Zero / GPU driver versions
   between Aurora and Sunspot. The memory allocator behavior may differ.

3. **Try `PYTORCH_XPU_ALLOC_CONF`**: Test memory allocator tuning:
   ```bash
   export PYTORCH_XPU_ALLOC_CONF="max_split_size_mb:512"
   export PYTORCH_XPU_ALLOC_CONF="expandable_segments:True"
   ```

4. **Memory profiling**: Enable `--profiling.enable_memory_snapshot`
   to get a detailed allocation trace. Compare Aurora vs Sunspot
   allocation patterns.

5. **Reduce memory by 70 MiB**: The 80B_wide TP=2 missed by 70 MiB.
   Try reducing overhead:
   - Disable W&B logging (`--metrics.no-enable_wandb`)
   - Reduce gradient clipping buffer
   - Try `torch.xpu.empty_cache()` before step 2

6. **File bug report**: If driver-level, file with Intel/ALCF support
   team with reproduction steps:
   - Node type, driver version, framework version
   - Exact command that fails
   - Memory allocation trace showing the 70 MiB gap

## 6. blendcorpus ↔ torchtitan parallelism aliasing

### Problem

`BlendCorpusDataLoader.__init__` constructs a `bc_cfg` for the
Megatron-style blendcorpus library by mapping torchtitan's
`parallel_dims` axes onto Megatron's parallelism knobs. Several of
these mappings are name-collisions, not semantic equivalents — and
each one is a silent foot-bullet the moment we turn the corresponding
parallelism on. Today they're all latent (we only run pure FSDP
without PP, CP, or real Megatron-TP), but they need to be cleaned up
before any of those features can be enabled with this dataloader.

Companion writeup with the full taxonomy:
[`docs/guides/known-bugs/blendcorpus-megatron-aliasing.md`](guides/known-bugs/blendcorpus-megatron-aliasing.md).

### Items to fix (all in `blendcorpus/blendcorpus_builder.py`)

1. **PP semantics mismatch.** Megatron's
   `pipeline_model_parallel_size` controls how the data sampler
   partitions samples across PP ranks (only the first PP stage gets
   data). torchtitan's PP doesn't work that way — the trainer decides
   which ranks read inputs. Passing `pp_degree=N` to the Megatron
   sampler will silently underfeed it by 1/N. Decide whether to
   collapse `pp_degree=1` always (since blendcorpus shouldn't drive
   PP partitioning at all in our flow) or to handle PP correctly.

2. **CP → SP aliasing.** torchtitan's CP shards the *sequence
   dimension* across ranks for attention. Megatron's
   `sequence_parallel_size` is a *TP variant* that shards activations
   inside transformer layers. Aliasing CP → SP is a name collision.
   With `cp > 1`, the global batch ends up divided by `cp_degree`
   instead of replicated across the CP group as CP wants. Should set
   `sequence_parallel_size=1` unconditionally and let the model handle
   CP at the layer level.

3. **`dp_world_size` double-source.** We pass `dp_world_size` from the
   torchtitan `batch_mesh` (DP × CP) but Megatron's sampler computes
   its own DP world from `world_size / (TP × PP × SP)`. With CP > 1
   and the SP aliasing above, they disagree and `requested_global_batch_size`
   becomes off-by-`cp_degree`. Fix follows from #2.

4. **Two parallel-state systems.** `bc_mpu.initialize_model_parallel`
   builds Megatron's *own* TP/PP/SP groups via
   `torch.distributed.new_group`, independent of torchtitan's
   `DeviceMesh`. Today inert because we use `comm_backend="standard"`
   and don't actually exercise Megatron's TP collectives, but it's a
   trap if anyone ever wires the Megatron-TP path up. Either
   (a) skip `initialize_model_parallel` entirely when
   `tp_degree == pp_degree == cp_degree == 1`, or (b) build Megatron's
   groups from torchtitan's existing meshes rather than duplicating.

5. **Cosmetic: hard-coded Megatron-only knobs.**
   - `data_impl="mmap"` — hard-coded inside `bc_cfg` instead of being
     a `Config` field. Promote to `Config` if anyone needs to override.
   - `dataloader_type: str = "single"` — Megatron-only sampler
     selector that doesn't map to anything torchtitan exposes. Either
     drop entirely or document why we keep `"single"`.
