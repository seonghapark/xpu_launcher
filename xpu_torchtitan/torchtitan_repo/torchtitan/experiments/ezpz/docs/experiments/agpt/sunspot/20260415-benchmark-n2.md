# agpt + MoE Full Benchmark -- Sunspot 2-node (2026-04-15)

## Environment

| Field        | Value                  |
|--------------|------------------------|
| Date         | 2026-04-15             |
| Commit       | d09d9708               |
| Machine      | Sunspot (x1921c7s2b0n0)|
| Job ID       | 12464406               |
| Nodes        | 2                      |
| Devices      | 24 (Intel Max 1550)    |
| Devices/Node | 12                     |
| Steps        | 10                     |
| Dataset      | blendcorpus (books)    |
| Backend      | xccl                   |
| Compile      | enabled                |

## Dense (agpt) Results

| Config          | TP | Memory            | TPS    | TFLOPS | MFU    | Wall (s) | W&B | Status |
|-----------------|----|-------------------|--------|--------|--------|----------|-----|--------|
| agpt_debugmodel | 1  | —                 | 34,573 | 7.98   | 2.68%  | 115 | [hne0eezr](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/hne0eezr) | OK |
| agpt_2b         | 1  | 46.80GiB (73.14%) | 5,489  | 61.41  | 20.59% | 94 | [ji5meq7e](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/ji5meq7e) | OK |
| agpt_7b         | 1  | —                 | —      | —      | —      | 25 | — | CRASH |
| agpt_8b         | 1  | —                 | —      | —      | —      | 25 | — | CRASH |
| agpt_20b        | 1  | 44.54GiB (69.61%) | 352    | 52.40  | 17.57% | 309 | [e55ktqmm](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/e55ktqmm) | OK |
| agpt_50b        | 1  | —                 | —      | —      | —      | 132 | [s56gq4j1](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/s56gq4j1) | OOM |
| agpt_80b        | 2  | —                 | —      | —      | —      | 68 | [copuhuvm](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/copuhuvm) | CRASH |
| agpt_80b_alt    | 2  | —                 | —      | —      | —      | 63 | [s0tbvyhs](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/s0tbvyhs) | CRASH |
| agpt_80b_wide   | 2  | —                 | —      | —      | —      | 66 | [pjcs5dau](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/pjcs5dau) | CRASH |
| agpt_80b_deep   | 2  | —                 | —      | —      | —      | 68 | [ye66kxw9](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/ye66kxw9) | CRASH |
| agpt_80b_deep_alt | 2 | —                | —      | —      | —      | 62 | [jtbreqcp](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/jtbreqcp) | CRASH |

## MoE Results

| Config          | TP | Memory            | TPS    | TFLOPS | MFU    | Wall (s) | W&B | Status |
|-----------------|----|-------------------|--------|--------|--------|----------|-----|--------|
| moe_debugmodel  | 1  | —                 | 10,901 | 18.76  | 6.29%  | 84 | [bn41qdds](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/bn41qdds) | OK |
| moe_500m        | 1  | —                 | 5,669  | 21.30  | 7.14%  | 91 | [162d143t](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/162d143t) | OK |
| moe_2b          | 1  | —                 | 3,515  | 25.54  | 8.56%  | 105 | [a9abh7n5](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/a9abh7n5) | OK |
| moe_4b          | 1  | —                 | 2,325  | 20.22  | 6.78%  | 111 | [cdbyfx5t](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/cdbyfx5t) | OK |
| moe_7b          | 1  | 33.40GiB (52.19%) | 1,066  | 19.27  | 6.46%  | 810 | [0371x6q5](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/0371x6q5) | OK |
| moe_10b_2b      | 1  | —                 | —      | —      | —      | 1813 | [2j4jan3j](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/2j4jan3j) | CRASH |
| moe_10b_2b_sdpa | 1  | 35.77GiB (55.90%) | 979    | 17.08  | 5.73%  | 882 | [23ejzbul](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/23ejzbul) | OK |

## Failure Analysis

### agpt_7b / agpt_8b — CRASH (blendcorpus cache race)

Both crashed in 25s during dataset initialization. The `run_benchmarks.sh` script
was cleaning `.cache/blendcorpus/*.npy` between runs, causing a race condition
where non-rank-0 processes attempted to read cache files before rank 0 finished
writing them on Lustre.

**Fix:** Disabled cache cleanup in `run_benchmarks.sh` (commit `2b0ed561`).

### agpt_50b — OOM

Exceeds 64GB per tile at TP=1. Requires TP=2+ at 2-node scale.

### agpt_80b (all 5 variants) — CRASH (Inductor compile bug)

All 80B variants crashed with:
```
InductorError: PendingUnbackedSymbolNotFound: Pending unbacked symbols {zuf0, zuf1}
```

**Root cause:** Upstream PR #2741 ("Enable per-layer compile with or without MoE")
consolidated `apply_compile_dense` and `apply_compile_sparse` into a single
`apply_compile` that unconditionally sets `torch._dynamo.config.capture_scalar_outputs = True`.
This is needed for MoE dynamic shapes but breaks the separately-compiled loss
function when `loss_parallel` + `ignore_index` in `F.cross_entropy` produce
unbacked symbols.

**Fix:** Reset `capture_scalar_outputs = False` after `apply_compile` in the
dense agpt parallelize path (commit `e8cbb8ef`).

**Workaround:** `--compile.components=model` (skip loss compilation).

### moe_10b_2b — CRASH (Inductor + MoE dynamic shapes)

Same `PendingUnbackedSymbolNotFound` error, but in this case `capture_scalar_outputs`
is legitimately needed for MoE routing. The `moe_10b_2b_sdpa` variant (which uses
SDPA attention instead of FlexAttention) succeeded, suggesting the crash may be
related to the attention path rather than MoE routing alone.

## Comparison with Previous Run (2026-04-13)

| Config    | 04-13 TPS | 04-15 TPS | 04-13 MFU | 04-15 MFU | Notes |
|-----------|-----------|-----------|-----------|-----------|-------|
| debugmodel| 49,103    | 34,573    | 3.80%     | 2.68%     | Different LBS (8 vs default) |
| agpt_2b   | 5,609     | 5,489     | 21.05%    | 20.59%    | Within noise |
| agpt_20b  | 351       | 352       | 17.53%    | 17.57%    | Consistent |
| 80b_deep  | 83 TPS    | CRASH     | 15.20%    | —         | Compile regression from upstream merge |

The 80B regression is entirely due to the `capture_scalar_outputs` change in the
upstream compile consolidation. The fix (commit `e8cbb8ef`) should restore 80B
functionality.

## Logs

Report: `outputs/benchmarks/20260415_150745/report.md`
Logs: `outputs/benchmarks/20260415_150745/`
