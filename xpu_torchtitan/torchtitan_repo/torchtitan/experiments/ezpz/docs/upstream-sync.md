# Upstream Sync Log

Tracks changes merged from `upstream/main` (pytorch/torchtitan) into the `ezpz`
branch, and any modifications required to keep `experiments/ezpz/{agpt,moe,qwen3}`
compatible.

## How to use this document

After each `git merge upstream/main`, check if the incoming commits touch:

1. **`models/llama3/`** — replay changes onto `experiments/ezpz/agpt/`
2. **`models/deepseek_v3/`** — replay changes onto `experiments/ezpz/moe/`
3. **`models/qwen3/`** — replay changes onto `experiments/ezpz/qwen3/`
4. **`models/common/`** — check if ezpz models depend on changed APIs
5. **`distributed/`** — check if ezpz trainer or parallelize files use removed/renamed APIs
6. **`trainer.py`** — check if ezpz trainer mirrors the same code path

Add an entry below with the date, upstream commits, what changed, and what
was required in ezpz.

---

## 2026-06-30 — 63rd sync (13 commits, `390ea37cc..upstream/main`)

Merged as `11356e218` in worktree `.worktrees/ezpz-63rd-sync`. **No replays
required** — nothing touches `models/llama3/` or `models/deepseek_v3/`, so
`agpt/` and `moe/` need no replay. Post-merge `git diff 006643ca8 HEAD --
experiments/ezpz/agpt/ experiments/ezpz/moe/` is EMPTY (our model code
untouched).

### One merge conflict (RL, resolved)
`experiments/rl/actors/generator.py`: our HEAD only dropped a TODO comment;
upstream wrapped the `ts.get_state_dict` call in a `spmd_types` if/else
(new `_get_spmd_state_dict` path). Took upstream — its `else` branch preserves
our exact call, so upstream is a strict superset. RL code, not on any path we
run.

### Upstream commits
| Commit | Title | ezpz impact |
|--------|-------|-------------|
| `952427842` | [spmd_types] FLUX enablement (#3823) | None (flux/ + spmd_types-guarded loss.py). |
| `d9ebce1bd` | Pin MONARCH_ACTOR_QUEUE_DISPATCH to 0 (#3832) | None. RL. |
| `0052a3870` | Set drop_zero_std_reward_groups=False for debug configs (#3802) | None. RL. |
| `9e4a8512b` | [rl] fix the breakable cudagraph env var (#3829) | None. RL. |
| `02f00eec3` | [RL] spmd_types: fix trainer logprob grad, generator TP all-reduce, FusedSwiGLU sync (#3827) | None (spmd_types + RL). |
| `a06d4e757` | Revert "[RL] spmd_types: fix trainer logprob gradient..." (#3826) | None. RL. |
| `5aefc0229` | [RL] spmd_types: fix trainer logprob gradient... (#3822) | None. RL. |
| `c2a3293fb` | [rl] Remove buffer from weight sync dtype conversion (#3825) | None. RL. |
| `3e819667f` | [rl] Enable batch-invariant using FSDP mixed precision (#2932) | None. RL. |
| `ac240f926` | Fix transformers_modeling_backend for pretrained dense models (#3772) | None (eval-backend path we don't use in training). |
| `0f5dfc1e9` | [rl] add perf run config (#3778) | None. RL. |
| `756213e15` | Upgrade DeepEP to DeepEP v2 APIs, cudagraphable (#3808) | **Touches `distributed/deepep/` + `models/common/token_dispatcher.py`, which ezpz `moe/token_dispatcher.py` imports.** The one change worth smoking on moe. |
| `0e2565100` | Fix RL spmd_types generator weight sync (#3804) | None. RL. |

### Shared files changed (none break our DTensor-backend agpt/moe)
- `components/loss.py` (+23): all under `spmd.no_typecheck()` /
  `get_spmd_backend()=="spmd_types"` guards — no-op on our default backend.
- `distributed/deepep/*` + `models/common/token_dispatcher.py` (+690): DeepEP
  v2 upgrade — the moe smoke exercises this.
- `models/{flux,gpt_oss,qwen3}/`: other families, not run here.

### Validation — smoke-passed (Sunspot, 2026-07-01)
Both 2N smokes run from the sync worktree so `$PBS_O_WORKDIR` uses the merged
code. Landed to `origin/ezpz` as `cf99e127e` after these passed.

- **agpt 2B (job 12469960): PASS.** Clean 10-step train, final loss 8.35,
  0 NaN — the regression check is green (agpt was untouched by the merge, and
  it still trains).
- **moe debugmodel (job 12469961): DeepEP-v2 path exercised OK.** Built the
  mesh + MoE model + expert backend (past the token-dispatcher code the merge
  changed), logged the expected pre-existing XPU note `torch._grouped_mm
  requires SM90+ CUDA; falling back to for_loop expert backend`, then OOM'd
  downstream on a compute kernel (`RuntimeError: level_zero backend failed
  with error: 40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)`). That is a **known XPU
  resource limit at debugmodel's default 2N settings, NOT a merge regression**
  — the changed DeepEP-v2 dispatcher ran without error before the OOM.
- **Conclusion: merge is safe.** agpt trains clean; the one ezpz-relevant
  upstream change (DeepEP v2) runs. Merge landed.
- Note: running jobs from a git worktree needs `.venv`, `.venv.tar.gz`, and
  `assets/hf` symlinked in (worktrees carry only tracked files) — this cost a
  few false-start smokes (missing tokenizer, missing venv) before the real
  validation. Config `ImportError: Cannot import config_registry` is a generic
  mask; the real cause was the missing tokenizer path.

## 2026-06-28 — 62nd sync (3 commits, `0e886617e..upstream/main`)

Merged clean (no conflicts) as `fc4e1ae84` in worktree `ezpz-62nd-sync`.
**No replays required.** All 3 commits touch only `experiments/rl/` (the
GRPO/RL experiment), which ezpz does not import from; our entire model /
parallelize / config / trainer surface is byte-identical pre- and
post-merge (`git diff 0cacad95d HEAD -- experiments/ezpz/ config/ models/
distributed/` is empty).

### Upstream commits

| Commit | Title | ezpz impact |
|---|---|---|
| `390ea37cc` | [rl] Overlap trainer->generator weight sync with next training step (#3810) | None. `experiments/rl/` only. |
| `ce4f4cc4d` | Enable chunked loss for Search R1 (#3813) | None. `experiments/rl/examples/search_r1/` only. |
| `abaf11c5a` | [rl] Fix generator initialization race condition (#3809) | None. `experiments/rl/` only. |

### Validation

Import probe (the sync_smoke phase-1 surface: agpt/moe parallelize +
config_registry + activation_checkpoint + trainer) confirmed unchanged --
no symbol-table diff to check. Skipped the distributed train smoke: the
merge changes zero bytes of any code path the smoke exercises, so it would
only re-confirm the 61st-sync baseline. RL-only sync, lowest-risk class.

---

## 2026-06-27 — 61st sync (7 commits, `0b083d3ee..upstream/main`)

Merged clean (no conflicts) as `5245d470e` in worktree
`ezpz-61st-sync`. **One replay required** (`600e79f63`): the #3779
`ChunkedCELoss -> ChunkedLossWrapper` rename. (My first-pass "no replays"
read was WRONG -- the runtime smoke caught it as `import_failed`; an audit
grep with a bad glob had hidden the two ezpz usages. The smoke gate
working as intended.)

### Upstream commits

| Commit | Title | ezpz impact |
|---|---|---|
| `0e886617e` | Fix weight-tying corrupting lm_head.weight in HF checkpoint (#3764) | **Relevant to eval, benign.** Touches `llama3/state_dict_adapter.py` (which agpt imports as `Llama3StateDictAdapter` for DCP->HF convert) + `distributed/utils.py`. The fix corrects a real safetensors corruption on tied embeddings; agpt uses the adapter unchanged (symbol intact, import OK). Improves eval/convert correctness, no replay. |
| `3c9940675` | Fix fused qkv load weights + init when FSDP > n_kv_heads (#3807) | **No-op for ezpz.** Only the `FusedQKVLinear` path in `common/attention.py`/`config_utils.py`. agpt keeps `fuse_qkv=False` (4 callsites), so it never constructs FusedQKV. SDPA forward unchanged. No replay. |
| `9530a874d` | [rl] grpo chunked loss (#3779) | **REPLAY (`600e79f63`).** Renames `ChunkedCELoss` -> `ChunkedLossWrapper` in `components/loss.py` (no back-compat alias). ezpz uses it in TWO files: `agpt/config_registry.py` (import + agpt_{2b,20b,80b}_chunkedce builders) and `trainer.py` (import + `isinstance(loss_fn, ...)` lm_head-plumbing gate). Replayed the rename; API-compatible (`.Config(num_chunks=8)`, `set_lm_head`, `_skip_lm_head` all unchanged). moe imports `Decoder`/`TransformerBlock` from `common/decoder.py`, fine. |
| `c8abb170a` | expand model integration tests to full_dtensor, spmd_types backends (#3740) | None -- CI test matrix only. |
| `fcbcb6d53` | Enable RL spmd backend selection (#3803) | None -- `distributed/utils.py` spmd-backend knob + experiments/{rl,forge,graph_trainer,torchft}; ezpz uses spmd_backend=default. |
| `2e962a153` | [rl] cudagraph capture size + batched-tokens knobs (#3806) | None -- experiments/rl only. |
| `574502c80` | Add DPRequestRouter, use in generator (#3768) | None -- experiments/rl generator only. |

### Replays required (3 -- all spmd_types-series fallout)

The runtime smoke caught these one at a time (an import probe alone misses
runtime attribute access; my first-pass static audit also missed them --
grep glob bug). All three are plumbing renames/API shifts with **zero
numerical effect** (final losses match the 60th-sync baseline exactly):

1. **`600e79f63`** -- `ChunkedCELoss -> ChunkedLossWrapper` (#3779), no
   back-compat alias. ezpz used it in `agpt/config_registry.py`
   (agpt_{2b,20b,80b}_chunkedce) + `trainer.py` (isinstance gate).
   API-compatible (`.Config(num_chunks=8)`, `set_lm_head`).
2. **`5701faef8`** -- `dist_utils.get_train_context -> get_spmd_context`
   (+ `spmd_typechecking` kwarg). ezpz `trainer.py` train_context build.
   Mirrored upstream; inert at our `spmd_backend=default`.
3. **`3cf8c9343`** -- upstream `set_pg_timeouts` switched from
   `distributed_c10d._set_pg_timeout` to `torch.distributed.set_timeout`,
   which our **torch 2.13 does not have**. The ezpz
   `_set_pg_timeouts_xpu_aware` shim no longer delegates to the upstream
   helper; it inlines the pre-sync `_set_pg_timeout` path (present in 2.13)
   before its XCCL patch. (Not an ezpz-rename -- a torch-version skew the
   sync introduced; scanned all ezpz `torch.distributed.*` calls, this was
   the only 2.13-missing one.)

### Verification

- Clean merge, no conflicts (`5245d470e`, worktree `ezpz-61st-sync`).
- **Runtime smoke PASSED** (`sync_smoke.sh`, job 12469757, own select=1)
  -> `VERDICT: ok` after the 3 replays:
  - `agpt_debugmodel` TP=1 rc=0 (loss -> 10.673)
  - `agpt_debugmodel` **TP=2** rc=0 (loss -> 10.662, mem halved)
  - `moe_debugmodel` rc=0 (loss -> 12.368)
  All three match the 60th-sync baseline exactly -> the replays are pure
  plumbing, no numerical change. (Smokes 12469753/755/756 failed on the 3
  issues above before the fixes -- the gate working as intended.)

---

## 2026-06-26 — 60th sync (23 commits, `daa9d7453..upstream/main`)

Merged clean (no conflicts) as `0b083d3ee` in worktree
`ezpz-60th-sync`. **No code replays required**, but this sync adds a
**new pip dependency**: `spmd_types==0.2.1`. The `[spmd_types]` commit
series adds a hard top-level `import spmd_types as spmd` to
`common/attention.py`, `common/moe.py`, and ~20 other core files that
ezpz agpt/moe import. The module is a pure-Python wheel but declares
`torch>=2.10`, so it was installed `--no-deps --no-cache` to protect the
XPU torch build (verified torch still `2.13.0.dev...+xpu`, untouched).

All the new spmd-typing logic is gated behind
`get_spmd_backend() == "spmd_types"` (default is `"default"`), so it is a
**functional no-op for ezpz** -- we just need the module importable so
the hard imports resolve. agpt subclasses `ScaledDotProductAttention`
(forward signature **unchanged** by this merge -- the new
`@spmd.local_map` decorator landed on `VarlenAttention.forward`, which
agpt does not use); moe imports `MoE`/`GroupedExperts`/
`TokenChoiceTopKRouter`/token_dispatcher, whose new spmd typecheck blocks
are all `spmd_backend`-gated. Import smoke (login node, merged code +
installed dep): agpt, common.attention, ezpz.moe all import OK.

### Upstream commits

| Commit | Title | ezpz impact |
|---|---|---|
| `0a4aa4542` | [spmd_types] llama3 enablement (#3763) | **Dep + no-op.** Adds `import spmd_types` to `common/attention.py` + `common/rope.py` and `@spmd.local_map` on `VarlenAttention.forward`. agpt subclasses `ScaledDotProductAttention` (untouched); not the varlen path. Gated by `spmd_backend`. No replay. |
| `9b91827ab` | [spmd_types] qwen3 enablement (#3655) | Same surface (`common/attention.py` + `rope.py`). ezpz has no live qwen3 fork. No replay. |
| `622c1acaa` | [spmd_types] dsv3 enablement (#3673) | `deepseek_v3/{model,parallelize,sharding}.py`. moe replays from dsv3 but the changes are spmd-typecheck annotations gated by `spmd_backend`; inert at `default`. No replay. |
| `7144431d6` | [spmd_types] GPT-OSS enablement (#3690) | `gpt_oss/` only; ezpz has no gpt_oss fork. None. |
| `565cb810c` | [spmd_types] fix MoE sequence-sharded combine (#3664) | `common/moe.py` + `moe_sharding.py`. ezpz/moe uses `sp_size=1` so the SP-combine path is never live (same as the 54th-sync MoE SP fix). Inert. No replay. |
| `b052f36fe` | [spmd_types] MoE sparse mesh transitions (#3654) | `common/moe.py` + `token_dispatcher.py`: adds `maybe_set_sparse_mesh()` context (no-op unless spmd active). moe imports through it; inert at `default`. No replay. |
| `fbf79bcc5` | [spmd_types] helion rope & fused_swiglu rules (#3741) | `overrides/{helion_rope,fused_swiglu}.py` typecheck rules. ezpz uses neither override. None. |
| `c86821ecd` | use funcol for custom LP autograd (#3793) | `distributed/` LP autograd; ezpz doesn't use loss-parallel custom autograd. None. |
| `40f1a553b` | [graph_trainer] EP overlap scheduling pass (#3328) | `common/token_dispatcher.py` + `experiments/graph_trainer/`. moe imports token_dispatcher; the change is graph-trainer scheduling, inert outside graph_trainer. No replay. |
| `1c0fde2b6` | [graph_trainer] graph EP chunking pass (#3325) | `experiments/graph_trainer/` only. None. |
| `616182011` | [graph_trainer] EP overlap eager chunking scaffolding (#3363) | graph_trainer only. None. |
| `a37b9f98d` | [graph_trainer] marked symbolic input dims in tracing (#3362) | `common/rope.py` symbolic-dim hooks; gated/tracing-only, agpt rope path unchanged at runtime. No replay. |
| `f7a8e22e3` | [graph_trainer] Fix failing CI tests (#3791) | CI/graph_trainer only. None. |
| `966983a73` | [rl] Async RL loop (#3642) | `experiments/rl/` (upstream Monarch path). ezpz has its own XPU rl port; upstream rl flows in but isn't on our path. None. |
| `fcde6db8e` | Add DP fan-in to generator + DP routing (#3765) | upstream rl generator. None for ezpz. |
| `baa6a6446` | [rl] Fix RL load weights: fused qkv load hooks bypassed (#3794) | upstream rl. None. |
| `b67239bc9` | [rl] annotate varlen Attention cudagraph capability (#3776) | upstream rl + varlen; ezpz doesn't use the rl varlen path. None. |
| `3e477ead0` | [rl] Fix model max seq length / batcher check (#3796) | upstream rl. None. |
| `7c951af4f` | [rl] vllm All-reduce patch in generator (#3759) | upstream rl/vllm. None (ezpz rl uses its own TRL vllm-serve path). |
| `457e55d2b` | keep all_to_all_single for compiled token_dispatcher (#3797) | `common/token_dispatcher.py`. moe imports it; restores `all_to_all_single` on the compiled path (mirrors our own `82100fd67` SAC save-list intent). Benign/beneficial. No replay. |
| `2ce277326` | [flex attention] pass score_mod to flex_attn (#3789) | `common/attention.py` FlexAttention. agpt's `SoftcappedFlexAttention` subclasses FlexAttention + uses score_mod -- **verify in smoke** that the score_mod plumbing change is compatible (XPU has no flex backend, so the live agpt path is SDPA; flex is debug-only). No replay expected. |
| `16de1d625` | float8: enable on ROCm gfx942+ (#3760) | `quantization/float8`. ezpz XPU doesn't use float8. None. |
| `1dbdbc672` | Skip torchcomms install in feature CI (#3792) | CI only. None. |

### Verification

- Clean merge, no conflicts (`0b083d3ee`, worktree `ezpz-60th-sync`).
- `spmd_types==0.2.1` installed `--no-deps --no-cache`; torch unchanged
  (`2.13.0.dev20260519+xpu`, `__init__.py` mtime pre-merge).
- Import smoke (login node, merged code): agpt, common.attention,
  ezpz.moe all import OK.
- **Runtime smoke PASSED** (`sync_smoke.sh`, job 12469667, own select=1)
  -> `VERDICT: ok`. All 3 default configs trained 2 deterministic steps
  clean:
  - `agpt_debugmodel` TP=1 rc=0 (loss 10.839 -> 10.673) -- identical to
    the 59th-sync baseline, so the merge is numerically consistent.
  - `agpt_debugmodel` **TP=2** rc=0 (loss 10.839 -> 10.662, mem halved
    to 13.4%) -- confirms the `[spmd_types]` attention/parallelize
    refactor is safe on the ezpz path under TP>1 (`spmd_backend=default`,
    typechecking off), the surface the 57th sync regressed at.
  - `moe_debugmodel` rc=0 (loss 12.910 -> 12.368).
  - First attempt (12469660) hit the known blendcorpus cold-cache
    build-then-load race at TP=2 (`EOFError` in `_build_index_mappings`)
    -- NOT a sync issue (TP=2 cleared the entire spmd/parallelize path
    and only tripped in dataloader init); the cold build left the cache
    warm and 12469667 passed. (Smoke ran from worktree `ezpz-60th-sync`
    with `.venv`/`.venv.tar.gz`/`assets/hf` symlinked from the main
    checkout.)

---

## 2026-06-25 — 59th sync (10 commits, `395833a46..upstream/main`)

Merged clean (no conflicts) as `daa9d7453`. **No code replays required**;
the shared `common/` + `protocols/` changes flow into ezpz agpt/moe
automatically. Config-build smoke passed: `agpt_2b`, `agpt_2b_real`,
`agpt_80b`, `moe_10b_2b`, `moe_10b_2b_sdpa` all build through the merged
paths; both ezpz models import.

### Upstream commits

| Commit | Title | ezpz impact |
|---|---|---|
| `3396a4b37` | Enable Fused qkv by default... (#3714) | **Auto via common/.** Flips llama3's `fuse_qkv` default False->True and adds `_fused_qkv_param_init` + a contiguous/`local_map`-wrapped `FusedQKVLinear.forward` in `common/attention.py` + `common/config_utils.py`. ezpz/agpt keeps its OWN `fuse_qkv=False` default (`agpt/__init__.py`), so the 2B/80B production chains are unchanged -- **zero checkpoint risk**. Attention forward sig unchanged (q_BLNH/k_BLNH/v_BLNH contract intact). moe uses MLA, unaffected. No replay. |
| `b3213ef8c` | Fix LoRA freezing for non-linear modules (#3456) | **Protocol change, benign for ezpz.** `ModelConfigConverter.convert()` now returns `Module.Config` (was `None`); `ModelSpec.traverse()` gains `recurse=`. ezpz/agpt has no converters. ezpz/moe calls `q.build().convert(config)` for side-effect (ignores return) at `moe/__init__.py:1188` -- still works because the Float8/quant converters mutate in place. LATENT GAP: if a converter ever returns a *replacement* config, moe would drop it; pre-existing pattern, not broken by this merge. moe quant config builds OK. No replay; optional hardening noted. |
| `73b64151e` | Disable Varlen + PP support (#3777) | **No-op for ezpz.** Adds a `ValueError` in `common/decoder.py` if PP>1 AND VarlenAttention. No ezpz config combines varlen+PP (agpt has a `debugmodel_varlen_attn` flavor but never with PP). No replay. |
| `7083a8055` | [spmd] normalize partial->replicate on size-1 mesh axes (#3762) | **Free fix for ezpz/moe.** `protocols/sharding.py` `resolve_placements` now normalizes `Partial` (not just `Shard`) to `Replicate` on size-1 axes -- fixes the MoE routed-expert `Partial(sum)` mismatch at `dp_shard=1`. ezpz/moe benefits automatically. No replay. |
| `7e7a234cd` | scaled bias rowwise linear (#3781) | None -- `quantization/{float8,mx}.py` rowwise path; ezpz doesn't use it. |
| `854898a36` | [graph_trainer] Fix DSv3 bucketing order... (#3770) | None -- `experiments/graph_trainer/` only. |
| `e20fe8a41` | [minAsyncMoE] Fix active swiglu int32 overflow (#3769) | None -- `distributed/minimal_async_ep/kernels.py`; ezpz/moe doesn't use minAsyncMoE. |
| `f713a0254` | [minAsyncMoE] Make Triton metadata contiguous (#3771) | None -- same minAsyncMoE kernels path. |
| `4054e01c4` | [Flux] Use CI local dataset for Flux validation (#3715) | None -- flux only. |
| `d4f3687cc` | Disable qwen3_5 for the TorchTitan CI (#3774) | None -- CI/qwen3_5 only; ezpz has no qwen3_5 fork. |

### Verification

- Clean merge, no conflicts (`daa9d7453`).
- Config-build smoke (login node): agpt_2b / agpt_2b_real / agpt_80b /
  moe_10b_2b / moe_10b_2b_sdpa all build; agpt + moe import OK.
- **Post-sync runtime smoke PASSED** (`sync_smoke.sh`, job 12469577,
  own select=1): import probe OK; all 3 default configs trained 2
  deterministic steps clean -> `VERDICT: ok`:
  - `agpt_debugmodel` TP=1 rc=0 (loss 10.84 -> 10.67)
  - `agpt_debugmodel` **TP=2** rc=0 (loss 10.84 -> 10.66, mem halved to
    13.4%) -- confirms the fused-QKV `common/attention.py` refactor is
    safe on the ezpz **non-fused** path under TP>1; the local_map /
    `in_dst_shardings` contract holds (this is the exact surface the 57th
    sync's q_BLNH rename regressed at TP>1).
  - `moe_debugmodel` rc=0 (loss 12.91 -> 12.37) -- exercises the
    `convert()` protocol change + the SPMD Partial-size-1 fix; no
    `Partial`/`in_src_shardings` mismatch. (One benign `_redistribute`
    perf warning on the moe norm -- upstream behavior, not a regression.)
- The 80B TP=4 production path (agpt_80b, fuse_qkv=False, compile=OFF) is
  unaffected by the fused-QKV default flip.

---

## 2026-06-24 — 58th sync (1 commit, `c6c2fb2c5..395833a46`)

Merged clean (no conflicts) as `2c0daba85`. No replays needed.

### Upstream commit

| Commit | Title | ezpz impact |
|---|---|---|
| `395833a46` | `Make generator actor aware of DP rank layout (#3743)` | None -- touches only `experiments/rl/actors/generator.py` + `rl/tests/test_engine_loop.py`. Fixes a v2-runner DP collective-mismatch hang (DP0 returns early when `total_num_scheduled_tokens == 0`, colliding with DP1's collective). ezpz/rl has its own GRPO loop and does not fork upstream `rl/actors/generator.py`, so nothing to replay. Relevant background for the Monarch port (`docs/rl/2026-06-14_monarch-torch213-deep-dive.md`) if/when we wire the upstream generator. |

### Verification

Static: ezpz imports clean post-merge (`agpt_debugmodel` + `moe_debugmodel`
build OK). No model/parallelize/config surface touched, so the
57th-sync smoke (job 12469469) coverage still holds.

---

## 2026-06-14 — 57th sync (59 commits, `7b579adde..c6c2fb2c5`)

Merged clean (no conflicts) as `1f288f2e7`. Largest sync in weeks --
6 commits touch `models/llama3/` or `models/deepseek_v3/` and need
replay onto `ezpz/agpt/` / `ezpz/moe/`. Replays are NOT included in
this merge commit; they will land as follow-up commits with smoke
tests per item.

### Upstream commits that need replay

| Upstream commit | Title | Source dir | ezpz target |
|---|---|---|---|
| `70dd94551` | `FusedQKVLinear checkpoint interop via state_dict hooks (#3656)` | `llama3/state_dict_adapter.py` | `ezpz/agpt/` -- no `state_dict_adapter.py` yet; check whether agpt needs FusedQKVLinear or whether we keep stock QKVLinear. |
| `b3b60dabf` | `delete --disable_loss_parallel flag (#3694)` | `llama3/{model,sharding}.py` + `deepseek_v3/{model,sharding}.py` | **Both forks have the same call site** (`model.py:49` / `model.py:313` -- `loss_parallel=not parallelism.disable_loss_parallel`). Need to remove the kwarg from both call sites AND from `agpt/sharding.py:39` + `moe/sharding.py:63` signatures. TP=on now implies LP=on for trainers via `tp_gather_logits=False`. |
| `c5d93d109` | `Refactor activation checkpointing into a policy class hierarchy (#3674)` | `llama3/{parallelize,config_registry}.py` + `deepseek_v3/{parallelize,config_registry}.py` | **Significant API change.** `ActivationCheckpointConfig(mode="...")` is replaced with a `Configurable` policy hierarchy. Every ezpz `agpt_*` / `moe_*` config that sets `ac_config` needs porting. The `apply_ac()` callers in `agpt/parallelize.py` + `moe/parallelize.py` need updating too. |
| `581f175dc` | `fuse swiglu using silu_and_mul kernel (#3712)` | `deepseek_v3/config_registry.py` | Check whether ezpz/moe MoE configs opt into `FusedGroupExperts` w13 fusion + `silu_and_mul` override. PR14 fork at `ezpz/moe/experts.py` may need parity. |
| `aa1d37414` | `Add FusedGroupedExperts override and offset-aware kernels (#3659)` | `deepseek_v3/config_registry.py` | Same as above -- offset-aware SwiGLU kernels for syncless EP (MinimalAsyncEP / HybridEP). Not currently used by ezpz/moe production. |
| `cd8950ba7` | `[MoE] Remove unused score_before_experts dispatcher flag (#3663)` | `deepseek_v3/__init__.py` | Trivial -- `ezpz/moe/__init__.py` builds the token dispatcher the same way; need to drop the `score_before_experts` kwarg from the dispatcher construction call (if present). |

### Replay outcomes (2026-06-24)

| Commit | Status | Notes |
|---|---|---|
| `b3b60dabf` (disable_loss_parallel) | **DONE** `fa6f0681e` | Dropped the kwarg from `agpt/{model,sharding}.py`, `moe/{model,sharding}.py`, and `trainer.py` (`get_train_context` no longer takes `enable_loss_parallel`). |
| `c5d93d109` (AC policy hierarchy) | **DONE** `bb38b95e1` | `ActivationCheckpointConfig(mode=...)` -> `FullAC`/`SelectiveAC`/`None`; `apply_ac()` -> `ac_config.build(...).apply(model)`. Rewrote the `ezpz/moe/activation_checkpoint.py` `_get_save_ops` monkey-patch as a clean `MoeSelectiveAC(SelectiveAC)` subclass (the extension point the refactor was designed for). Also fixed two `cfg.activation_checkpoint.mode =` mutation sites in the `agpt()`/`moe()` config-registry wrappers (slots Config has no `mode` attr). |
| `cd8950ba7` (score_before_experts) | **NO REPLAY (deliberate divergence)** | Upstream removed the flag because it was *dead* in their dispatcher. The ezpz fork's `moe/token_dispatcher.py` genuinely branches on it (`_local_reorder`: scores applied before experts when True; `combine`: scores applied after when False -- two live, mutually-exclusive code paths). ezpz's `make_ezpz_token_dispatcher_config` owns its own `score_before_experts` param and the dispatcher its own `Config` field; ezpz imports only `get_attention_config` + `make_ffn_config` from `config_utils` (NOT the `make_token_dispatcher_config` whose signature lost the kwarg), so nothing breaks. Keeping the field. Verified `moe_debugmodel` + `moe_debugmodel_ep` build clean. |
| `70dd94551` (FusedQKVLinear state_dict hooks) | **NO REPLAY (inherited via import, inactive)** | ezpz/agpt imports `Llama3StateDictAdapter` directly from upstream (no fork) -- it gets the change for free. ezpz/agpt's model uses stock `QKVLinear`, never `FusedQKVLinear.Config`, so the new `self.fuse_qkv` branch in the adapter is always False. Nothing to port. |
| `581f175dc` (fuse swiglu via silu_and_mul) | **NO REPLAY (opt-in, unused)** | Touches `deepseek_v3/config_registry.py` to opt into `FusedGroupExperts` w13 fusion + `silu_and_mul` override. ezpz/moe uses `EzpzGroupedExperts(GroupedExperts)` with a `compute_backend` selector and never references `FusedGroupExperts` / `silu_and_mul`. Not enabled by any ezpz config. |
| `aa1d37414` (FusedGroupedExperts offset-aware) | **NO REPLAY (opt-in, unused)** | Offset-aware SwiGLU kernels for syncless EP (MinimalAsyncEP / HybridEP). ezpz/moe doesn't enable these EP backends in production; zero references in `ezpz/moe/`. |

Smoke verification (job `12469466`, `venvs/rl-monarch-torch213` py3.13.6
+ torch 2.13, sunspot 1N, `--debug.seed 42 --debug.deterministic`):
- `agpt_debugmodel` (FullAC + LP): loss `10.83863 -> 10.67256`, rc=0.
  Step-1 loss bitwise identical across two runs (`12469465`, `12469466`).
- `moe_debugmodel` (MoeSelectiveAC + LP, seq_len=512/lbs=1 to fit 1
  tile): loss `12.90956 -> 12.36751`, rc=0.

#### Follow-up: TP>1 attention local_map regression (caught at 64N, fixed)

The two debugmodel smokes above run at **TP=1**, which does NOT exercise
`model.parallelize()`'s local_map wrapping. A 64N `agpt_80b` (TP=2)
functionality run (job `12469471`) then crashed at init on **every
rank**:

```
AssertionError: XPUScaledDotProductAttention: local_map is set but
in_dst_shardings is missing entries for: ['q', 'k', 'v']
```

Root cause: the 57th sync adopted the Noam Shazeer shape-suffix naming
convention upstream (see CLAUDE.md "Shape-suffix tensor names"). Upstream
`ScaledDotProductAttention.forward` renamed its positional args to
`q_BLNH/k_BLNH/v_BLNH`, and `set_gqa_inner_attention_local_map` now keys
`in_dst_shardings` by those suffixed names. The local_map contract check
(`protocols/module.py:_maybe_wrap_local_map`) matches `in_dst_shardings`
keys against the wrapped forward's **positional-arg names** -- and the
ezpz attention forks (`agpt/__init__.py` `EzpzScaledDotProductAttention`
+ `SoftcappedFlexAttention`, `moe/__init__.py`
`EzpzScaledDotProductAttention`) still used bare `q/k/v`. The mismatch
only asserts under TP>1, so the TP=1 debugmodel smokes passed.

Fix (commit pending): renamed the positional params of all three ezpz
attention `forward`s to `q_BLNH/k_BLNH/v_BLNH` (transposing into local
`q/k/v` in the body). Verified with a TP=2 smoke (job `12469472`,
`smoke_agpt_tp2.sh`, agpt_2b 2N) before relaunching 64N.

**Lesson: post-sync smokes must include a TP>1 config.** Pure-FSDP
(TP=1) debugmodels miss the entire local_map / sharding-contract
surface. `sync_smoke.sh` should grow a TP=2 entry.

Venv note: `venvs/rl-monarch-torch213` needed `sh`, editable `ezpz`
(`-e ../ezpz`, the installed 0.19.0 wheel was missing `get_timestamp`),
and editable `blendcorpus` (`-e deps/blendcorpus`) added (all `--no-deps`,
torch untouched) before the agpt/moe train path would run. The repo-root
`.venv` is currently py3.14 (torchtitan import fails on
`importlib.metadata`) and `.venv.tar.gz` is stale.

### Other notable upstream commits (no replay needed)

| Commit | Title | Why no replay |
|---|---|---|
| `c6c2fb2c5` | `[rl] Checkpoint + resume the RL training loop (#3735)` | `experiments/rl/` only; ezpz/rl has its own GRPO loop. May want to study for our Monarch port though. |
| `df81df57a` | `Switch to v1 vllm model runner (#3749)` | `experiments/rl/actors/generator.py`; same as above. |
| `739bb9a9e` | `bring back broadcasted rope branch, spmd.local_map (#3750)` | `models/common/rope.py`; ezpz consumes upstream RoPE unchanged. |
| `46c797105` | `Raise GPT-OSS Dynamo recompile limit (#3737)` | `models/gpt_oss/`; ezpz doesn't use gpt_oss. |
| `b934b0c2b` | `Exclude ROCm from has_cuda_capability (#3738)` | `tools/utils.py`; ezpz already monkey-patches this on XPU (see `ezpz/rl/xpu_overrides.py:has_xpu_kernels`). Upstream's tweak is CUDA/ROCm-only, leaves the XPU branch alone. |
| `e5cc36550` | `[spmd_types] qwen3 sharding (#3653)` | `models/qwen3/sharding.py`; no `ezpz/qwen3/` fork exists. |
| `bfd0a998c` | `[rl] add inference_perf_hillclimb skill (#3716)` | `experiments/rl/.claude/skills/`; not used. |
| `e4035785d` | `[RL] enable GPT-OSS for titan RL loop (#3687)` | `experiments/rl/`; ezpz/rl uses Qwen3 only currently. |
| Many more | misc CI / docs / spmd_types refactors | None hit ezpz directly. |

### Verification

Static: ezpz imports cleanly post-merge (no missing symbols at merge
time; all forked files are intact). Dynamic smoke deferred -- the 6
real replays above will each be smoke-tested as they land.

**Until the replays land, agpt and moe production configs that pass
`disable_loss_parallel=...` will fail with `TypeError: unexpected
keyword argument` if torchtitan's upstream config dataclass dropped
the field**. Check before re-launching production training.

---

## 2026-06-13 — 56th sync (2 commits, `0a73d82a4..7b579adde`)

Merged clean (no conflicts) as `3935fc654`. No replays needed — both
commits are additive or scoped outside ezpz.

### Upstream commits

| Commit | Title | ezpz impact |
|---|---|---|
| `588fc12fd` | `[RL] Add deterministic loss guard for GRPO training (#3474)` | None — `experiments/rl/` only (generator/trainer/rollouter + new `loss_compare.py`). ezpz/rl has its own `train_grpo.py` / `train_sft.py`. |
| `7b579adde` | `Add MinimalAsyncEP (#3561)` | **Mostly additive** — new `MinimalAsyncEPTokenDispatcher` class in `common/token_dispatcher.py`, new `distributed/minimal_async_ep/` module, new deepseek_v3 + graph_trainer configs. Two small in-place mods to `AllToAllTokenDispatcher` (add `output_size=total` hint to `repeat_interleave` for compile-time shape). PR14's fork at `ezpz/moe/token_dispatcher.py` has the same `repeat_interleave` call but no replay needed: the `output_size` arg is only a static-shape hint for `torch.compile`, runtime behavior identical without it (and we run MoE with `--compile.no-enable`). |

### Verification

Static: ezpz imports cleanly. Dynamic smoke deferred — PR14's
`moe_10b_2b_sdpa_ep` 8N validation just landed (jobs `12468735` +
`12468736`), and this sync touches no path that wasn't covered.

---

## 2026-06-12 — 55th sync (3 commits, `96ab7487d..0a73d82a4`)

Merged clean (no conflicts) as `439ccf220`. No replays needed — all
three commits land in upstream code paths that ezpz consumes via
import-only (no forks, no overrides):

### Upstream commits

| Commit | Title | ezpz impact |
|---|---|---|
| `3b8e060853` | `Remove unused MetricsProcessor.lr_schedulers (#3644)` | None — ezpz's local `metrics.py` fork was removed earlier (PR #14 commit `8fbd896e5`); we now consume upstream `MetricsProcessor` directly. The deleted attribute was never read anywhere in ezpz. |
| `14fb67575` | `qwen3.5 tok_embeddings LocalMap region (#3648)` | None — `models/qwen3_5/sharding.py` only. ezpz doesn't use qwen3_5. |
| `0a73d82a4` | `avoid GradAccumulator init in ChunkedCELoss no_grad path (#3652)` | None — `components/loss.py` internal fix; ezpz uses the upstream `ChunkedCELoss` symbol unchanged (the ezpz `loss.py` fork was removed in PR #14 commit `8fbd896e5`). |

### Verification

Static: ezpz consumes all three upstream modules via direct import; no
override surface touched. Dynamic smoke deferred — last sync's
agpt_2b_chunkedce bitwise IDENTICAL (job `12468696`) already covered
the `ChunkedCELoss` code path this sync touches.

---

## 2026-06-12 — 54th sync (3 commits, `1c02a5cee..96ab7487d`)

Merged clean (no conflicts). No replays needed — all three commits
either don't touch surfaces ezpz overrides, or fix bugs in code paths
ezpz doesn't exercise.

### Upstream commits

| Commit | Title | ezpz impact |
|---|---|---|
| `88030eec1` | `[rl] Fix batch invariant logprob calculation by forcing vllm use trainer's function (#3629)` | None — touches `experiments/rl/actors/generator.py`. ezpz/rl has its own `train_grpo.py` / `train_sft.py` that don't import from `experiments/rl/actors/`. |
| `5ba439938` | `[Bug] Fix MoE SP token combine indices (#3604)` | **Inherited (no-op in practice)** — fixes a `B > 1` × `sp_size > 1` bug in `common/token_dispatcher.py`. ezpz/moe re-imports the dispatcher unchanged, so the fix flows automatically. Our MoE configs (`moe_2b_ep`, `moe_10b_2b_sdpa_ep`) run with `sp_size == 1`, so the buggy code path was never live for us. |
| `96ab7487d` | `chore(ci): migrate ROCm matrix from 7.1 to 7.2 (#3267)` | None — CI matrix + ROCm loss reference files only. ezpz doesn't run on ROCm or hit these CI configs. |

### Verification

Static: `import torchtitan.experiments.ezpz.{train, optimizer.containers, moe}` all succeed.

Dynamic:

1. **agpt_2b_chunkedce bitwise check** (job `12468696`, 2N,
   `bitwise_sync_check.sh` comparing `434cfe5d1` pre-merge vs
   `f8be3bcd1` post-merge with `--debug.seed=42 --debug.deterministic`)
   — **VERDICT: IDENTICAL** (loss + grad_norm match bit-for-bit
   across all 20 steps; head step 20 = pre step 20 = 10.66272 /
   18.1259).
2. **moe_10b_2b_sdpa_ep 10-step smoke** (job `12468697`, 2N) —
   **passed** (loss 12.89 → 8.87 over 10 steps; grad_norm stayed
   bounded; ~80 GiB peak). EP=2, SP=1, so the `5ba439938` SP fix
   doesn't enter our code path — this just confirms no regression
   in the EP forward/backward.

---

## 2026-06-12 — 53rd sync (2 commits, `1cc10d1ed..1c02a5cee`)

Merged clean (no conflicts). No replays needed.

### Upstream commits

| Commit | Title | ezpz impact |
|---|---|---|
| `772dd1b6c` | `[Checkpointer] Remove the dependencies on PyTorch distributed state_dict APIs (#3623)` | None — refactors `components/checkpoint.py` internals + adds `components/checkpoint_utils.py`. ezpz imports `CheckpointManager` + `ModelWrapper` by name; both still exported. Verified imports cleanly. |
| `1c02a5cee` | `Revert "Add deterministic topk for MoE routing" (#3647)` | Reverts PR #3600 from the 52nd sync. ezpz/moe re-imports `TokenChoiceTopKRouter` from `common/moe.py`, so the change flows through automatically — no ezpz-side replay needed (same as the original add). |

### Verification

Static: `import torchtitan.experiments.ezpz.{train, optimizer.containers, moe}` all succeed.
Dynamic smoke: deferred — 52nd-sync's `agpt_2b_chunkedce` bitwise
IDENTICAL already covers the checkpoint/optimizer code paths this
sync touches, and the topk revert is a no-op for ezpz.

---

## 2026-06-12 — 52nd sync (18 commits, `a97767611..1cc10d1ed`)

Merged clean (no conflicts). Two small replays required: PR #3643
(`spmd.R` → `spmd.I` for non-SP attn_x_layout) onto `agpt/sharding.py`,
and PR #3626 (rename MoE expert weight FQNs `w{1,2,3}` → `w{1,2,3}_E[F]D`)
onto `moe/state_dict_adapter.py`. Big-ticket items in the range:
PR #3619 virtual padding for SP token dispatcher (touches the same
`common/token_dispatcher.py` PR #14 already forked + extended), PR
#3600 deterministic topk for MoE routing (adds `torchtitan/ops/topk.py`),
PR #3641/3643/3468 SPMD types reorg, PR #3371 qwen3_vl → qwen3_5
rename.

### Upstream commits

| Commit | Title | ezpz impact |
|---|---|---|
| `fd712e814` | `[qwen3_5] evolve qwen3_vl to qwen3_5 (#3371)` | None — qwen3_vl was deleted/renamed; ezpz doesn't import it |
| `7f0749e64` | `Add a router for multiple generators (#3583)` | None — experiments/rl, separate from ezpz/rl |
| **`db0a72345`** | **`Using "virtual padding" to calculate number_local_tokens per SP rank, and fix combine() shape mismatch #3595 (#3619)`** | **Inherited** — touches `common/token_dispatcher.py` which moe re-imports. ezpz/moe's own dispatcher fork (post-PR #14) needs awareness; see "Inherited" below |
| `67ca69023` | `[graph_trainer] Add view replay for CPU activation offloading (#3522)` | None — graph_trainer experiment |
| `831e36e8a` | `Pass original size/stride as explicit args to ao.reload (#3598)` | None — torchao reload path, not used by ezpz |
| `19258f64a` | `[RL] Enable regional_inductor in FlexAttention (#3563)` | None — experiments/rl |
| `f59f47ec4` | `[graph_trainer] Fix GraphTrainer CI for the cu130 nightly; quarantine upstream-blocked H100 tests (#3588)` | None — CI only |
| **`2816b97c2`** | **`[MoE] fix MoE state dict convert fqn names (#3626)`** | **Replay required** — see below |
| `07828a431` | `[graph_trainer] Replace stable_topological_sort with _move_overlap_nodes (#3419)` | None — graph_trainer |
| `34c6f5b0a` | `Memory snapshot: python-only stacks for fast dumps + configurable max_entries (#3628)` | None — `tools/profiler.py`; ezpz uses default profiler config |
| `7c5ea5143` | `Make GraphTrainer overlap scheduling work with current nightlies (#3635)` | None — graph_trainer |
| `2f1ca1d6b` | `[graph_trainer] Match Eager FSDP bucket order (#3590)` | None — graph_trainer |
| `3975b85b1` | `[GraphTrainer] Fix test_deterministic: update model hash and calling update_from_config (#3605)` | None — graph_trainer test |
| `1d9af9ad0` | `Add deterministic topk for MoE routing (#3600)` | New `torchtitan/ops/topk.py`; ezpz/moe doesn't import it (uses `common/moe.py`'s router directly). No replay. |
| `a0c831c49` | `[spmd_types] spmd infra (#3641)` | None — additive infra |
| `bb453e0ca` | `[rl] continuous-batching generator + multi-turn rollouts (#3593)` | None — experiments/rl |
| **`eecdf5096`** | **`[spmd_types] decoder sharding in spmd.* (#3643)`** | **Replay required** — see below |
| `1cc10d1ed` | `[spmd_types] embedding vocab parallel (#3468)` | None — opt-in via new `models/common/embedding.py`; ezpz models don't import it yet |

### Replay: rename MoE expert weight FQNs (commit `2816b97c2`)

Upstream renamed the HF→torchtitan mapping for routed expert weights
in `deepseek_v3/state_dict_adapter.py`:

| HF | old torchtitan FQN | new torchtitan FQN |
|---|---|---|
| `model.layers.{}.mlp.experts.{}.gate_proj.weight` | `layers.{}.moe.experts.w1` | `layers.{}.moe.experts.w1_EFD` |
| `model.layers.{}.mlp.experts.{}.up_proj.weight`   | `layers.{}.moe.experts.w3` | `layers.{}.moe.experts.w3_EFD` |
| `model.layers.{}.mlp.experts.{}.down_proj.weight` | `layers.{}.moe.experts.w2` | `layers.{}.moe.experts.w2_EDF` |

The new names match the Shazeer shape-suffix style that upstream
already adopted for the actual `GroupedExperts` param names back
in the 41st sync (PR #3425). The state-dict adapter was the last
holdout still using the un-suffixed names. ezpz/moe's own
`state_dict_adapter.py` mirrored the old surface, so we replay the
exact same 3-key rename onto our adapter.

### Replay: `spmd.R` → `spmd.I` for non-SP attn_x_layout (commit `eecdf5096`)

PR #3643 reorganized decoder sharding into the `spmd.*` namespace.
The mechanical change touching llama3 is one line in
`llama3/sharding.py`: when sequence-parallelism is OFF, the attention
output layout switches from `dense_activation_placement(tp=spmd.R)`
(Replicate) to `dense_activation_placement(tp=spmd.I)` (Identity).
`R` would force a redundant all-reduce that's a no-op in the eager
case but trips up the new spmd type-checking. `I` correctly says "the
tensor is already in the right shape, no reshard needed".

ezpz/agpt mirrors the same `attn_x_layout` construction in
`agpt/sharding.py` — same one-line replay.

### Inherited: PR #3619 virtual padding for SP token dispatcher

Upstream fixed a `combine()` shape-mismatch bug in
`common/token_dispatcher.py` by switching from per-rank-actual to
virtual-padded `number_local_tokens` calculation under SP. This is
exactly the area PR #14 forked into `ezpz/moe/token_dispatcher.py`,
so the inheritance story matters:

- ezpz/moe's `LocalTokenDispatcher` still imports `common/moe.py`
  helpers, so the upstream `common/token_dispatcher.py` changes
  flow through cleanly when our fork doesn't override the relevant
  call.
- The specific helpers PR #3619 touches (`_compute_input_splits`
  and `combine`'s shape derivation under SP) live in
  `common/token_dispatcher.py`. ezpz/moe's fork redefines
  `LocalTokenDispatcher.dispatch` + `combine`, so the upstream fix
  doesn't auto-propagate. We currently don't run SP on MoE
  (SP > 1 is untested for ezpz/moe), so this is a future concern
  rather than an immediate breakage.
- **Action:** revisit if/when we enable SP for ezpz/moe — port the
  virtual-padding logic into our forked dispatcher.

### Verification

Three smoke runs total. Net result: merge-ready.

1. **`agpt_2b_chunkedce` 20-step bitwise sync** (job `12468664`, 2N,
   `--debug.deterministic`, post-merge HEAD `3738b6dfb` vs pre-merge
   `1b43fb152`) — **IDENTICAL**. Loss + grad_norm match bit-for-bit
   across all 20 steps. ~3 min wall.

2. **`moe_10b_2b_sdpa_ep` 10-step drift check** (job `12468666`, 2N,
   seed-only no determ) — **drift ≤4e-4 nats / ≤0.03 grad_norm**
   through step 10. Step 1 bit-identical (12.92541 / 2.2413 on both
   sides), subsequent drift follows the same FP-summation noise
   pattern we already characterized in the PR #14 A/B work
   (non-deterministic XCCL reduction order). Both head and pre
   completed cleanly; no infra issues.

3. **`moe_debugmodel_ep` 10-step `--debug.deterministic`** (jobs
   `12468665` + `12468667`, both attempts) — **infrastructure
   failure, unrelated to sync**. oneCCL bails with
   `comm.cpp:661 get_scaleout_device_buf: EXCEPTION: malloc
   scaleout_device_buf failed` during the first `reduce_scatter_tensor`
   on both attempts on different node sets. The determinism flag
   forces XCCL to use larger workspace pools; this path is broken
   on the current oneCCL build regardless of the sync. Filed as
   pre-existing infra, not a 52nd-sync regression.

The trainer fix (PR #3641 keyword-only `get_train_context`) was a
post-merge replay required to even boot the training entry point —
discovered by the head phase of job `12468663`'s first attempt.
Committed at `3738b6dfb`.

---

## 2026-06-10 — 51st sync (9 commits, `842d354f9..a97767611`)

Merged clean (no conflicts). One replay required: PR #3571 dropped
the `mask_type` field from `GQAttention.Config` + the tuple return
from `get_attention_config()`, and our `_ezpz_get_attention_config`
+ `_build_agpt_layers` were calling that surface.

### Upstream commits

| Commit | Title | ezpz impact |
|---|---|---|
| `e6adf26b1` | Fix experiemental CI trigger condition + RL CI editable install (#3541) | None — experimental/rl CI, not ezpz |
| `0f929e734` | `[rl] PR2/N — AlphabetSort task; remove SumDigits (#3582)` | None — experimental/rl task registry, separate from ezpz/rl |
| `873868905` | `[rl] PR 1/N - rollout logger (#3581)` | None — experimental/rl |
| **`169545712`** | **`[BE] deprecate SDPA and causal mask_type for language models (#3571)`** | **Replay required** — see below |
| `a9c5edc74` | `ci: skip fsdp_symm_mem integration test on ROCm (#3597)` | None — CI only |
| `51f107fbe` | `[spmd_types] state sharding and redistribution infra (#3587)` | None — adds new infra, doesn't change existing APIs |
| `9d3c7d205` | `[graph_trainer] hoist collective-PG reassignment into its own default pass (#3592)` | None — graph_trainer experiment |
| `a97767611` | `[graph_trainer] Fix precompile pickle failure with FlexAttention BlockMask (#3428)` | None — graph_trainer experiment |
| `98efb19e1` | `[Full DTensor] Enable full_dtensor for all MoE models (#3447)` | **No replay needed** (opt-in path) — see below |

### Replay: drop `mask_type` from GQAttention plumbing (commit `d27f9dbb9`)

Upstream removed the `mask_type` field from `GQAttention.Config` and
changed `get_attention_config()` to return just the inner-attention
config (no longer a `(config, mask_type)` tuple). It also removed the
SDPA branch from the language-model `get_attention_config()` entirely
(the text/chat dataloaders always emit per-document positions which
SDPA can't consume).

ezpz mirrored that surface in two places:

- `ezpz/agpt/__init__.py`
  - `_ezpz_get_attention_config()` now returns just the config (was a
    `(config, mask_type)` tuple). Keeps the XPU SDPA branch alive —
    upstream removed SDPA entirely from language models but XPU lacks
    a working FlexAttention backend, so we still need it.
  - `_build_agpt_layers()` drops the local `mask_type` variable and
    the `mask_type=...` kwarg passed to `make_gqa_config()`.

- `ezpz/moe/__init__.py`
  - Same `_ezpz_get_attention_config()` API change rippled through.
  - ezpz/moe's `Attention.Config` (MLA, defined in `moe/model.py`)
    is OUR own class and still has its own `mask_type` field —
    upstream's removal only affected `GQAttention.Config`. So we
    synthesize `_mask = "causal"` locally to keep moe behavior.

### Held: PR #3447 Full DTensor for MoE

Adds `--training.spmd_backend=full_dtensor` support to qwen3 /
llama4 / gpt_oss / deepseek_v3 parallelize.py via new helpers in
`distributed/full_dtensor.py` and `distributed/fsdp.py`. ezpz/moe's
parallelize.py has its own (non-full_dtensor) path that doesn't
go through the deepseek_v3 codepath touched by this PR, so no
replay needed unless/until we want to opt ezpz/moe into the
full_dtensor path.

### Verification

Bitwise sync check `12468399` (`agpt_2b_chunkedce`, 2N, 20 steps,
HEAD vs pre-merge `a09216324`) — **IDENTICAL**. Loss + grad_norm
match bit-for-bit across all 20 steps. 5:45 wall.

---

## 2026-06-09 — 50th sync (2 commits, `465cd676e..842d354f9`)

Two-commit follow-up to the 49th sync, both small. Merged clean (no
conflicts).

### Upstream commits

| Commit | Title | ezpz impact |
|---|---|---|
| `7f5d11932` | `[graph_trainer] memory_policy: always save sym_size/shape ops (#3546)` | None — graph_trainer is a separate experiment we don't depend on. |
| `842d354f9` | `[spmd_types] global_valid_tokens: float | None (#3586)` | **Replay required** — see below. |

### Replay: `global_valid_tokens` retype (commit `bb31cab9e`)

Upstream retyped `global_valid_tokens` from `torch.Tensor` to
`float | None` in `Trainer.forward_backward_step` and
`BaseLoss.__call__`/`ChunkedCELoss.__call__`. In the no-DP branch of
`Trainer._inner_training_loop`, it also switched
`local_valid_tokens.float()` (Tensor) →
`float(local_valid_tokens.item())` (Python float) so the runtime value
matches the annotation.

Mirrored in `ezpz/trainer.py:541` and `ezpz/validator.py:137`. The DP
branch is left as-is (still returns a Tensor from `dist_sum`) —
upstream itself does the same; the annotation is loose and consumers
accept either at runtime.

Functional impact for current production / smoke runs: **none**. All
live runs go through the DP branch.

### Verification

Bitwise sync check `12468341` (`agpt_2b_chunkedce`, 2N, 20 steps,
HEAD vs pre-merge `453a386e8`) — **IDENTICAL**. Loss + grad_norm
match bit-for-bit across all 20 steps. 5:40 wall.

---

## 2026-06-09 — 49th sync (15 commits, `21c77d165..465cd676e`)

Merged with **one conflict** in `torchtitan/experiments/__init__.py`
(`ft.llama3` → `torchft.llama3` rename vs our list of ezpz experiments
in the same `_supported_experiments` frozenset). Resolved by taking the
upstream rename and keeping our ezpz entries.

### Notable upstream commits

| Commit | Title | ezpz impact |
|---|---|---|
| `465cd676e` | `[spmd_types] SpmdLayout for NamedPlacement (#3501)` | None for now (touches `models/{llama3,qwen3,deepseek_v3}/sharding.py`; ezpz/agpt + ezpz/moe have their own sharding files that don't mirror this layout). **TODO**: revisit if we hit a SpmdLayout-related error |
| `d92336fee` | `[BE] deprecate llama4, move apply_fsdp to common file (#3573)` | **Replay candidate**: ezpz/agpt + ezpz/moe each have a local `apply_fsdp` that's a subset of the new `apply_fsdp_to_decoder` in `distributed/fsdp.py`. Holding for a focused refactor session rather than replaying mid-sync |
| `43e52fd01` | `Refactor gypo.py into trainer.py and train.py (#3451)` | None (RL experiment — we have our own `experiments/ezpz/rl/train_grpo.py` + `train_sft.py`) |
| `da230bbb2` | `[rl][bug] Fix gradient accumulation zero grad (#3575)` | None (same — separate RL trainer; our train_grpo doesn't override the zero_grad path) |
| `2b22fc197` | `[BE] rename experiments/ft to experiments/torchft (#3574)` | **Replay required** (see below) |
| `577533a44` | `[fix]Allow initial_load_in_hf without initial_load_path (#3578)` | None (checkpoint.py change; we don't override that path) |
| `bc878a7c6` | `Llama3 stat dict adapter infers the head size...` | None (we don't use the Llama3 state-dict adapter directly; ezpz/agpt has its own checkpoint conversion under `eval/convert_to_hf.py`) |

### Replay: `experiments.ft` → `experiments.torchft` (commit `a96cffec7`)

Upstream renamed `torchtitan.experiments.ft` to `torchtitan.experiments.torchft`
and prefixed exported classes with `TorchFT`. Five ezpz files imported from
the old paths and needed updating:

- `agpt/__init__.py` — `FaultTolerantModelSpec` import + `fragment_llm` import
- `agpt/config_registry.py` — `FaultTolerance` import
- `moe/config_registry.py` — `FaultTolerance` import
- `trainer.py` — `FaultTolerance`, `FTManager` → `TorchFTManager`,
  `FTOptimizersContainer` → `TorchFTOptimizersContainer`,
  `FTCheckpointManager` → `TorchFTCheckpointManager`. Aliased on import
  so the rest of trainer.py is unchanged.

Verified `import torchtitan.experiments.ezpz.{agpt, moe, train, trainer,
rl.train_grpo, rl.train_sft}` all succeed post-rename.

### Held: `apply_fsdp` consolidation

Upstream PR #3573 consolidated llama3's + llama4's `apply_fsdp` functions
into a single `apply_fsdp_to_decoder` in `distributed/fsdp.py`. Per the
replay protocol we *should* also consolidate ezpz/agpt's local `apply_fsdp`
to call the new helper.

Holding for a focused session because:

1. ezpz/agpt's local `apply_fsdp` is a **subset** of `apply_fsdp_to_decoder`
   — missing `weight_tying`, `dp_mesh_dims`, `enable_symm_mem`,
   `ep_degree`, `edp_mesh` parameters. None of those are required for
   our current configs, but they're not no-ops either.
2. The consolidation is a real refactor touching ezpz/agpt/parallelize.py
   and ezpz/moe/parallelize.py — not a 1-line sed.
3. The current ezpz `apply_fsdp` is battle-tested across all v2 production
   runs (2B/20B/80B). Worth being careful before swapping it.

**TODO** track: replay PR #3573 onto ezpz/agpt + ezpz/moe in a separate
commit when there's bandwidth.

---

## 2026-06-06 (48th sync — 1-commit follow-up to spmd_types/AC story; no ezpz replay)

Pulled 1 commit (`641b5f6b8..21c77d165`) immediately after the 47th sync:

- **[`21c77d165` — disable autograd multithreading
  (#3565)](https://github.com/pytorch/torchtitan/pull/3565).** 5-line
  add to `torchtitan/distributed/utils.py:init_distributed`:
  ``torch.autograd.set_multithreading_enabled(False)``. Needed for
  AC functionality with the new spmd_types backend: multi-threaded
  autograd means BWD recompute threads can't access PGs (e.g.
  ``current_mesh().get_group("tp")``) for collectives. Touches only
  core `distributed/utils.py`. ezpz inherits the new behavior via
  `dist_utils.init_distributed(config.comm, ...)` in
  `FaultTolerantTrainer.init_distributed`. No replay needed; no
  conflicts.

---

## 2026-06-06 (47th sync — RoPE + optimizer refactors replayed; SMOKES PASSED, READY TO MERGE)

**Status: READY TO MERGE — worktree `ezpz-46th-47th-sync` validated
end-to-end.** Both structural refactors replayed; 7 of 9 configs
smoked clean (the 2 skips are pre-existing constraints, not
regressions). 4 post-smoke bugs fixed in-place. See journal for the
full smoke matrix.

Pulled 34 commits (`27aa49077..641b5f6b8`) from `upstream/main`.

### Replayed (RoPE refactor, PR #3458)

[`02d24f017` — Move centralized freqs_cis to each transform
layer](https://github.com/pytorch/torchtitan/pull/3458). Three
structural changes:

1. `RoPE.Config` split into `ComplexRoPE.Config` and `CosSinRoPE.Config`
   — `backend="complex"|"cos_sin"` field gone, backend encoded in
   type.
2. Top-level `Model.Config.rope` removed; each layer's
   `Attention.Config` owns a `rope: RoPE.Config`. `decoder.forward`
   no longer threads `freqs_cis`.
3. `apply_rotary_emb_{complex,cos_sin,single_complex}` removed.
   Caller pattern: `self.rope = config.rope.build()` then
   `q, k = self.rope(q, k, positions)`.

Replayed across 4 ezpz files in commit `02dd1e7fe`:

* `agpt/__init__.py`: `_build_agpt_layers(rope=RoPE.Config)` plumbing;
  `rope_backend` kept as a back-compat keyword that selects the
  subclass internally; dropped top-level `AgptModel.Config(rope=...)`.
* `agpt/config_registry.py`: `_set_rope_backend` rewritten to swap
  each per-layer `attention.rope` to a fresh `ComplexRoPE.Config` /
  `CosSinRoPE.Config`, carrying forward all other fields via
  `dataclasses.fields()`.
* `moe/__init__.py`: `_make_moe_attn_config(rope=...)` +
  `_build_moe_layers(rope=...)` plumbing; all 11 config functions
  push their `ComplexRoPE.Config` from `moeModel.Config(...)` into
  `_build_moe_layers(rope=...)`. `_small` keeps its outlier
  `theta=50000` / `max_seq_len=256128`.
* `moe/model.py`: `Attention.Config` drops the legacy
  `rope_{factor,max_seq_len,original_seq_len}` triple, gains
  `rope: RoPE.Config`. `Attention.__init__` reads
  `config.rope.{max_seq_len, original_seq_len, rope_factor}`, builds
  `self.rope`. `Attention.forward` drops `freqs_cis` and replaces
  two `apply_rotary_emb_single_complex` calls with one
  `self.rope(q_pe, k_pe.unsqueeze(2), positions)` returning both
  rotated tensors. `moeTransformerBlock.forward` drops `freqs_cis`.
  `moeModel.Config.update_from_config` drops the now-redundant
  rope-sync block.

**Verified end-to-end** under torch 2.13 venv (login node import test):
agpt configs `debugmodel` / `2B` / `80B` and all 10 moe flavors
`debugmodel` / `500M` / `2B` / `small` / `4B` / `7B` / `16B` / `236B`
/ `10B_2B` / `10B_2B_sdpa` all build cleanly. Numerics-equivalence
smoke against pre-merge baselines NOT yet run.

### Replayed (mixed-optimizer refactor, PR #3269)

[`632f67f12` — [optimizer] support mixed
optimizers](https://github.com/pytorch/torchtitan/pull/3269) replaces
the flat `OptimizersContainer.Config(lr=8e-4)` shape with a per-group
shape:

```python
OptimizersContainer.Config(
    param_groups=[ParamGroupConfig(pattern=r".*", optimizer_name="AdamW",
                                   optimizer_kwargs={"lr": 8e-4})],
    implementation="fused",
)
```

`OptimizersContainer.__init__` walks model params first-match-wins,
batches by `optimizer_name`, and instantiates one optimizer per
`(model_part, optimizer_name)` pair. Subclasses register additional
optimizer types via `_resolve_optimizer_cls(name)`. The flat `lr` /
`beta1` / `beta2` / `eps` / `weight_decay` / `name` fields are gone
from `OptimizersContainer.Config`; users supply them through
`ParamGroupConfig.optimizer_kwargs`.

Replayed across 6 ezpz surfaces in commit `bac0a3473`:

* `optimizer/containers.py`: each custom container subclass is now
  thin — just overrides `_resolve_optimizer_cls(name)` to register
  its optimizer class name. Dropped the flat-field Config classes
  and `_build_optimizer_kwargs` helpers. Added matching
  `default_<name>(lr=..., **kwargs)` factories
  (`default_muon`, `default_sophiag`, `default_mano`,
  `default_spam`, `default_adopt`, `default_muon_clip`,
  `default_schedule_free`, `default_torch_muon`) mirroring
  upstream's `default_adamw`. Kept the runtime extras
  (`SophiaG.update_hessian`, `ScheduleFree.train_mode/eval_mode`,
  `_CompositeOptimizer`, `register_muonclip_qk_pairs`).
  `TorchMuonOptimizersContainer` keeps its bespoke `__init__`
  (shape-based split doesn't fit pattern grouping) but reads its
  kwargs from `param_groups[0].optimizer_kwargs`.
* `optimizer/__init__.py`: re-export 8 new `default_<name>`
  factories + the previously-missing `ScheduleFree` / `TorchMuon`
  container symbols.
* `agpt/config_registry.py`, `moe/config_registry.py`: swap
  baseline `OptimizersContainer.Config(lr=8e-4)` to
  `default_adamw(lr=8e-4)`.
* `competition/configs.py`: 28 `<Custom>OptimizersContainer.Config(lr=X)`
  callsites swapped to `default_<name>(lr=X)`. 19 post-construction
  `cfg.optimizer.lr = X` mutations rewritten to
  `cfg.optimizer.param_groups[0].optimizer_kwargs["lr"] = X`.
* `train.py`: rewrote the `_build_optimizer_config` swap helper
  (`--optimizer <name> --optimizer.<key>=<val>`). The old version
  walked `dataclasses.fields(OptimizersContainer.Config)` for
  `lr`/`beta1`/etc. — those fields no longer exist. New version
  is a thin dispatch through `_OPTIMIZER_FACTORIES[name](**overrides)`
  with a `_coerce_override` helper for bool/int/float/str string
  coercion. Dropped the now-unused `dataclasses` import.

**Verified end-to-end** under torch 2.13 venv (login node import test):

- agpt configs (debugmodel, 2B, 20B, 80B) — all build, all
  show `param_groups=1 first_pg=AdamW/0.0008`.
- moe registry (10 flavors) + baseline configs
  (`moe_2b`, `moe_2b_ep`, `moe_debugmodel`) — all build.
- All **48 of 48** competition configs build cleanly (after fixing
  19 `cfg.optimizer.lr = X` mutations).
- All 9 `--optimizer <name>` swap paths exercised through
  `_build_optimizer_config` with single + multi-kwarg overrides.

`lr_finder.py:205` only reads container class names for output
directory layout — no code change needed.

Numerics-equivalence smoke against pre-merge baselines NOT yet run.

### Other commits in this sync (no ezpz replay required)

- `641b5f6b8` / `fec0c175d` / `c0428bb18` — spmd_types backend
  config + manual loss parallel CE. Adds a runtime dep on
  `spmd_types==0.2.1` (already in upstream `requirements.txt`).
  Installed into `.venv` via `uv pip install`.
- `06d4a35e2` — Qwen3 30B-A3B config. ezpz doesn't have a qwen3
  folder; n/a.
- 10 graph_trainer-only commits; n/a.
- 4 RL commits — possibly need attention if ezpz/rl/ shares the
  same surface; not yet audited.
- 6 CI / ROCm / Monarch / checkout infra commits; n/a.

### Post-replay smokes + fixes

Numerics smoke against pre-merge baselines:

- **moe_2b_ep 2N** (job 12468156): 10 steps clean, loss 12.95 → 7.92,
  memory bit-identical to 2026-06-02 baseline.
- **agpt_80b TP=2 4N** (job 12468157): 20 steps clean, all losses
  within ±0.08 nat of 2026-06-02 baseline, MFU + memory + grad-norm
  shape match.

Wider coverage smoke surfaced 3 real bugs in the initial replay, all
now fixed in the worktree:

| Commit | Fix |
|---|---|
| `6871e736b` | `optimizer/containers.py` Config-dispatch bug. Each custom container subclass now carries its own empty `Config(OptimizersContainer.Config): pass` so `build()` instantiates the subclass instead of the base. Otherwise `NotImplementedError: Optimizer Muon not added` at trainer init. |
| `455013ed5` | 5 missed `cfg.optimizer.lr = X` mutations in `moe/config_registry.py` (moe_16b, moe_671b, moe_10b_2b, moe_10b_2b_sdpa, smoke_moe_500m_50steps). Same fix pattern as the 19 in competition/configs.py. |
| `975a5bcd1` | `moe_10b_2b_sdpa{,_ep}` defaults changed to `(LBS=1, AC="selective")`. Prior `(LBS=2, AC="none")` OOMs at first forward on 2N. AC="full" hits `CheckpointError: Recomputed values have different metadata` from MoE router non-determinism under recompute. AC="selective" excludes the router from the save list → router never gets recomputed → shapes stay stable. |

One pre-existing bug found in passing (not a replay regression but
fixed while we were here):

| Commit | Fix |
|---|---|
| `8746dfe2c` | `datasets.py` `_make_text_processor` returned a local closure that couldn't be pickled by PyTorch's `forkserver` DataLoader workers. Every HF-dataset run with `--dataloader.num-workers >= 1` crashed at first batch with `PicklingError`. Fix: module-level helper + `functools.partial`. |

Final smoke matrix:

| Config | Status | Notes |
|---|---|---|
| moe_2b_ep 2N | ✅ baseline match | |
| agpt_80b TP=2 4N | ✅ baseline match | |
| agpt_2b | ✅ 10 steps, 12.95 → 7.63 | |
| agpt_2b_real | ✅ 10 steps, 12.99 → 8.50 | validates CosSinRoPE swap |
| agpt_20b | ✅ 10 steps, 12.90 → 10.39 | noisy (no warmup) but trains |
| moe_2b (LBS=2) | ✅ 10 steps, 12.94 → 8.52 | LBS=16 OOMs (pre-existing) |
| moe_10b_2b_sdpa_ep (LBS=1 + AC=selective, new defaults) | ✅ 10 steps, 12.96 → 9.44, 80% mem | |
| speedrun_2b_muon (LBS=1) | ✅ 10 steps, 12.93 → 9.32 | validates Muon dispatch |
| speedrun_2b_sophiag (LBS=1) | ✅ step 1 reached training | validates SophiaG dispatch |
| moe_10b_2b | ⏭ skipped | block_causal mask + HF-dataset mismatch (pre-existing) |
| moe_10b_2b_sdpa{,_ep} @ AC=full | ⏭ known broken | MoE router non-determinism under recompute (long-standing) |

### Action items (post-merge)

- File pytorch/pytorch issue for MoE + AC=full `CheckpointError`
  (long-standing; PR #3146/#3450 fixed the forward path only).
- Audit the 4 RL commits in this sync — `experiments/ezpz/rl/` may
  need attention if they touch shared surfaces.

Merge commit: `fb1c5a319` (worktree).
Replay commits: `02dd1e7fe` (RoPE), `bac0a3473` (mixed-optimizer).
Post-smoke fixes: `6871e736b`, `455013ed5`, `975a5bcd1`, `8746dfe2c`.

---

## 2026-06-02 (46th sync — graph_trainer-only deltas, no ezpz replay)

Pulled 2 commits (`04a309858..27aa49077`) from `upstream/main`. Both
land entirely inside `torchtitan/experiments/graph_trainer/` —
unrelated to `experiments/ezpz/`. No replay needed. No conflicts.
Tree clean.

- **[`27aa49077` — [graph_trainer] Add full recompute memory policy
  (#3429)](https://github.com/pytorch/torchtitan/pull/3429).** Adds
  a `FULL_RECOMPUTE` memory policy alongside the existing SAC
  policies; +358 / -15 across `configs.py`, `memory_policy.py`,
  `selective_activation_remat.py`, plus new `tests/test_passes.py`
  and `tests/test_sac_peak_memory.py`. Touches nothing outside
  `experiments/graph_trainer/`.
- **[`051562e31` — [graph_trainer] Re-enable DSv3 eager bitwise
  deterministic tests (#3482)](https://github.com/pytorch/torchtitan/pull/3482).**
  Single-file test-toggle change in
  `experiments/graph_trainer/tests/test_bitwise_deterministic.py`
  (+10 / -10).

Merge commit: `45a2b2568`. No ezpz code or doc changes triggered.

### Action items

(none from this sync.)

After replaying, verify convergence didn't break by running both smoke
tests and checking against the saved baselines — see
[`baselines/README.md`](baselines/README.md).

---

## 2026-06-02 (45th sync — PR #3450 closes our PR #3436 thread)

Upstream merged in 7 commits (`b72d98648..04a309858`).

- **[`4c6f72d21` — \[MoE\] Using routing map instead of histc to count number_tokens_per_expert (#3450)](https://github.com/pytorch/torchtitan/pull/3450).**
  **This is the upstream fix that supersedes our PR #3436.** Replaces
  both `torch.histc` callsites (`models/common/moe.py:262` and
  `models/common/token_dispatcher.py:92`) with a boolean
  `routing_map_BLE` built via `scatter_` in the router, then
  `routing_map.sum(...)` to derive `num_tokens_per_expert_E`. The
  `routing_map` is computed once and reused across both router and
  dispatcher. Same direction we recommended (`scatter_add_` over
  `bincount`); the upstream approach goes one step further by
  computing the map once and threading it through. Also lands a new
  `torchtitan.ops.scatter_add.deterministic_scatter_add` op used in
  the `combine()` path.

  Router signature changed: `forward()` now returns a 4-tuple
  `(topk_scores_BLK, topk_expert_ids_BLK, routing_map_BLE,
  num_tokens_per_expert_E)` instead of 3.

  `dispatch()` now takes a new `num_local_tokens_per_expert_E`
  positional arg.

  ezpz doesn't override these methods directly and doesn't unpack the
  router output (the parent class does), so the API change inherits
  cleanly. ezpz `set_moe_sharding_config` calls upstream's helper,
  so the new `num_local_tokens_per_expert_E` sharding declaration is
  also picked up automatically. **No ezpz replay needed.**

  Closes our PR #3436 thread:
  - https://github.com/pytorch/torchtitan/pull/3146 (incomplete fix
    that landed only the `aten.topk.default` save)
  - https://github.com/pytorch/torchtitan/pull/3436 (our targeted
    histc → bincount swap, kept open in case #3450 stalled)
  - https://github.com/pytorch/torchtitan/pull/3450 (this — the
    clean fix done right)

- **[`0acaaa1d7` — Enables HybridEP torch.compile support and adds eager integration tests (#3360)](https://github.com/pytorch/torchtitan/pull/3360).**
  HybridEP-specific (`distributed/deepep/hybridep.py`) +
  `deepseek_v3` config. XPU doesn't have DeepEP/HybridEP, so this
  doesn't reach ezpz. No replay.

- **[`3c4d47024` — \[Quantization\] Consolidate MXFP8 Linear and GroupedExperts converters (#3473)](https://github.com/pytorch/torchtitan/pull/3473).**
  MXFP8 quantization consolidation in `components/quantization/`.
  ezpz doesn't use MXFP8. No replay.

- **[`faa96e7a6` — \[graph_trainer\] Fix DTensor shadow-node embedding OOB; re-enable 14 disabled tests (#3480)](https://github.com/pytorch/torchtitan/pull/3480).**
  `graph_trainer` only. No-op for ezpz.

- **[`8e64b62e0` — \[graph_trainer\] Skip tests broken by upstream PyTorch nightly regressions (#3432)](https://github.com/pytorch/torchtitan/pull/3432).**
  `graph_trainer` test skips. No-op for ezpz.

- **[`b598831dd` — \[CI\] Migrate H100 8-GPU integration test to OSDC (ARC) (#3464)](https://github.com/pytorch/torchtitan/pull/3464).**
  CI plumbing only. No-op.

- **[`04a309858` — \[CI\] Migrate GraphTrainer & RL H100 integration tests to OSDC (ARC) (#3479)](https://github.com/pytorch/torchtitan/pull/3479).**
  CI plumbing only. No-op.

### Smoke status

**Imports + syntax check green** post-merge. **Live 2N smoke
deferred**: the `.venv` symlink target
(`/opt/aurora/26.26.0/spack/.../python-3.12.12-5zo3wzv/bin/python3`)
isn't present on the Sunspot compute nodes I tried (allocs 12467805 +
12467806). `ezpz_setup .venv` activates the env but the `ezpz` shim
script's shebang still points at the broken target and falls back to
`/usr/bin/python3`. Venv needs rebuilding before the per-the-journal
`GroupedExperts`-touching smoke can run.

### Smoke status

**Imports + syntax check green** post-merge. **Live 2N smoke
deferred to Aurora** — the project `.venv/bin/python` symlinks at
`/opt/aurora/26.26.0/spack/.../python-3.12.12-5zo3wzv/bin/python3`
which is an Aurora-only Spack build (Sunspot's nearest equivalent is
`python-3.12.12-nvje3vk` — different hash). The
`GroupedExperts`-touching smoke per the journal lesson should run
on Aurora the next time a production alloc is available.

### Action items

- Close PR #3436 with a comment pointing at PR #3450 as the
  upstream-canonical fix that landed.
- Smoke `moe_2b_ep` per the journal's "always smoke after
  `GroupedExperts` touches" lesson, once the venv invocation is
  sorted.
- Rebuild + re-yeet the `.venv` on a Sunspot compute node, then
  smoke `moe_2b_ep` to validate the PR #3450 routing_map flow per
  the journal lesson.

### Discovered during smoke attempt (2026-06-02)

`moe_2b_ep` failed at `ParallelDims.build_mesh` with
`RuntimeError: No backend for the parent process group or its backend
does not support splitting`. Root cause: `ProcessGroupXCCL` never
overrides `Backend::supportsSplitting()`. Workaround installed under
[`experiments/ezpz/xccl_split_group_workaround.py`](../xccl_split_group_workaround.py)
(monkey-patches `DeviceMesh._init_one_process_group` to fall back to
`new_group` when the accelerator backend's `supports_splitting` is
False). Full diagnosis in
[`docs/upstream-issues/xccl_split_group_unsupported.md`](upstream-issues/xccl_split_group_unsupported.md).

---

## 2026-06-01 (44th sync — CLAUDE.md perf-iters guidance)

Upstream merged in 1 commit (`b72d98648`).

- **[`b72d98648` — Have agents run perf runs with at least 10 iterations (#3403)](https://github.com/pytorch/torchtitan/pull/3403).**
  4-line addition to project-root `.claude/CLAUDE.md`: a new
  "Performance Testing" subsection instructing AI agents to use
  ≥10 training steps (`--training.steps 10`) on perf comparisons,
  so startup + warmup don't dominate. No code touched. No ezpz
  replay needed. (Our own stats A/B work for the PR #3436 reply
  already used 50-step runs; well within the guidance.)

---

## 2026-05-31 (43rd sync — interleaved multi-source dataloader)

Upstream merged in 1 commit (`221041490`).

- **[`221041490` — \[data\] Add weighted interleaved multi-source dataloader (#3063)](https://github.com/pytorch/torchtitan/pull/3063).**
  Pure additions: new `InterleavedHuggingFaceTextDataLoader`,
  `InterleavedChatDataLoader`, `HFDataSource`/`ChatDataSource`
  config wrappers, an `InterleavedDataset` weighted sampler under
  `torchtitan/hf_datasets/interleaved.py`, plus tests. Existing
  `HuggingFaceTextDataLoader` + `DATASETS` (both imported by
  `experiments/ezpz/datasets.py` and
  `experiments/ezpz/blendcorpus/blendcorpus_builder.py`) are
  unchanged. **No ezpz replay needed.** Imports smoke-tested green.

Side note: `git log HEAD..upstream/main` showed 8 unmerged commits,
but `git cherry -v` flagged 7 as patch-equivalent duplicates from
the 41st/42nd-sync bookkeeping (same situation we documented in
the 42nd-sync entry). `git merge` correctly applied only #3063 as a
file-changing commit and recorded the rest as a marker.

---

## 2026-05-29 (42nd sync — RoPE refactor + 5 smaller commits)

Upstream merged in 6 commits (`28483d0eb..065c2625d`).

- **[`28483d0eb` — RoPE refactor: Using model's max_sequence_length as the upper bound of Training.sequence_length (#3395)](https://github.com/pytorch/torchtitan/pull/3395).**
  Significant restructuring of model `update_from_config`:
  - `trainer_config` parameter renamed to `config`.
  - `seq_len > rope.max_seq_len` is now a hard `ValueError` (was a warning).
  - TP `n_heads`/`n_kv_heads` validation, MoE `deepep`/`hybridep`
    EP=1 guard, MoE `moe_force_load_balance` debug flag, and the
    `rope.max_seq_len` sync all moved from per-model overrides into
    `Decoder.Config.update_from_config`.
  - Added async out-of-bounds `_maybe_check_max_pos` inside
    `apply_rotary_emb_{complex,single_complex,cos_sin}`.
  - **Required ezpz replays**:
    - `experiments/ezpz/agpt/model.py` ([b52e64841](https://github.com/saforem2/torchtitan/commit/b52e64841)):
      rename `trainer_config`→`config`.
    - `experiments/ezpz/moe/model.py` ([b52e64841](https://github.com/saforem2/torchtitan/commit/b52e64841)):
      rename `trainer_config`→`config`, delegate base validation to
      `Decoder.Config.update_from_config`, drop the now-redundant
      rope/MoE/TP checks (kept the per-layer attention rope-field
      sync, the for_loop XPU fallback, the CP+MoE attention check,
      and `set_moe_sharding_config`). Dropped unused imports
      (`dataclasses`, `DeepEPTokenDispatcher`, `HybridEPTokenDispatcher`).
    - `experiments/ezpz/trainer.py` ([04199e522](https://github.com/saforem2/torchtitan/commit/04199e522)):
      same `trainer_config`→`config` rename at the
      `model_config.update_from_config(...)` callsite. Missed in the
      initial b52e64841 pass; caught by the 2N smoke.

### Smoke results (2026-05-29, alloc 12467655)

- `agpt_2b`: 50 steps clean in 154 s, peak 24.34 GiB (38.04%),
  ~5,850 TPS, 21.95% MFU. Matches the 2026-05-27 baseline.
- `moe_2b_ep` at LBS=2: 50 steps clean in 323 s, peak 26.99 GiB
  (42.18%), 3,135 TPS, 9.09% MFU, loss step 50 = 6.13. Matches the
  2026-05-27 baseline. Also surfaced a separate pre-existing miss
  from the 41st sync — see commit [88dbd916e](https://github.com/saforem2/torchtitan/commit/88dbd916e).

### Bonus catch-up from 41st sync

The 41st sync ([#3425](https://github.com/pytorch/torchtitan/pull/3425)
MoE shape-suffix rename) renamed `GroupedExperts` parameters from
`w1`/`w2`/`w3` to `w1_EFD`/`w2_EDF`/`w3_EFD`. We didn't smoke
`moe_2b_ep` after that sync landed, so three ezpz-side references
to the old names slipped through:

- `experiments/ezpz/moe/sharding.py`: `_GROUPED_EXPERTS_PARAM_LAYOUT` keys.
- `experiments/ezpz/moe/__init__.py`: `_depth_experts_init` keys.
- `experiments/ezpz/moe/experts.py`: `self.w[123]` reads in `_experts_forward`.

All three caught at trainer init / first forward by today's smoke
and fixed in [88dbd916e](https://github.com/saforem2/torchtitan/commit/88dbd916e).
Lesson: always smoke `moe_2b_ep` after a sync that touches
`GroupedExperts`.

- **[`92abc88e7` — Fix model test failure in #3395: rope refactor (#3448)](https://github.com/pytorch/torchtitan/pull/3448).**
  Three-file fix-forward for #3395: `common/decoder.py` TP validation
  now handles attention configs without an `n_kv_heads` field;
  `common/rope.py` accepts a pre-broadcast 4D RoPE cache; `qwen3_vl`
  signature tweak. Pure consumption — no ezpz replay.

- **[`065c2625d` — \[be\] remove redundant torch.use_deterministic_algorithms call (#3452)](https://github.com/pytorch/torchtitan/pull/3452).**
  One-line drop of a duplicate
  `torch.use_deterministic_algorithms(True)` call in
  `distributed/utils.py:set_determinism`. No ezpz impact.

- **[`3079bc32b` — \[Module\]\[Qwen3-VL\] Module-build conversion for vision encoder (#3446)](https://github.com/pytorch/torchtitan/pull/3446).**
  qwen3_vl only. No-op for ezpz.

- **[`9c94bda0f` — \[Module\]\[Flux\] Convert Flux modules to fully config-based (#3445)](https://github.com/pytorch/torchtitan/pull/3445).**
  flux only. No-op for ezpz.

- **[`c04f92fdd` — \[rl\] Add Batcher in RL Loop (#3347)](https://github.com/pytorch/torchtitan/pull/3347).**
  RL experiment only. No-op for ezpz.

### Side note on the merge bookkeeping

`git log HEAD..upstream/main` initially showed 7 unmerged commits,
but `git cherry -v HEAD upstream/main` flagged the 41st-sync MoE [8/n]
commit (`200100e7d`) as already present in our branch under a
different SHA (`56dc8e1d4`) — it had been re-applied via a previous
merge with different metadata. `git merge` correctly skipped it.

---

## 2026-05-28 (41st sync — MoE [8/n] shape-suffix rename)

Upstream merged in 1 commit (`200100e7d`).

- **[`200100e7d` — \[MoE\]\[8/n\] Use shape suffix for MoE (#3425)](https://github.com/pytorch/torchtitan/pull/3425).**
  Pure rename refactor applying the [Shazeer shape-suffix style](https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd)
  across MoE tensors (e.g. `x → x_BLD`, `routed_output → routed_output_RD`,
  `selected_experts_indices → topk_expert_ids_TK`). Touches
  `common/moe.py`, `token_dispatcher.py`, `deepseek_v3/`, `llama4/`,
  `qwen3/`, `gpt_oss/`, the optimizer, and the EP unit tests.
  Loss-comparator verified `--assert-equal` on llama4_debugmodel
  (TP=2 EP=2). Several method signatures change (e.g.
  `dispatch(x, top_scores, selected_experts_indices)` →
  `dispatch(x_TD, topk_scores_TK, topk_expert_ids_TK)`), but those
  are internal to upstream and not called from `experiments/ezpz/`.
  **No ezpz replay needed.** Imports smoke-tested green
  (`python3 -c "import torchtitan.experiments.ezpz.{agpt,moe.model}"`).

Upstream merged in 7 commits (`19c567f76..af33f7638`).

**Upstream commits (7):**

- **[`af33f7638` — \[Module\] Replace from_nn_module with native Module subclasses (#3398)](https://github.com/pytorch/torchtitan/pull/3398).**
  Deleted `models/common/{embedding,linear,rmsnorm}.py`; consolidated
  into a single `models/common/nn_modules.py` with native Module
  subclasses (`class Linear(nn.Linear, Module)`, etc.). The `Config`
  dataclass + constructor API is unchanged. **Required ezpz replay**
  ([`b052f29e4`](https://github.com/saforem2/torchtitan/commit/b052f29e4)):
  fix 3 import paths in `ezpz/agpt/__init__.py` + `ezpz/moe/model.py`.

- **[`7074b056a` — \[MoE\]\[SAC\] Use deterministic ops in MoE routing (#3146)](https://github.com/pytorch/torchtitan/pull/3146).**
  Commit message advertises `histc → bincount` swap in
  `TokenChoiceTopKRouter`/`TokenReorderer` plus `aten.topk.default`
  on the SAC save list — but **the merged diff only contains the
  save-list change**. Smoke-tested 2026-05-27
  ([report](experiments/moe/sunspot/20260527-smoke-n2-40th-sync.md)):
  `--debug.deterministic` on MoE+XPU **still fails** with the same
  `_histc_xpu` error from 2026-05-21. The `histc` call at
  `torchtitan/models/common/moe.py:262` is untouched. Verified via
  GitHub API that PR #3146's only file change is
  `activation_checkpoint.py`. Action item: file upstream issue.

- **[`7fcd9beac` — \[MoE\]\[7/n\] Keep 3D tensors through MoE, flatten inside GroupedExperts (#3423)](https://github.com/pytorch/torchtitan/pull/3423).**
  Continues the MoE refactor from PR #3386/#3389. Touches
  `models/common/moe.py`, `moe_sharding.py`, `gpt_oss/moe.py` only —
  not `deepseek_v3/model.py`. ezpz's `moe/model.py` doesn't expose
  the 2D-flatten seam directly (the MoE wrapper does it internally),
  so no replay needed. Inherits transitively.

- **[`58b034444` — Add FSDP symmetric memory configuration and related tests (#3105)](https://github.com/pytorch/torchtitan/pull/3105).**
  New `parallelism.enable_fsdp_symm_mem` config flag, plumbed
  through each model's `parallelize.py` as a new kwarg to
  `apply_fsdp`. ezpz has its own local `apply_fsdp` in
  `agpt/parallelize.py` and `moe/parallelize.py`, so the kwarg
  doesn't reach our path automatically. **Not blocking** — symm_mem
  is an optimization, and XPU's CCL likely doesn't support it
  anyway. Skip the replay; if we want it later, we wire the kwarg.

- **[`f39e12458` — Re-enable FlexAttention bitwise tests (#3331)](https://github.com/pytorch/torchtitan/pull/3331).**
  graph_trainer tests only. No-op for ezpz.

- **[`c59f8c9b2` — \[graph_trainer\] Use separate EP process groups for overlap (#3369)](https://github.com/pytorch/torchtitan/pull/3369).**
  graph_trainer only. No-op for ezpz.

- **[`78b08dd62` — \[graph_trainer\] Add DeepSeek V3 16B SDPA config (#3361)](https://github.com/pytorch/torchtitan/pull/3361).**
  graph_trainer registry entry. No-op for ezpz.

### Replays in ezpz

[**`b052f29e4` — fix(ezpz): replay PR #3398 — import Linear/RMSNorm from nn_modules**](https://github.com/saforem2/torchtitan/commit/b052f29e4).
Pure import-path swap; class API is unchanged.

### Smoke results (2026-05-27)

Job 12467455 on 2N Sunspot. Report:
[`docs/experiments/moe/sunspot/20260527-smoke-n2-40th-sync.md`](experiments/moe/sunspot/20260527-smoke-n2-40th-sync.md).

- `agpt_2b` clean (24.34 GiB, matches 2026-05-22 baseline).
- `moe_2b_ep` at new LBS=2 default clean (27.09 GiB, ~3,200 TPS,
  9.4% MFU — numerically identical to 2026-05-22).
- `moe_2b_ep` with `--debug.deterministic` **still fails**: PR #3146
  is incomplete (see PR #3146 note above).

### Action items

- File upstream issue on `pytorch/torchtitan` for the incomplete
  PR #3146 (commit message describes a `histc → bincount` swap that
  isn't in the diff).

---

## 2026-05-23 (39th sync — DebugMode numerics debugger)

Upstream merged in 1 commit (`19c567f76`).

- **[`19c567f76` — Debug Numerics with DebugMode (#3323)](https://github.com/pytorch/torchtitan/pull/3323).**
  Adds a DebugMode-based numerics debugger as `torchtitan/tools/numerics_debugging/{activation_tracer,compare_numerics}.py` (+~2 KLOC), with a corresponding `.claude/skills/numerics_debugging/` skill and a `make_fx_tracer.py` tweak under `experiments/graph_trainer/`. Captures per-op activations between two runs to spot bitwise/numeric drift (eager vs aot_fx_trace, FSDP vs no-FSDP, before vs after a refactor). **No ezpz replay needed** — pure tooling addition, doesn't change any code path ezpz exercises.

Side benefit: the new `numerics_debugging` skill is automatically picked up in this session and could be useful next time we need to bisect a silent loss-curve divergence (e.g. the kind of bug bf16-master RMSNorm-freeze was).

---

## 2026-05-22 (38th sync — MoE [6/n] dispatcher split + ChunkedCELoss/TP grad fix)

Upstream merged in 4 commits (`cfe97c605..c2a3771a4`).

**Upstream commits (4):**

- **[`da38566d3` — \[MoE\]\[6/n\] Extract local_reorder, split DeepEP/HybridEP
  dispatchers (#3389)](https://github.com/pytorch/torchtitan/pull/3389).**
  Direct continuation of PR #3386 (37th sync).
  - **Extract `local_reorder` helper** on `LocalTokenDispatcher` —
    deduplicates the histc/argsort/score-weighting block shared between
    `LocalTokenDispatcher.dispatch()` and
    `AllToAllTokenDispatcher.dispatch()`.
  - **Split `DeepEPTokenDispatcher` into `DeepEPTokenDispatcher` +
    `HybridEPTokenDispatcher`** — eliminates `comm_backend` string
    branching; config fields (`non_blocking_capacity_factor`,
    `pad_multiple`) are now class-specific. Callsites use `isinstance`
    instead of `getattr(..., "comm_backend", ...)`. The old `DeepEPMoE`
    wrapper class is gone entirely; the dispatcher classes now own that
    logic.
  - **Bundled DeepEP CUDA stream race fix** (#3413) — `torch.cuda.synchronize()`
    before `get_dispatch_layout`; XPU-irrelevant.

- **[`c2a3771a4` — \[loss\] Fix ChunkedCELoss + TP gradient placement
  mismatch (#3412)](https://github.com/pytorch/torchtitan/pull/3412).**
  Internal to `torchtitan/components/loss.py`. `GradAccumulator` used
  to wrap its buffer with the reference activation's placement
  (Replicate), but the buffer holds chunk gradients which can be
  `Partial(sum)` under TP/ColwiseParallel lm_head. Fix: capture
  `_placements` from the first added chunk. No ezpz replay needed.

- **[`b5852826b` — Fix imports for latest DeepEP (#3414)](https://github.com/pytorch/torchtitan/pull/3414).**
  Two-line change in `torchtitan/distributed/deepep/deepep.py`. XPU
  doesn't use DeepEP. **No-op for ezpz.**

- **[`3f721b4b5` — \[graph_trainer\] Fix fsdp_passes compat with
  BitsetAncestors from pytorch passes (#3416)](https://github.com/pytorch/torchtitan/pull/3416).**
  graph_trainer is a Meta-internal experiment that ezpz doesn't use.
  **No-op for ezpz.**

### Replays in ezpz

[**`d87729ad8` — fix(ezpz/moe): replay PR #3389 — isinstance dispatch on
token_dispatcher Config**](https://github.com/saforem2/torchtitan/commit/d87729ad8).
Mirror upstream's pattern in `experiments/ezpz/moe/model.py`:
- Import `DeepEPTokenDispatcher` and `HybridEPTokenDispatcher` from
  `torchtitan.models.common.token_dispatcher`.
- Replace `getattr(..., "comm_backend", "standard") in ("deepep", "hybridep")`
  with `isinstance(token_dispatcher_cfg, (DeepEPTokenDispatcher.Config,
  HybridEPTokenDispatcher.Config))`.
- Drop the dead `MoE → DeepEPMoE.Config` swap; `DeepEPMoE` no longer
  exists upstream.

ezpz doesn't actually exercise the deepep/hybridep paths on XPU (no
DeepEP kernels), but we keep the EP=1 guard so any CUDA-side ezpz
user who flips the dispatcher config to deepep gets the same error
semantics as upstream `deepseek_v3`.

### Smoke test results

Report:
[`docs/experiments/moe/sunspot/20260522-smoke-n2-38th-sync.md`](experiments/moe/sunspot/20260522-smoke-n2-38th-sync.md).

- `agpt_2b` (2N, LBS=1, GBS=24) — clean, 50 steps in 140 s, peak
  24.34 GiB, byte-comparable to the prior post-resync baseline.
- `moe_2b_ep` at LBS=1 — clean, peak 14.95 GiB vs the 37th-sync
  baseline of 15.03 GiB; TPS within 1.4%. Numerically equivalent.
- `moe_2b_ep` at LBS=2 — clean, peak 27.08 GiB, ~3,200 TPS, 9.4% MFU.
- `moe_2b_ep` at the previous registry-default LBS=16 OOMs on the
  bf16 vocab projection (`[16 × 8192, 256128] × 2 B ≈ 62.5 GiB`).
  Same pre-existing `_ep` vocab-projection OOM the 37th-sync
  follow-up flagged. Closed by pinning `moe_2b_ep` to LBS=2 in
  [`59354e43f`](https://github.com/saforem2/torchtitan/commit/59354e43f),
  mirroring
  [`f2cbc0327`](https://github.com/saforem2/torchtitan/commit/f2cbc0327)
  for `moe_debugmodel_ep`.

### Side issue surfaced

Stale `outputs/checkpoint/step-100` from a pre-PR-3159 (35th sync)
revision is no longer loadable post-merge (`Missing key in
checkpoint state_dict: layers.0.attention.qkv_linear.wk.weight.` —
Llama3 `qkv_linear` weight layout changed in PR #3159). Old checkpoint
backed up to `outputs/checkpoint-20260522-120005`; safe to delete.

---

## 2026-05-20 (37th sync — MoE clean DTensor boundaries + graph_trainer regional_inductor)

Upstream merged in `89987072b` (2 commits, `cfe97c605..963c20cba`).

**Upstream commits (2):**

- **`963c20cba` — [MoE][5/n] Refactor MoE to clean DTensor boundaries for
  shared/routed experts (#3386).** Substantial restructuring of how MoE
  TP/EP sharding is wired.
  - **Deleted:** `torchtitan/distributed/expert_parallel.py` (`ExpertParallel`,
    `TensorParallel` ParallelStyles), `ColwiseParallelWithGradPlacement`
    from `torchtitan/distributed/tensor_parallel.py`.
  - **New:** `torchtitan/models/common/moe_sharding.py` providing
    `set_moe_sharding_config(moe_cfg, *, enable_ep, enable_sp,
    expert_param_layout)` which populates `sharding_config` declarations
    on the MoE wrapper, router gate, shared experts (when present), and
    routed `GroupedExperts` — replacing the old
    `apply_moe_ep_tp(model, tp_mesh, ep_mesh)` parallelize-time pass.
  - **`GroupedExperts.parallelize`** is a new override that calls
    `super().parallelize(parallel_dims)` then
    `self.token_dispatcher.wire_meshes(ep_mesh=..., tp_mesh=...)` —
    keeping dispatch/combine mesh-aware at runtime under CooR precompile
    without an explicit pass.
  - **`MoE.forward` simplified.** Drops the explicit `isinstance(x,
    DTensor): x.to_local(grad_placements=Partial)` block at the
    top — replaced by the config-driven enter/exit redistribution
    declared on the MoE wrapper's `sharding_config`. Also splits the
    shared-experts addition out of `combine()`: shared experts run after
    `experts()` (or in parallel with DeepEP's async combine, then
    `sync_combine()` waits). Same math; different float reduction order.
  - **`GroupedExperts.forward`** drops the `shared_experts` kwarg
    accordingly.
  - **`parallelize_deepseekv3` simplified.** Old flow:
    ```
    if tp_enabled: model.parallelize(parallel_dims)
    if tp_enabled or ep_enabled: apply_moe_ep_tp(...)
    ```
    New flow:
    ```
    if tp_enabled or ep_enabled: model.parallelize(parallel_dims)
    ```
    No separate MoE pass — the config-based sharding handles both dense
    and MoE submodules in one call.
  - **`set_deepseek_v3_sharding_config`** gains `enable_ep: bool` and
    populates MoE submodule sharding configs unconditionally
    (`resolve_mesh` filters out disabled axes at runtime).
- **`83e490429` — [graph_trainer] Rework `full_inductor_compilation_pass`
  via `regional_inductor` + CPU attr migration (#3346).** Refactor of the
  graph_trainer inductor compilation pass. Touches
  `experiments/graph_trainer/` exclusively. No ezpz dependency.

**Replayed onto ezpz (PR #3386):**

| File | Change |
|------|--------|
| `experiments/ezpz/moe/parallelize.py` | Drop `apply_moe_ep_tp` function entirely + its imports (`ExpertParallel`, `TensorParallel`, `ColwiseParallelWithGradPlacement`). Replace the old "TP via configs / EP+TP via `apply_moe_ep_tp`" two-pass flow with the new single-pass `if parallel_dims.tp_enabled or parallel_dims.ep_enabled: model.parallelize(parallel_dims)`. Keep our local `apply_fsdp` (Aurora `ShardPlacementResult` workaround) and `disable_fsdp_gradient_division` (CCL backend) unchanged. |
| `experiments/ezpz/moe/sharding.py` | Add `enable_ep` kwarg to `set_moe_sharding_config` (avoids name collision with upstream's identically-named helper by importing the upstream one as `_set_moe_block_sharding_config`). Call upstream's helper on every MoE-enabled layer with `expert_param_layout = {"w1": Shard(1), "w2": Shard(2), "w3": Shard(1)}` (matches `EzpzGroupedExperts.{w1,w2,w3}` and upstream's `_GROUPED_EXPERTS_PARAM_LAYOUT` for deepseek_v3). |
| `experiments/ezpz/moe/model.py` | Pass `enable_ep=parallelism.expert_parallel_degree > 1` into our `set_moe_sharding_config` call from `update_from_config`. Update the surrounding comment. |
| `experiments/ezpz/moe/config_registry.py` | Stale-docstring scrub: drop the `apply_moe_ep_tp` mention from the 500M smoke-config docstring. |

**Replayed onto ezpz (PR #3346):** none. `experiments/graph_trainer/` only.

**Verification:** Smoke validated on Sunspot 2N, job 12467131:
- `moe_debugmodel` LBS=2 / 50 steps: loss 12.92 → 7.00 (Δ -0.01 vs 35th-sync
  baseline 7.01). Memory exactly matches baseline (16.99 GiB / 26.55%).
  TPS ~12,700.
- `moe_2b` LBS=1 / 50 steps: loss 12.95 → 6.11 (Δ -0.05 vs 35th-sync
  baseline 6.16). Memory 14.97 GiB (+0.5 GiB vs baseline 14.47 GiB,
  expected from new graph shape). TPS ~2,900.
- Both within ±0.05 nats of baseline at step 50 — fully consistent with
  PR #3386's documented behavior (graph reordering moves `shared_experts`
  add point, changing FP reduction order). `for_loop` expert backend
  fires the same warning count as baseline (5 + 17). No NaN/OOM. No
  recompilation events.

Full smoke report:
[`docs/experiments/moe/sunspot/20260520-smoke-n2-pr3386-replay.md`](experiments/moe/sunspot/20260520-smoke-n2-pr3386-replay.md).

---

## 2026-05-20 (36th sync — RL CI fixes only, no replay)

Upstream merged in `8b14712a8` (2 commits, `52a292d29..cfe97c605`).
Both touch `experiments/rl/` exclusively; ezpz does not subclass the
upstream RL actors.

**Upstream commits (2):**

- `cfe97c605` — [rl] use wandb=False in CI (#3408). README + integration
  test update to use `MetricsProcessor.Config(enable_wandb=True)` after
  the API change in #3391. Touches `experiments/rl/README.md` (+2) and
  `experiments/rl/tests/integration_tests.py` (+3). No ezpz dependency.
- `2057b5621` — [rl] fix CI timeout: vllm engine V2 teardown (#3365).
  Touches `experiments/rl/actors/generator.py` only (+5 −8). No ezpz
  dependency.

**Replayed onto ezpz:** none. Zero ezpz/{agpt,moe,qwen3} files touched
by these commits.

**Verification:** No code change required; no smoke needed. The build
state is identical to the 35th sync verification (which validated
agpt + moe end-to-end on Sunspot 2N on 2026-05-20).

---

## 2026-05-19 (35th sync — Full DTensor for Llama3 + graph_trainer churn + RL observability)

Upstream merged in `a14987132` (22 commits, `ee4e91a13..52a292d29`).

**Upstream commits touching ezpz-relevant core paths (2):**

- **`d64eabcce` — [Full DTensor] Config-based Full DTensor for Llama3
  (#3159).** Large refactor of the config-based sharding API.
  - **Breaking signature change:** `Module.parallelize(mesh)` →
    `Module.parallelize(parallel_dims)`. Each Module now self-resolves
    its SPMD submesh via `parallel_dims.get_module_mesh(axes)` from the
    axes referenced by its `NamedPlacement`s, instead of receiving a
    bare `tp_mesh` from the caller. Required for the new
    `--training.full_dtensor` mode where DP/CP/TP all participate in the
    Module's mesh.
  - `ShardingConfig` got new optional fields (`out_src_shardings`,
    `local_input_grad_placements`, `local_output_grad_placements`) —
    all default `None`, so existing callsites still construct the same
    shape.
  - `set_gqa_inner_attention_local_map` (which ezpz/agpt calls) is
    unaffected at the call site: arg-name changes (`xq/xk/xv` → `q/k/v`,
    `in_placements/out_placements` → `in_dst_shardings/out_src_shardings`)
    are internal to the helper.
  - `trainer.py` gained a `full_dtensor`-gated `parallelize_inputs` call
    and a `pred.to_local()` fallback for the `disable_loss_parallel`
    path. ezpz's `FaultTolerantTrainer` overrides `train_step` /
    checkpoint plumbing but does not override `_get_batch` or
    `forward_backward`, so it picks up both changes for free.
  - `apply_fsdp` gained an optional `dp_mesh_dims: DataParallelMeshDims`
    kwarg, only used under `full_dtensor`. ezpz/agpt's local `apply_fsdp`
    does not take the kwarg; that's fine since `full_dtensor=False` is
    the default and we don't enable it.
- **`a2a0d99e3` — RL: observability spans across trainer/generator/
  controller (#3234).** Adds `@sl.log_trace_span` decorators around RL
  actor methods. ezpz/rl uses its own `GRPOTask` registry and doesn't
  subclass the upstream RL actors, so this is a no-op for ezpz.

**Replayed onto ezpz:**

`experiments/ezpz/agpt/parallelize.py`:

- `model.parallelize(tp_mesh)` → `model.parallelize(parallel_dims)`.
  Async-TP plumbing still takes `parallel_dims.get_mesh("tp")` at the
  callsite. Module docstring updated.

`experiments/ezpz/moe/parallelize.py`:

- Same `model.parallelize(tp_mesh)` → `model.parallelize(parallel_dims)`
  swap on the dense-path TP application. `apply_moe_ep_tp` still takes
  the per-axis meshes directly (it doesn't go through
  `Module.parallelize`). Module docstring updated.

`experiments/ezpz/moe/model.py`: docstring reference to
`Module.parallelize(tp_mesh)` updated to `(parallel_dims)`.

**Other upstream commits in this batch (no ezpz impact):**

- `52a292d29` — [graph_trainer] Gate overlap_fsdp_ag_rs_pass behind a config flag (#3241).
- `8aa96acb2` — [graph_trainer] Fix eager SAC policy mm counter to reset at layer boundaries (#3397).
- `cf3c4312e` — [graph_trainer] Re-enable FlexAttention tests after upstream fix (#3394).
- `ebfceebe1` — [rl] Add TITO generator and gen metrics (#3391).
- `4238c3575` — Re-enable compile for gpt-oss integration tests (#3373).
- `6a1b334e8` — Fix optimizer state and module state coupling (#3356).
- `013890bf9` — [graph_trainer] Defer cudagraph compatibility check (#3355).
- `c9126af8a` — [graph_trainer] Add bucketing ops to precompile serialization filter (#3354).
- `9f9ae4d81` — [graph_trainer] Fix H100 CI failure from DeepEP compilation break (#3390).
- `a866d04a0` — Fix memory snapshot pickle protocol for memory visualizer compat (#3375).
- `cb7bf09ab` — ci: declare workflow-level `contents: read` on 2 workflows (#3367).
- `a670699ca` — [graph_trainer] Skip dense numerics tests due to upstream DTensor regression (#3372).
- `b64292ba1` — [graph_trainer] Remove fsdp_reshard_after_fwd_pass (#3370).
- `f0795f0d5` — [RL] - Enable experiment metrics (#3237).
- `8cdfdc236` — [rl] Better initial weight loading (#3318).
- `3781d1d2d` — Back out [graph_trainer] SAC remat fresh FakeTensor storage (#3358).
- `4614e3022` — [graph_trainer] Support multiple FSDP PGs in overlap_fsdp_ag_rs_pass (#3351).
- `d4a3c6349` — [graph_trainer] Generalize minimal_fx_tracer to module + optimizer roots (#3164).
- `cee49826b` — [graph_trainer] Fix SAC remat to produce fresh FakeTensor storage (#3343).
- `1690e0aea` — [cpu-offloading] Encode last consumer as dep arg in ao.wait (#3333).

**Verification:** `agpt/parallelize.py` and `moe/parallelize.py` import
cleanly via `.venv/bin/python -c "import ...parallelize"` post-replay;
both files round-trip the new `Module.parallelize(parallel_dims)`
signature.

**Compute-node smoke (2026-05-20, Sunspot 2N, job 12467124):** all four
configs ran 50 steps cleanly; no NaN/OOM, monotonic loss descent.

| Config              | LBS | Final loss | TPS/GPU | MFU    | Peak mem            | Report |
|---------------------|----:|-----------:|--------:|-------:|---------------------|--------|
| `agpt_debugmodel`   | 2   | 6.77       | ~37,500 | ~2.9%  | 2.60 GiB (4.06%)    | [link](experiments/agpt/sunspot/20260520-smoke-n2-postresync.md) |
| `agpt_2b` (LBS=1)   | 1   | 6.01       | ~6,100  | ~22.8% | 24.34 GiB (38.04%)  | [link](experiments/agpt/sunspot/20260520-smoke-n2-postresync.md) |
| `agpt_2b` (LBS=2)   | 2   | 6.12       | ~7,200  | ~27.0% | 44.73 GiB (69.91%)  | [link](experiments/agpt/sunspot/20260520-smoke-n2-postresync.md) |
| `moe_debugmodel`    | 2   | 7.01       | ~13,000 | ~9.0%  | 16.99 GiB (26.55%)  | [link](experiments/moe/sunspot/20260520-smoke-n2-postresync.md)  |
| `moe_2b` (LBS=1)    | 1   | 6.16       | ~2,900  | ~8.4%  | 14.47 GiB (22.62%)  | [link](experiments/moe/sunspot/20260520-smoke-n2-postresync.md)  |

`agpt_2b` at LBS=2 matches the historical [2026-04-25 n=2 baseline](experiments/agpt/sunspot/20260425-scaling-2b-venv-torch213.md)
(7,142 TPS / 27.6% MFU) within noise -- the resync did not perturb
steady-state throughput. The `for_loop` MoE expert backend (PR #13)
still fires correctly on XPU under the new parallelize signature.

Note: default `moe_2b()` is LBS=16, which OOMs on a 64 GiB Max 1550 tile
with a single 62.53 GiB allocation; LBS=1 is the safe per-GPU batch for
the for_loop path. This is a pre-existing XPU constraint, not caused by
the resync.

---

## 2026-05-13 (34th sync — RL vLLM v2 + repeat_interleave revert + graph_trainer AOT removal)

**Upstream commits (3 in batch):**

- `6f2fa2f9a` — [rl] switch to vllm v2 engine (#3330). Touches
  `experiments/rl/actors/generator.py`. ezpz/rl is unaffected — we
  don't subclass the upstream RL actors directly.
- `7b418ab30` — Revert "Avoid repeat_interleave output-size sync (#3274)"
  (#3335). Restores prior behavior in
  `models/common/token_dispatcher.py` after the optimization broke
  deepseek_v3 at TP=1 + PP=4 + EP=32 + AC=full. Pure revert; no new
  logic. ezpz/moe defers to upstream `LocalTokenDispatcher` so this
  silently restores correctness on any high-EP config.
- `1a22c2da1` — [graph_trainer] Remove deprecated AOT compile mode
  (#3327). Touches `experiments/graph_trainer/` only; no ezpz dependency.

**Replayed onto ezpz:** none. Zero ezpz/{agpt,moe,qwen3} files
touched by these commits.

**Verification:** `agpt_{debugmodel,2b,2b_real,20b,80b,80b_real}` and
`moe_{debugmodel,500m,10B_2B_sdpa}` all build cleanly post-merge.

---

## 2026-05-12 (33rd sync — `_grouped_mm` only path + graph_trainer churn)

**Upstream commits (12 in batch):**

- `b301dfa0` — **[MoE] Remove expert for-loop fallback (#3308).** Deletes
  `_run_experts_for_loop` and the `use_grouped_mm` config field from
  `models/common/moe.py`. `GroupedExperts._experts_forward` now always
  calls `torch._grouped_mm`. Upstream's argument is that
  `torch._grouped_mm` already provides a CUDA fallback path on pre-SM90
  hardware. **This breaks `experiments/ezpz/moe/model.py` which used
  `use_grouped_mm = False` as the XPU fallback** (XPU has no
  `_grouped_mm` kernel at all).
- `d57df092` — Make ChunkedCELoss support `torch.autograd.grad` (#3249).
- `5ca23a5d` — [GraphTrainer] Add Context Parallel support (#3305).
- `1a0fe3e3` — [graph_trainer] Refactor passes.py into focused modules (#3319).
- `e9dbff63` — [graph_trainer] Refactor selective activation remat to in-place (#3270).
- `2ceff82b` — [graph_trainer] Add log_timer utility for tracing step timing (#3311).
- `0fadde3b` — [graph_trainer] Fix AutoParallel input_fn to include positions tensor (#3315).
- `34801c00` — [graph_trainer] Improve SAC tagging and CPU offload pass metadata (#3321).
- `0b5e8998` — Fix precompile tests (#3316).
- `ca4c7f22` — [rl] Register customized config parser to vllm + less vllm config dependency (#3242).
- `7f602b98` — Add AGENTS.md symlinks for Codex usage (#3326).
- `7f070c93` — Enhance Lychee Link Checker (Resiliency & Performance) (#3203).

**Replayed onto ezpz:**

`experiments/ezpz/moe/`:

- New `experts.py` defining `EzpzGroupedExperts(GroupedExperts)` with a
  `compute_backend: Literal["for_loop", "grouped_mm"]` config field.
  Default `"grouped_mm"` defers to upstream; `"for_loop"` re-vendors
  the `_run_experts_for_loop` body that #3308 deleted, restoring the
  XPU / pre-SM90 path.
- New `make_ezpz_experts_config(...)` wrapper in `__init__.py` that
  calls upstream's `make_experts_config(...)` then re-wraps the result
  as `EzpzGroupedExperts.Config`. `_build_moe_layers` now threads a
  `compute_backend` kwarg (default `"grouped_mm"`) through to it.
- `model.py` `update_from_config` previously mutated
  `experts.use_grouped_mm = False` on pre-SM90 devices; that field no
  longer exists. Replaced with `experts_cfg.compute_backend = "for_loop"`
  guarded by `getattr(..., "compute_backend", "grouped_mm")` so the
  block is robust to future config-shape changes.

`experiments/ezpz/agpt/`: no replay needed; #3308's deletion was
MoE-only, and none of the other upstream commits in this batch touch
`models/llama3/` in a way ezpz/agpt depends on.

**Notes for downstream PRs:**

- Open PR #9 (Sam Wheeler — HSDP fix), #10 (Sam Wheeler —
  `batched_mm_padded` backend), and #11 (Nathan Nichols — MoE
  optimizations) all assume the pre-#3308 `GroupedExperts` shape
  (`use_grouped_mm` config field, `_run_experts_for_loop` importable
  from `models/common/moe.py`). They will need to rebase onto the
  resync'd `ezpz` and adapt to the `EzpzGroupedExperts` subclass.
  PR #10's `compute_backend` selector becomes a third option in
  `ExpertComputeBackend`; PR #11's expert-side optimizations layer
  onto the for-loop method here.

---

## 2026-05-05 (32nd sync — observability + MoE token-pad + CP fix + RL/graph_trainer churn)

**Upstream commits (11 in batch):**

- `b2cd149f` — Observability: structured logging + training instrumentation (#3176).
  Adds `torchtitan/observability/` module + `init_structured_logger()` in
  `Configurable.build()` + `@sl.log_trace_span(...)` decorators on hot
  paths in `trainer.py` and `validate.py`. Optional jsonl/database
  output for per-step timing spans.
- `d3414079` — [MoE] Pad token count to a multiple of `sp_size` in
  `AllToAllTokenDispatcher` (#3193). Internal correctness fix.
- `179d9e10` — CP AllGather on the wrong dimension (#3206). FlexAttention
  + Context Parallelism gather was on the wrong axis. Bug fix.
- `d3c96e80` — [mxfp8] Fix `MXFP8GroupedExpertsConverter` to actually swap
  `GroupedExperts` params (#3199). Quantization plumbing.
- `af8d2430` — [GraphTrainer] Skip identity-slice rewrite when start/end/step
  are dynamic Nodes (#3195).
- `706ed8d8` — [GraphTrainer] Annotate generated FX code with user source
  lines (#3194).
- `080c1d4c` — [GraphTrainer] Annotate loss region with `module_fqn` (#3207).
- `0b6a29e6` — [rl] Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  for monarch RDMA (#3221).
- `2ae13405` — [rl] add `import torch` to provisioner bootstrap (#3220).
- `0b148a2e` — [CI] Run integration tests in parallel (#3144).
- `f522db0e` — Fix commands in DSV3 readme (#3116).

**Impact on ezpz:**

- **Observability (#3176)** does NOT affect us at runtime — our
  `FaultTolerantTrainer.__init__` inlines the upstream `Trainer.__init__`
  body (FT-required) rather than calling `super().__init__()`, so the
  new `init_structured_logger()` and `@sl.log_trace_span` decorators
  on upstream `Trainer` methods don't propagate. Likewise, our
  `EzpzValidator.validate()` overrides `Validator.validate()` entirely,
  so the new `@sl.log_trace_span("eval")` decorator on the parent
  doesn't apply to us. Imports verified to still work after merge.
  If we want trace spans on the ezpz path we'd need to add the
  decorators ourselves; not blocking.
- **MoE token-pad (#3193)** — affects `AllToAllTokenDispatcher` only;
  our `moe_10b_2b_sdpa_ep` uses `comm_backend="standard"` →
  `All2AllTokenDispatcher` (different code path). Safe to take.
- **CP AllGather fix (#3206)** — ezpz never enables CP
  (`cp_degree` always 1 in our configs). Safe to take.
- All other commits scoped to `experiments/{rl,graph_trainer}/`,
  quantization, CI, or docs. ezpz doesn't depend.

**Replay into ezpz `agpt/` or `moe/`:** None needed. No
`models/llama3/`, `models/deepseek_v3/`, or `models/common/` changes
in this batch beyond the internal token_dispatcher fix.

Merge commit: `0690e67e`.

---

## 2026-05-04 (31st sync — precompile revert + PP refactor + CI baseline)

**Upstream commits:**

- `60650551` — Revert precompile PRs (#3107, #3178) to fix CI (#3212).
  Backs out the bucketing-pass + FlexAttention precompile bitwise
  deterministic-tests changes that were merged in the 2026-05-03 sync;
  the upstream PyTorch fix (pytorch/pytorch#181529) those changes
  depended on still hadn't landed on `viable/strict`.
- `db5da4b6` — Refactor pipeline parallel helpers for graph PP reuse
  (#2724). Extracts pipeline metadata + module splitting + PP
  rank-to-stage mapping out of `pipeline_llm` so graph PP can reuse
  it. Renames `build_pipeline_schedule`,
  `generate_llm_fqn_per_model_part`, and `pipeline_module_split` to
  private (underscore prefix) — they are no longer public API.
- `95ae6a00` — `ci: regenerate qwen3_moe_rocm_mi350x.txt baseline with
  actual MI350 losses (#3196)`. Loss-baseline file regen for ROCm
  MI350 CI; no production impact.

**Impact on ezpz:** None.

- The PP refactor (#2724) only renamed helpers used internally by
  `pipeline_llm`; ezpz doesn't import any of `build_pipeline_schedule`,
  `generate_llm_fqn_per_model_part`, or `pipeline_module_split`
  directly (verified via grep across `experiments/ezpz/`).
- The revert (#3212) only touches
  `experiments/graph_trainer/precompile_main.py`,
  `experiments/graph_trainer/tests/test_bitwise_deterministic.py`,
  and `experiments/transformers_modeling_backend/pipeline.py` — none
  of which ezpz depends on.
- The baseline regen is test data only.

Merge commit: `fc10014d`.

---

## 2026-05-03 (30th sync — graph_trainer only)

**Upstream commits:**

- `627126fe` — Enable bucketing pass in precompile path (#3107)
- `d27d0c36` — [GraphTrainer] Fix FlexAttention precompile bitwise
  deterministic tests (#3178)
- `14f6a75e` — [GraphTrainer][AutoDev] Fuse RMSNorm kernels via regional
  Inductor compilation (#3132). Adds `performance_passes.py` and a new
  test file.

**Impact on ezpz:** None. All 3 commits scoped to
`experiments/graph_trainer/`. ezpz doesn't depend.

**Replay status:** Clean fast-forward merge. ezpz imports verified.
No baseline re-check needed.

---

## 2026-05-01 (29th sync — RL vLLM compile-time + graph_trainer skill)

**Upstream commits:**

- `2e5f1371` — Improve compilation time (~50s → ~15s for vLLM) (#3145).
  Scoped to `experiments/rl/{generate,grpo,models}` + RL tests.
- `37fe579a` — [GraphTrainer] Add weekly report generator skill (#3200).
  One new file: `experiments/graph_trainer/.claude/weekly_report.md`.

**Impact on ezpz:** None.
- ezpz doesn't depend on `experiments/rl/` or `experiments/graph_trainer/`.

**Replay status:** Clean fast-forward merge. ezpz imports verified.
No baseline re-check needed.

---

## 2026-05-01 (28th sync — graph_trainer qwen3 + CI lint + FT llama3 attn_backend)

**Upstream commits:**

- `2b935fa3` — [GraphTrainer] Add Qwen3 MoE bitwise deterministic tests
  and fix weight-tying gradient bug (#3174). graph_trainer + qwen3 only.
- `70340f4e` — [CI] Use replace-imports-with-any (#3180). Removes
  `# pyrefly: ignore[missing-module-attribute]` comments now redundant
  with the new pyrefly config; lint-only.
- `9732db4a` — [ft] Forward attn_backend to llama3 config functions
  (#3182). 2-line change to `experiments/ft/llama3/__init__.py` —
  adds `attn_backend="sdpa"` parameter to that registry. We don't use
  `ft.llama3` (we have our own `ezpz.agpt.model_registry`).

**Impact on ezpz:** None. All three commits scoped to graph_trainer,
linter cleanups, or ft.llama3 (which we don't use).

**Replay status:** Clean fast-forward merge. ezpz imports verified.
No baseline re-check needed (no code path that affects agpt/moe changed).

---

## 2026-04-30 (27th sync — HybridEP comm_backend cleanup + autoparallel/deepseek_v3 deletion)

**Upstream commits (truly new — most others were patch-equivalent
duplicates of the ETP deprecation already replayed at `97e44a5a`):**

- `f7940d5b` — Remove stale `experiments/autoparallel/deepseek_v3/`
  (#2271). 5 files / 534 lines deleted.
- `0138bde8` — [HybridEP] Read comm_backend from model config instead
  of ParallelismConfig (#3177). Adds optional `non_blocking_capacity_factor`
  kwarg to deepseek_v3 model factory functions.
- `115f4c9c` — [HybridEP] Enable HybridEP with graph_trainer (#3007).
  graph_trainer + 1-line deepseek_v3/__init__.py change.

**Impact on ezpz:** None.
- ezpz/moe doesn't use `non_blocking_capacity_factor` (no HybridEP code path)
- We don't depend on `experiments/autoparallel/deepseek_v3/`

**Merge conflicts:** Two modify/delete conflicts on
`experiments/autoparallel/deepseek_v3/{config_registry,parallelize_deepseekv3}.py`
— ezpz had local mods to those files from a previous ETP-removal patch
sequence. Accepted upstream's deletion (we don't use the dir).

**Replay status:** Clean otherwise. ezpz imports verified.
No baseline re-check needed (no code path that affects agpt/moe changed).

---

## 2026-04-30 (26th sync — ETP deprecation + varlen window + CI)

**Upstream commits:**

- `c4b409af` — [MoE][4/n] deprecate expert tensor parallel (ETP) (#3167).
  Removes `ExpertTensorParallel` class, the `etp` mesh axis, the
  `expert_tensor_parallel_degree` and `expert_parallel_comm_backend`
  config fields, and all ETP references across models and experiments.
  `comm_backend` now lives on `moe.experts.token_dispatcher`, not on
  `parallelism`.
- `a2b5ee66` — varlen window size configurable (#3173). Adds
  `attention.window_size` plumbing for varlen attention; additive only.
- `47d3045f` — [CI] Move lychee link checker to nightly workflow (#3158).
- `efebe6de` — [GraphTrainer] Remove SAC peak-memory test from H100 CI
  (#3170).
- `4e48ebec` — Add MoE loss comparison guard to CI workflow (#3081).
- `6495ce57` — [GraphTrainer] Clean up precompile bitwise deterministic
  tests (#3169).

**Impact on ezpz:**
- `ezpz/moe/parallelize.py` imported `ExpertTensorParallel` (now deleted)
  and called `apply_moe_ep_tp` with `etp_mesh` / `ep_etp_mesh` kwargs that
  the upstream signature no longer accepts.
- `ezpz/moe/model.py` referenced `parallelism.expert_parallel_comm_backend`
  to decide between `MoE` and `DeepEPMoE`. That config field is gone;
  `comm_backend` now lives on the per-expert token_dispatcher config.
- `ezpz/agpt/`, `ezpz/qwen3/`, and the trainer don't touch ETP — no
  changes needed.
- `experiments/ezpz/moe_runs/*.json` snapshots still contain
  `"expert_tensor_parallel_degree": 1`. These are historical run records;
  not stripping. Re-running them through the current config parser would
  fail and require dropping that key.

**Replay (mirrors `c4b409af` deepseek_v3 changes):**
- `ezpz/moe/parallelize.py`:
  - Drop `ExpertTensorParallel` from the `expert_parallel` import.
  - `apply_moe_ep_tp(...)` signature: drop `etp_mesh` / `ep_etp_mesh`.
  - Drop the `etp_mesh`/`ep_etp_mesh` kwargs from the call inside
    `parallelize_moe`.
  - Collapse the `experts_mesh` / `experts_plan` selection to the two
    surviving cases (EP disabled → TP-shard experts; EP enabled →
    `ExpertParallel()`).
- `ezpz/moe/model.py`:
  - Read `comm_backend` from
    `layer_cfg.moe.experts.token_dispatcher` (matches upstream
    `deepseek_v3.model`).
  - Raise `ValueError` if `comm_backend in ("deepep","hybridep")` and
    `expert_parallel_degree == 1` (was previously implicit — now
    explicit, matching upstream).
  - Keep the `MoE → DeepEPMoE.Config` swap inside the same
    `comm_backend in ("deepep","hybridep")` branch.

**Verified:** Both modified files parse with `ast.parse`. No baseline
re-check needed — ezpz/moe doesn't run on Aurora MoE production right
now (MoE SIGABRT regression is still open from the 00b7f569 sync), so
there's nothing live to break. Will smoke when MoE work resumes.

---

## 2026-04-29 (25th sync — graph_trainer + VLM deletion + minor)

**Upstream commits:**

- `01a8b068` — [GraphTrainer] log_activation_memory_policy (#3062)
- `533795ee` — [graph_trainer] simple_fsdp unconditional for NGPU=1 (#3148)
- `a890192e` — [graph_trainer] Fix test_trace_module backends (#3155)
- `ef8e2820` — [graph_trainer] Async TP graph pass (#3129)
- `0ad6772a` — [VLM] deprecate vlm experiment (#3151) — deletes 18
  files / 2,248 lines from `experiments/vlm/`
- `a3a01604` — fix(hf_datasets): ChatDataset shuffle before split (#3131)
- `719085ae` — Use CrossEntropyLoss for torchcomms 3D compile tests (#3157)

**Impact on ezpz:** None.
- 4 commits scoped to `experiments/graph_trainer/`
- 1 commit deletes `experiments/vlm/` (ezpz doesn't depend on it)
- 1 commit touches `hf_datasets/text_datasets.py` for `ChatDataset`
  only — ezpz uses BlendCorpus or HuggingFaceFW streaming via
  `datasets.py`, not `ChatDataset`
- 1 commit is a single line in CI test config

**Replay status:** Clean fast-forward merge. ezpz imports verified.
No baseline re-check needed (no code path that affects agpt/moe changed).

---

## 2026-04-29 (24th sync — All2All token dispatcher consolidation)

**Upstream commits:**

- `20628f4e` — [MoE][3/n] consolidate EP=1 and EP>1 to all use
  `All2AllTokenDispatcher` (#3125). `AllToAllTokenDispatcher` now falls
  back to `LocalTokenDispatcher` behavior when `ep_mesh is None`.
  Default `comm_backend` changed from `None` to `"standard"`; the
  `None` → `LocalTokenDispatcher` path was removed entirely.
  `make_token_dispatcher_config` and `make_experts_config` now require
  a non-None `comm_backend`.
- `35c5d529` — [graph_trainer] Remove `apply_graph_ac` (#3147). No
  impact on ezpz (graph_trainer experiment only).

**Impact on ezpz:** `moe.model_registry()` and `_build_moe_layers()` both
defaulted `moe_comm_backend` to `None`, which is no longer valid. At
build time `make_experts_config(comm_backend=None)` raises
`ValueError: Unknown comm_backend: 'None'`.

**Replay:**
- `moe/__init__.py`: change both `moe_comm_backend: str | None = None`
  defaults to `moe_comm_backend: str = "standard"` (mirrors upstream).
  Drop the `if moe_comm_backend is not None` guard around the
  token-dispatcher rebuild loop in `model_registry` — the dispatcher is
  now always rebuilt with the user's chosen backend (or "standard" by
  default), and EP=1 is handled by the dispatcher's local-fallback path
  rather than by skipping the rebuild.

**Verified:** Both smoke configs build cleanly post-fix. Existing
`moe_debugmodel_ep` / `moe_7b_ep` configs that pass
`moe_comm_backend="standard"` explicitly continue to work (kwarg is
redundant but not wrong).

---

## 2026-04-29 (post-21st-sync regression: legacy `output.weight` checkpoint load)

**Symptom:** Both 20B continuation jobs (`8453664`, `8453665`) crashed at
checkpoint load with:

    RuntimeError: Missing key in checkpoint state_dict: lm_head.weight

The 2B continuation (`8453662`) didn't reach checkpoint load — segfaulted
on a bad node during `set_determinism()` — but would have hit the same
error.

**Root cause:** The 21st upstream sync (`b6c04698`) renamed
`Decoder.Config.output` to `Decoder.Config.lm_head`, replayed in ezpz
`03b9f486`. Production checkpoints saved before the rename
(2B step 33,700+, 20B step 4,100+) still have `output.*` keys on disk;
new code's state_dict has `lm_head.*`; DCP's strict matching crashes.

**Fix:** `experiments/ezpz/checkpoint_compat.py` monkey-patches
`CheckpointManager.dcp_load` to bridge the rename on the fly:
state_dict keys go `lm_head.*` -> `output.*` before `dcp.load`, then
back to `lm_head.*` before `model.load_state_dict`. Applied from
`train.py`, idempotent.

Verified by `utils/verify_checkpoint_compat.py` round-tripping the 20B
step-4100 `output.weight` ([256128, 5120] bf16): non-zero, finite,
sensible values.

**Lesson:** Field renames in `Decoder.Config` (and any upstream `Module`
attribute rename) need an explicit DCP backwards-compat plan. Going
forward, when replaying upstream renames, also save a fresh checkpoint
with the new naming on the next opportunity so the shim can be removed.

**Removal (2026-04-29):** Shim and verifier deleted. The bf16-tainted
production checkpoints were renamed to `*.bf16-norm-bug-20260429` and
will not be resumed from. The shim was unconditionally renaming
`lm_head` -> `output` in the load path, which BROKE auto-resume from
new-style checkpoints (the new restart writes `lm_head.*` keys; the
shim rewrote the load state_dict to ask for `output.*`, missed the
on-disk metadata). Files removed: `experiments/ezpz/checkpoint_compat.py`,
`experiments/ezpz/utils/verify_checkpoint_compat.py`. The
`patch_checkpoint_manager()` call removed from `train.py`.

---

## 2026-04-28 (23rd sync — graph_trainer experiment + ROCm CI only)

**Upstream commits:**

- `a364b4b4` — [GraphTrainer] Add full inductor compilation pass (#3141)
- `9ed1a028` — [graph_trainer] Joint graph bucketing + prefetching
  composes with SAC (#3056)
- `69761ca8` — [ROCm][CI] Re-disable experimental workflows for ROCm (#3140)

**Impact on ezpz:** None. All three commits are scoped to
`experiments/graph_trainer/` and `.github/workflows/`. No changes to
`models/`, `protocols/`, `distributed/`, `trainer.py`, or `components/`.

**Replay status:** Clean fast-forward merge. ezpz imports verified.
No baseline re-check needed (no code path that affects agpt/moe changed).

---

## 2026-04-28 (22nd sync — quantize-on-config, LocalMapInnerAttention removal, MeshAxisName rename)

**Upstream commits:**

- `6348d93d` — quantize on config instead of on model (#3127). Removes
  `protocols/model_converter.py`, drops `model_converters` from
  `JobConfig`, drops `model_converters` argument from all `parallelize_*`
  signatures. New pattern: pass `quantization=[Float8LinearConverter.Config(...)]`
  to `model_registry()`, which applies the converter to the model
  config at registry time. Also removes `FaultTolerantModelSpec` from
  `protocols/model_spec` (moved to `experiments/ft/config/job_config`).
- `b9e33527` — [Module] Remove LocalMapInnerAttention, use static
  LocalMapSpec (#2986). Replaces runtime DTensor detection in
  `LocalMapInnerAttention` with a static `LocalMapConfig` set on the
  inner-attention sharding_config via
  `set_gqa_inner_attention_local_map`. All inner attention types
  (SDPA, FlexAttention, Varlen) now inherit `Module` directly.
- `053dbf9a` — [Module] Rename MeshDimName → MeshAxisName (#3113).
  Transparent for ezpz: our sharding files use the helpers from
  `decoder_sharding`, which were updated upstream.
- + 6 minor (linter fixes, MATH backend revert, qwen3 RL cleanup,
  GraphTrainer CI, claude.md updates).

**Impact on ezpz:** Three breaking changes hit at import time:

- `from torchtitan.protocols.model_converter import ModelConvertersContainer` → gone
- `from torchtitan.models.common.attention import LocalMapInnerAttention` → gone
- `from torchtitan.protocols.model_spec import FaultTolerantModelSpec` → gone

**Replay:**

- `agpt/parallelize.py` — drop `model_converters` parameter and import.
- `moe/parallelize.py` — drop `model_converters` parameter and import.
- `agpt/__init__.py` — `LocalMapInnerAttention` → `Module` for the
  `SoftcappedFlexAttention` base class and `_ezpz_get_attention_config`
  return-type annotation. Re-import `FaultTolerantModelSpec` from its
  new location.
- `moe/model.py` — `LocalMapInnerAttention.Config` →
  `Module.Config` for `Attention.Config.inner_attention`.
- `agpt/sharding.py`, `moe/sharding.py` — add
  `set_gqa_inner_attention_local_map(layer_cfg.attention.inner_attention)`
  call so inner attention gets the static `LocalMapConfig` it now
  needs (replaces the runtime DTensor wrapper).
- `moe/__init__.py` — extend `model_registry` with `quantization`
  parameter; apply each converter to the config via `q.build().convert(config)`.
- `moe/config_registry.py` — `moe_671b()` re-registers via
  `model_spec=model_registry("671B", quantization=[...])` instead of
  setting `cfg.model_converters`.
- `ezpz/trainer.py` — drop the runtime model_converters build/convert,
  the `register_step_post_hook` post_optimizer_hook, the
  `model_converters=` kwarg from both `parallelize_fn` and `pipelining_fn`
  call sites, the `QuantizationConverter` import. Switch
  `has_quantization` to read from `model_config` via
  `torchtitan.components.quantization.utils.has_quantization`.

**Verified:** agpt + moe import cleanly, `model_registry()` returns
valid spec, `update_from_config` populates sharding_config including
the new `LocalMapConfig` on inner_attention.

---

## 2026-04-27 (21st sync — config-based DTensor sharding)

**Upstream commits:**

- `1786292d` — [Module][Full DTensor] Config-based sharding infrastructure with
  llama3 adoption (#2963). Replaces string-keyed plan dicts with `ShardingSpec`
  attached to `Module.Config`. New `protocols/sharding.py` and `protocols/module.py`.
- `8e820844` — [Module][Full DTensor] Config-based sharding for Qwen3, Llama4,
  DeepSeek V3, GPT-OSS (#2969). All models now use `Module.parallelize()`.
- `786e26f8` — ChunkedCELoss (#2937)
- `1ea5d511` — [AutoParallel] Use autoparallel_backend() for torch.compile (#3114)
- + 5 more (GraphTrainer, Qwen3VL, vLLM cache reset)

**Impact on ezpz:**

- **llama3/parallelize.py REWRITTEN** — old `parallelize_module()` API replaced
  with `Module.parallelize()`. The ezpz `agpt/parallelize.py` still uses the old
  API. **Needs replay** but deferred to avoid breaking running experiments.
- **New files:** `llama3/sharding.py`, `deepseek_v3/sharding.py`,
  `protocols/sharding.py`, `protocols/module.py`, `common/decoder_sharding.py`
- **trainer.py** — 53 lines changed (model initialization flow updated)

**Replay status (2026-04-28):**

- **agpt: DONE.** Three commits land the replay:
  - `03b9f486 fix(ezpz): unbreak imports after upstream loss/lm_head refactors`
    — drops `build_cross_entropy_loss` imports / `build_loss_fn=` kwargs
    (#2937 removed both); renames `output=Linear.Config(...)` to
    `lm_head=Linear.Config(...)` (Decoder.Config field rename); ezpz/trainer.py
    switches to `config.loss.build(compile_config=...)`.
  - `472f4743 refactor(ezpz/agpt): replay config-based DTensor sharding`
    — adds `agpt/sharding.py` and `agpt/model.py`. Rewrites
    `agpt/parallelize.py` as a thin orchestrator that calls
    `model.parallelize(tp_mesh)` instead of the old `parallelize_module()` plan.
    Preserves agpt-specific `disable_fsdp_gradient_division`
    (force_sum_reduction for CCL/XPU), the `capture_scalar_outputs=False`
    reset after compile, and the `[norm, lm_head]` joint FSDP grouping.
  - Smoke test: 2N debug-scaling pending.
- **moe: DONE.** Same shape of replay against deepseek_v3:
  - New `moe/sharding.py` mirrors `models/deepseek_v3/sharding.py` but
    binds against `experiments.ezpz.moe.model.Attention` (our MLA Attention
    is a separate class from upstream's, even though structurally identical).
  - `moe/model.py`: extends `moeModel.Config.update_from_config` to call
    `set_moe_sharding_config` after the existing rope/MoE sync logic.
  - `moe/parallelize.py`: rewritten as a thin orchestrator. The non-MoE TP
    plumbing (`apply_non_moe_tp` with manual ColwiseParallel/RowwiseParallel/
    SequenceParallel/PrepareModuleInput plans for attention + norms +
    dense FFN) is gone — replaced by `model.parallelize(tp_mesh)`.
    `apply_moe_ep_tp` is kept (mirrors upstream — MoE blocks are still
    parallelized at parallelize-time, not via sharding_config).
    `apply_fsdp` is still inlined locally (avoids `ShardPlacementResult`
    import which doesn't exist in Aurora's PyTorch) and includes the
    Shard(0) fallback for when expert hidden dim isn't FSDP-divisible.
    Per-block `block.compile(backend=...)` workaround kept (XPU can't do
    fullgraph=True for MoE routing).
  - Smoke build verified locally: imports + `update_from_config` +
    `Module.parallelize(1D mesh)` + meta `cfg.build()` all work on the
    moe `debugmodel` flavor.
- **qwen3: REMOVED.** ezpz/qwen3 had drifted significantly from upstream
  (single `layer:` template vs upstream's `layers: list`, custom
  `__init__(config, *, layer_id, dim, n_layers)` block constructor,
  `self.output` references, no `lm_head` slot in qwen3_configs entries).
  README marked it "still under development" and no production jobs ever
  used it. Removed via `git rm -r` rather than carry the replay debt
  for unused code. History retained — restore with
  `git checkout <parent-sha> -- torchtitan/experiments/ezpz/qwen3` if
  ever needed.

**Float8 tensorwise TP:** dropped from `agpt/parallelize.py` during the
replay. The new sharding API doesn't expose an equivalent yet
(`Float8ColwiseParallel` etc. were tied to the old plan API). We weren't
using it in production. Revisit when float8 lands in the new API upstream.

---

## 2026-04-27 (20th sync)

**Upstream commits:**

- `cca3be50` — Fix reproducible training resume across epoch boundaries for
  map and streaming datasets (#3008). Fixes two bugs in `HuggingFaceTextDataset`
  and `ChatDataset` related to checkpoint resume after epoch re-loop.
- `a7205469` — [rl] Env rollout based + controller refactor (#3073). Refactors
  upstream RL experiment (not our ezpz/rl).

**Changes required in ezpz:** None. Clean merge. Verified `_validate_dataset`
still exists and is compatible with our `datasets.py` monkey-patch.

**No replay needed:** Neither commit touches `llama3/`, `deepseek_v3/`, or
`models/common/`.

---

## 2026-04-25 (19th sync)

**Upstream commits:**

- `bfc2914b` — Use `current_accelerator` for device in AutoParallel calls to enable XPU (#3092)
- `eb518a1d` — Add fused QKV support to Qwen3-VL state_dict_adapter (#3102)
- `42b73643` — Fix SAC test compatibility with PyTorch indexed storage (#3098)
- + 11 more (GraphTrainer CPU offload, CI, ROCm)

**Changes required in ezpz:** None. Clean merge.

---

## 2026-04-24 (18th sync — torch 2.10 XCCL fixes + DTensor TP revert)

**Context:** Investigating and fixing training hangs on torch 2.10
(`aurora_frameworks-2025.3.1`) with the XCCL backend. Also fixing
80B TP=2 regression on both torch 2.10 and 2.13.

**Root causes identified:**

1. **Blendcorpus barrier hang (torch 2.10):** The XCCL C++ backend
   ignores `opts.device` for `barrier()` operations, defaulting all
   ranks to device 0. This causes hangs during blendcorpus dataset
   building which uses `barrier(group=mpu.get_data_parallel_group())`.

2. **TP=2 forward pass hang (torch 2.10):** Removing the IPEX import
   (`import intel_extension_for_pytorch`) in the 17th sync broke TP
   collectives on torch ≤2.10, because IPEX provides XPU operator
   overrides that the XCCL backend relies on.

3. **80B AC + compile crash (torch 2.13):** The `use_local_output=False`
   DTensor TP change (`fc3880c5`) causes `DeviceMesh` objects to leak
   into the AOT autograd saved state, triggering
   `AssertionError: expected all tensors_saved_with_vc_check to be Tensors`.
   This is a PyTorch bug — AC's version check doesn't handle non-Tensor
   objects from DTensor-parallelized modules.

**Changes required in ezpz:**

| File | Change | Commit |
|------|--------|--------|
| `blendcorpus/blendcorpus_builder.py` | Gloo barrier workaround: temporarily replace `dist.barrier` with a CPU-side gloo barrier during dataset building when `bound_device_id` is None | `312045b3` |
| `train.py` | Re-enable IPEX import gated on `torch.__version__ < "2.11"` | `312045b3` |
| `agpt/parallelize.py` | **Revert** `fc3880c5`: restore `use_local_output=enable_sp` (pre-DTensor-TP default). The full DTensor TP requires a newer torch that handles DeviceMesh in AC autograd context. | `8e9ebc23` |

**Verified on Sunspot (2 nodes, 24 XPU tiles):**

| Config | torch 2.10 | torch 2.13 |
|--------|-----------|-----------|
| agpt_2b (TP=1) | PASS | PASS |
| agpt_20b (TP=1) | PASS | PASS |
| agpt_80b (TP=2, compile) | PASS (22.5 tflops) | PASS w/o compile (49 tflops) |
| moe_7b | PASS | PASS |

**80B status by torch version:**

| Torch | Compile | AC | Result |
|-------|---------|-----|--------|
| 2.10 | ON | full | PASS (with IPEX, 22.5 tflops, 7.5% MFU) |
| 2.13 | OFF | full | PASS (49 tflops, 16.4% MFU) |
| 2.13 | ON | full | FAIL (DeviceMesh in AOT autograd — upstream bug) |

---

## 2026-04-23 (17th sync)

**Upstream commits:**

- `703b72ef` — [full DTensor] Use all DTensor for Qwen3 and llama4 at TP region (#2149)
- `95a25420` — fix(hf_datasets): shuffle HuggingFaceTextDataset on re-loop (#3023)
- `d52d2475` — [rl] Generator refactor (#3001)
- `a676793a` — [rl] Rename inference example (#3045)
- `0de35f96` — [profiler] Suppress Callable field from tyro CLI parsing (#3038)
- + 10 more (GraphTrainer, CI, ROCm)

**Breaking changes in `models/llama3/parallelize.py`:**

- `use_local_output=True` → `use_local_output=False` for embed_plan,
  norm_plan, and rowwise_output_plan. TP now keeps tensors as DTensors
  instead of converting to plain tensors.

**Changes required in ezpz:**

| File | Change | Commit |
|------|--------|--------|
| `agpt/parallelize.py` | Replay `use_local_output=False` for embed, norm, rowwise plans | `fc3880c5` |
| `agpt/parallelize.py` | **Reverted in 18th sync** — causes hangs on torch 2.10 and AC crash on 2.13 | `8e9ebc23` |

---

## 2026-04-20 (16th sync)

**Upstream commits:**

- `dd0cbe65` — [GraphTrainer] Remove compile_with_inductor annotation from qwen3 FlexAttention (#3019)
- `ac154e60` — [GraphTrainer] Remove unused standalone GraphTrainerConfig dataclass (#3018)
- `526c7de2` — [graph_trainer] Fix test_bitwise_deterministic AttributeError (#3028)
- `4a61bd87` — Update .md files to use --profiler (#3026)
- `a91f57bd` — [graph_trainer] Add MANIFESTO.md (#3014)
- `fdc71a30` — [Typing] Remove unused pyrefly ignores in token_dispatcher (#3027)

**Files changed:** `experiments/graph_trainer/`, docs, `models/common/token_dispatcher.py` (typing only).

**Changes required in ezpz:** None. Clean merge.

---

## 2026-04-20 (15th sync)

**Upstream commits:**

- `08da451f` — [Typing] Pyrefly: --remove-unused-ignores during pre-commit (#3010)

**Files changed:** 35 files across models/ and distributed/ — removing unused
`# pyrefly: ignore` comments. No functional changes.

**Changes required in ezpz:** None. Clean merge.

---

## 2026-04-18 (14th sync)

**Upstream commits:**

- `7cec1660` — [MoE][2/n] Move EP setup from trainer to config registry and model_registry params (#2960)
- `ca46729b` — Update models/README.md (#3012)
- `518d708f` — Todo debt tracker updates (#3011)
- `114fedc9` — fix CP n_tokens_seen overcounting (#2990)
- `de7fc7bd` — Feature/rfc 2408 compute comms overlap (#3020)
- `ab87f584` — ci: install torchvision in transformers backend workflow (#3002)
- `09ea7d8e` — [MoE][1/n] Introduce token dispatcher and replace token reorderer (#2842)
- `c88cfb03` — Add Qwen3-14B GRPO config and 14B_varlen model registry entry (#2941)
- `d2adfd99` — [graph_trainer] Add debug_graph_passes config (#3003)
- `ae4b0a7a` — Unbreak CI (#2999)
- `ad7a8653` — Fused QKV GQAttention implementation (#2878)
- `aa0214b8` — [ROCm][CI] Turn off ROCm CI on experimental workflows (#2874)
- `2359c2f5` — [rl] Remove test_bitwise_identity.py (#2997)
- `bdbd6582` — [graph_trainer] Copy forward metadata to backward subgraphs (#2875)
- `132683a2` — [graph_trainer] Add torch.no_grad() and graph-based SAC (#2766)
- `5e46f947` — [GraphTrainer] Enable FlexAttention bitwise deterministic tests (#2989)
- `d3918650` — [ez][graph_trainer] fix comment (#2996)
- `1c4e18b9` — [graph_trainer] Add CooR precompile support (#2975)
- `ea6a1cdf` — [GraphTrainer] Fix custom_codegen_pass hang (#2993)
- `dafd87a9` — [GraphTrainer] Disable custom_codegen_pass and clean up (#2992)
- `8b7a85cc` — Fix memory snapshot hang with aot_fx_trace codegen (#2991)
- `be09084b` — [GraphTrainer] Add custom codegen pass with dual-path profiling (#2955)

**Major breaking changes:**

1. **Attention backend abstraction:** `inner_attention`/`mask_type` params replaced
   with `attn_backend: str` + `get_attention_config()` helper
2. **Fused QKV:** `GQAttention.Config` now uses `qkv_linear: BaseQKVLinear.Config`
   instead of separate `wq`/`wkv`; `make_gqa_config()` takes `fuse_qkv` param
3. **Token dispatcher:** `TokenReorderer` removed from `moe.py`, replaced by
   `LocalTokenDispatcher` and subclasses in new `token_dispatcher.py`
4. **Profiler migration:** `torchtitan.tools.profiling` deleted, replaced by
   `torchtitan.tools.profiler` with `Profiler` class
5. **EP config refactor:** `score_before_experts` moved from `MoE.Config` to token
   dispatcher; `make_experts_config()` now requires `top_k` param

**Changes required in ezpz:**

| File | Change | Commit |
|------|--------|--------|
| `trainer.py` | Migrate `maybe_enable_profiling`/`maybe_enable_memory_snapshot` to `Profiler.build()` | `e9022648` |
| `agpt/__init__.py` | Replace `inner_attention`/`mask_type` with `attn_backend`/`fuse_qkv`; add `_ezpz_get_attention_config()` XPU-aware wrapper; remove deepcopy overlay functions | `8f6d5cad` |
| `agpt/parallelize.py` | Add `FusedQKVLinear` import and detection; branching TP plan for fused vs unfused QKV | `8f6d5cad` |
| `moe/__init__.py` | Replace `inner_attention`/`mask_type` with `attn_backend`; update `make_moe_config` (remove `score_before_experts`); update `make_experts_config` (add `top_k`, `comm_backend`) | `9d3f3823` |
| `moe/parallelize.py` | Remove `DeepEPExpertParallel`/`ReordererSequenceParallel` imports; remove `hybridep_non_blocking_expert_capacity_factor` reference; EP now handled by token dispatcher | `9d3f3823` |

---

## 2026-04-15 (13th sync)

**Upstream commits:**

- `98ec7b4f` — [rl] Add batched RL training with varlen sequence packing (#2906)

**Files changed:** `experiments/rl/` only (actors, types, config_registry).

**Changes required in ezpz:** None. Clean merge.

---

## 2026-04-15 (12th sync)

**Upstream commits:**

- `0610235b` — Add bf16 optimizer state support via step pre-hook (#2732)

**Files changed:** `components/optimizer.py`, `docs/bf16_optimizer_states.md`, tests.

**Changes required in ezpz:** None. Clean merge.

---

## 2026-04-15 (11th sync)

**Upstream commits:**

- `5242bdf7` — Enable per-layer compile with or without MoE (#2741)
- `74485ea3` — Update pytorch nightly to cu13 (#2945)
- `42170d8a` — Revert FlexAttn max_autotune default to True (#2964)

**Breaking changes:**

- `distributed/compile.py` — consolidated `apply_compile_dense` and
  `apply_compile_sparse` into single `apply_compile`. Unconditionally sets
  `torch._dynamo.config.capture_scalar_outputs = True`.

**Changes required in ezpz:**

| File | Change | Commit |
|------|--------|--------|
| `agpt/parallelize.py` | Update import from `apply_compile_dense` to `apply_compile`; reset `capture_scalar_outputs = False` after apply for dense models (prevents unbacked symbol crash in compiled loss with TP + loss_parallel) | `d09d9708`, `e8cbb8ef` |

---

## 2026-04-14 (10th sync)

**Upstream commits:**

- `b35ca339` — [GraphTrainer][AutoDev] Extract remove-noop graph passes into dedicated module (#2952)
- `6d8c7e90` — [GraphTrainer] Update bitwise hash (#2962)

**Files changed:** `experiments/graph_trainer/` only (pass refactoring).

**Changes required in ezpz:** None. Clean merge.

---

## 2026-04-14 (9th sync)

**Upstream commits:**

- `6f7a6a79` — [CI] Fix torchvision::nms error in RL integration tests (#2959)
- `4f73d027` — [AutoDev] Restrict agent to only read actionable board items (#2953)
- `f5ecda7e` — [GraphTrainer] Enable regional_inductor for GraphTrainer (#2869)
- `b245fcaa` — [RL] Two small fixes in `inference_example.py` (#2944)
- `c630d30f` — [rl] Add torchcomms installation to setup instructions and remove xformers (#2943)

**Files changed:** `experiments/graph_trainer/`, `experiments/rl/`, CI workflows only.

No changes to `models/`, `distributed/`, or `trainer.py`.

**Changes required in ezpz:** None. Clean merge.

---

## 2026-04-13 (8th sync)

**Upstream commits:**

- `878041cb` — [Bugfix] Reenable llvm with triton pin update (#2873) <!-- codespell:ignore-line -->

**Files changed:** `models/common/attention.py` — removed `DISABLE_LLVM_OPT=1`
env var workaround. The upstream Triton pin (pytorch/pytorch#179586) fixed
the LLVM change that caused FlexAttention failures.

**Changes required in ezpz:** None. We don't use FlexAttention on XPU.

---

## 2026-04-13 (7th sync)

**Upstream commits:**

- `5c02bb54` — Enable 2 tier compilation with flex attention (#2929)
- `9f055b43` — [AutoDev] Default FlexAttn max_autotune to False (#2935)

**Files changed in `models/common/`:**

- `models/common/attention.py` — `FlexAttention.inductor_configs` changed:
  `wrap_inductor_compiled_regions` True (was False),
  `max_autotune` False (was True),
  `coordinate_descent_tuning` False (was True).

**Changes required in ezpz:** None. Our agpt configs use
`XPUScaledDotProductAttention` (not FlexAttention), and MoE configs use
`ScaledDotProductAttention` via the `_sdpa` variants. FlexAttention is only
used by `moe_debugmodel_flex_attn` and `moe_10b_2b` (non-SDPA), which are
not run on XPU.

---

## 2026-04-12 (6th sync)

**Upstream commits:**

- `dba7154b` — [GraphTrainer][AutoDev] Add remove_identity_slice_pass graph pass (#2920)
- `b9e8d1d1` — [GraphTrainer][AutoDev] Add remove_identity_view_pass graph pass (#2919)
- `314577bc` — [GraphTrainer][AutoDev] Add remove_detach_pass graph pass (#2917)

**Files changed:** `experiments/graph_trainer/passes.py`, `experiments/graph_trainer/tests/test_passes.py` only.

No changes to `models/`, `distributed/`, or `trainer.py`.

**Changes required in ezpz:** None. Clean merge.

---

## 2026-04-11 (5th sync)

**Upstream commits:**

- `12979c66` — [Qwen3VL] add qwen3 vl (#2409)
- `ab598d9c` — Skip parallelize_fn for seed checkpoint creation (#2928)
- `a5a60cc2` — [DeepEP] hide buffer init from sac (#2794)
- `dd33cde0` — TODO Debt Tracker (#2887)
- `1f9c677a` — [GraphTrainer] Nightly scout hands off (#2933)
- `98ff51e0` — [rl] Update monarch version in README (#2930)

**Files changed in `models/llama3/` and `models/deepseek_v3/`:**

- `models/llama3/parallelize.py` — removed `gradient_divide_factor` param from
  `apply_fsdp`, `disable_fsdp_gradient_division` moved to `distributed/fsdp.py`
- `models/deepseek_v3/parallelize.py` — same: removed `gradient_divide_factor`

**Breaking changes in `distributed/`:**

- `distributed/parallel_dims.py` — removed `fsdp_gradient_divide_factor` property
- `distributed/fsdp.py` — added `disable_fsdp_gradient_division` function
  (previously each model had its own copy)

**Changes required in ezpz:**

| File | Change | Status |
|------|--------|--------|
| `moe/parallelize.py` | Remove `gradient_divide_factor` from `apply_fsdp` call | Done |
| `moe/parallelize.py` | Replace `apply_compile_sparse` with per-block compile without `fullgraph=True` — MoE routing dynamic shapes cause `FailOnRecompileLimitHit` after upstream `00b7f569` removed `maybe_enable_amp` context boundary | Done |
| `moe/__init__.py` | Add `_10b_2b_sdpa()` variant using SDPA instead of FlexAttention — FlexAttention triggers `torch.autocast(dtype=torch.float32)` in MoE router which XPU doesn't support | Done |
| `moe/config_registry.py` | Add `moe_10b_2b_sdpa()` config entry | Done |
| `agpt/parallelize.py` | No changes needed — doesn't use `gradient_divide_factor` | OK |

**Note on MoE compile breakage:**

Upstream `00b7f569` (merged in 4th sync) removed `maybe_enable_amp` from
`trainer.py`'s `forward_backward_step`. This changed how `torch._dynamo`
traces MoE models — the context manager previously provided a graph boundary
that prevented recompilation. Without it, `fullgraph=True` in
`apply_compile_sparse` hits the recompilation limit on MoE routing's dynamic
expert dispatch. Fixed by applying compile per-block without `fullgraph=True`
in the ezpz MOE parallelize.

**Note on FlexAttention + MoE on XPU:**

The `10B_2B` config uses `FlexAttention` which calls
`torch.autocast(device_type="xpu", dtype=torch.float32)` in the MoE router
(`torchtitan/models/common/moe.py:231`). XPU autocast only supports bf16/fp16,
not fp32, causing a crash. Added `10B_2B_sdpa` variant that uses SDPA with
causal masking instead.

**Verified working (2026-04-11):**

| Config | compile | Status |
|--------|---------|--------|
| `moe_debugmodel` | enabled | OK |
| `moe_debugmodel` | disabled | OK |
| `moe_10b_2b` | enabled | FAIL — FlexAttention fp32 autocast on XPU |
| `moe_10b_2b_sdpa` | enabled | Testing |

---

## 2026-04-10 (4th sync)

**Upstream commits:**

- `00b7f569` — [FSDP2] replace amp and replicate with fully_shard (#2900)
- `7ef10559` — add CITATION.cff file (#2925)
- `55658e93` — [RL] Make reference model and KL penalty optional (#2750)
- `6930593b` — [GraphTrainer] Add bitwise deterministic test to H100 CI (#2921)
- `e24e465c` — [GraphTrainer] Changed tagging flex_attn via graph_pass (#2924)

**Files changed in `models/llama3/` and `models/deepseek_v3/`:**

- `models/llama3/parallelize.py` — removed `apply_replicate`, removed
  `fsdp_enabled` / `dp_replicate_enabled` branching, always use `fully_shard`
- `models/deepseek_v3/parallelize.py` — same refactor as llama3

**Breaking changes in `distributed/` and `trainer.py`:**

- `distributed/utils.py` — removed `maybe_enable_amp()` function. AMP is now
  handled internally by `fully_shard`'s `MixedPrecisionPolicy`.
- `distributed/parallel_dims.py` — `_mesh_exist` now returns `True` for `"fsdp"`
  always, so FSDP mesh exists even at degree 1.
- `trainer.py` — removed `self.maybe_enable_amp` from `forward_backward_step`,
  breaking MoE compile with `fullgraph=True` (see note above).

**Changes required in ezpz:**

| File | Change | Commit |
|------|--------|--------|
| `agpt/parallelize.py` | Removed `apply_replicate`, always use `fully_shard` | `5930b392` |
| `moe/parallelize.py` | Same: removed `apply_replicate` import and usage | `5930b392` |
| `trainer.py` | Removed `maybe_enable_amp` references (2 locations) | `6a2a09ec` |

---

## 2026-04-10 (3rd sync)

**Upstream commits:**

- `00b7f569` — [FSDP2] replace amp and replicate with fully_shard (#2900)
- `7ef10559` — add CITATION.cff file (#2925)
- `55658e93` — [RL] Make reference model and KL penalty optional (#2750)
- `6930593b` — [GraphTrainer] Add bitwise deterministic test to H100 CI (#2921)
- `e24e465c` — [GraphTrainer] Changed tagging flex_attn via graph_pass (#2924)

**Files changed in `models/llama3/` and `models/deepseek_v3/`:**

- `models/llama3/parallelize.py` — removed `apply_replicate`, removed
  `fsdp_enabled` / `dp_replicate_enabled` branching, always use `fully_shard`
- `models/deepseek_v3/parallelize.py` — same refactor as llama3

**Breaking changes in `distributed/`:**

- `distributed/utils.py` — removed `maybe_enable_amp()` function. AMP is now
  handled internally by `fully_shard`'s `MixedPrecisionPolicy`.

**Changes required in ezpz:**

| File | Change | Commit |
|------|--------|--------|
| `agpt/parallelize.py` | Removed `apply_replicate`, removed `fsdp_enabled` branching, always use `fully_shard` | `5930b392` |
| `moe/parallelize.py` | Same: removed `apply_replicate` import and usage, always use `fully_shard` | `5930b392` |
| `trainer.py` | Removed `maybe_enable_amp` references (2 locations) | `6a2a09ec` |

---

## 2026-04-10 (2nd sync)

**Upstream commits:**

- `5470cc7e` — GQAttention: Combine `q_norm` and `k_norm` into `qk_norm` (#2872)
- `4b45999f` — [GraphTrainer] Add FlexAttention bitwise deterministic tests (#2903)

**Files changed in `models/llama3/` and `models/deepseek_v3/`:**

- Neither `models/llama3/` nor `models/deepseek_v3/` changed.

**Breaking changes in `models/common/`:**

- `models/common/attention.py` — `GQAttention.Config` renamed `q_norm` and
  `k_norm` fields to a single `qk_norm` field. Removed the `__post_init__`
  validation that checked both were set together.

**Changes required in ezpz:**

| File | Change | Commit |
|------|--------|--------|
| `qwen3/__init__.py` | Updated 12 config entries from `q_norm=..., k_norm=...` to `qk_norm=...` | `4fe7aa48` |

`agpt` and `moe` were unaffected — `agpt` doesn't use QK norms, and `moe` has
its own `q_norm` field on a custom `DeepSeekAttention` config (not `GQAttention`).

---

## 2026-04-10 (1st sync)

**Upstream commits:**

- `8328876d` — [RL] Fix RL h100 workflow + `enable_gqa` flag in attention (#2891)
- `3c811045` — [GraphTrainer] Document aot_fx_trace compilation mode (#2912)

**Files changed in `models/llama3/` and `models/deepseek_v3/`:**

- Neither changed.

**Changes in `models/common/`:**

- `models/common/attention.py` — added `enable_gqa` parameter to
  `FlexAttention.forward()` and `ScaledDotProductAttention.forward()`.
  The ezpz branch already had the `VarlenAttention` forwarding for
  `enable_gqa`, which matched the upstream addition — no conflict.

**Changes required in ezpz:**

- None. Clean merge.
