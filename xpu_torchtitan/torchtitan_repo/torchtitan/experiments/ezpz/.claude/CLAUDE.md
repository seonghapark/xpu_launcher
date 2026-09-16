# ezpz Experiment Guide

## Golden Rules

1. **Never modify code outside `experiments/ezpz/`.** All changes stay here.
   If upstream core doesn't support something, work around it in ezpz.

2. **Every experiment must be tracked.** Any job submitted or run executed
   must have a clear purpose and be documented in an appropriate markdown
   file under `docs/`. Link reports from parent READMEs. End-of-session,
   update `docs/journal.md` with what happened.

3. **Upstream sync protocol:**
   - When pulling changes from upstream `main`, replay any changes to
     `llama3/` onto `ezpz/agpt/` and `deepseek_v3/` onto `ezpz/moe/`.
   - Always update `docs/upstream-sync.md` with what changed and what
     was replayed.

4. **Use `backup` instead of `rm`.** The `~/.local/bin/backup` command
   renames with a timestamp instead of deleting.

5. **Never `pip install` torch or its deps.** Can silently replace the
   Intel XPU build with CUDA. Use `--no-deps --no-cache --link-mode=copy`
   for any package that touches torch. Verify with
   `python3 -c "import torch; print(torch.xpu.is_available())"` after.

6. **Never run pre-commit hooks** as a verification step. User handles
   linting separately.

7. **Commit style:** Break changes into logically grouped commits with
   descriptive messages. Never batch unrelated changes.

8. **Never kill running jobs** without explicit user confirmation.

9. **Never discard local changes** — always stash-pull-pop, never
   reset/checkout to throw away work.

10. **Terse responses** — don't summarize what you just did at the end
    of every reply. The user can read the diff.

## Hardware & Platform

- **Machines:** Sunspot (Intel Max 1550 XPU), Aurora (Intel Max 1550 XPU), Polaris (A100)
- **Scheduler:** PBS (`qsub`, `qstat`, `qdel`)
- **XPU limitations:** No flash attention, no Triton backend, selective AC may not work.
  Always check XPU compatibility before suggesting optimization strategies.
- **torch.optim.Muon** is available since PyTorch 2.9 — prefer it over the custom
  Newton-Schulz implementation. The custom `optimizer/muon.py` is 35% slower.
- **HSDP (`dp_replicate × dp_shard > 1`) is untested for ezpz models** as of
  2026-05-04 — see "Recent Findings" below for the `aten.normal_.default`
  failure mode. Stick with pure FSDP (`dp_replicate=1`) until that's fixed.

## Environment Setup

- **torch 2.13 venv:** `.venv/` in repo root, copy to compute with `ezpz yeet-env`
- **torch 2.10 conda env:** `source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env`
  Loads `frameworks/2025.3.1` module + user venv overlay. Has lm_eval, vllm, transformers.
- **PBS scripts must NOT use `set -euo pipefail`** — venv activate has unbound vars
- **Compute nodes need proxy:** `export http_proxy=http://proxy.alcf.anl.gov:3128`
  (also `https_proxy`, `ftp_proxy`). Required before any `curl`, `pip`, or HF download.

## Aurora-Specific

### Queue Structure

| Queue | Nodes | Walltime | Notes |
|-------|-------|----------|-------|
| debug | 1–2 | 1h | Fast turnaround |
| debug-scaling | 1–256 | 1h | Best for small-scale tests |
| capacity | 1–16 | 168h | Long-running, good for evals |
| prod → small | 256–1024 | 12h | Production training |

### Job Submission

- **Current production path (native auto-retry):** Use
  `scripts/submit_agpt_{2b,20b,80b}_autoretry.sh`, which use
  `ezpz launch --auto-retry` (ezpz >= 0.17.1, PR #170) -- ezpz owns the
  split + scrape + swap + retry loop. The scripts yeet the venv to the
  whole (un-split) nodefile themselves (auto-retry does NOT broadcast)
  and compute `--nproc`/GBS from the active count (`NHOSTS_TRAIN*12`).
  Portable: Aurora PBS headers by default (AuroraGPT/prod/home:flare,
  data list `olmo-mix-1124`), Sunspot via qsub overrides
  (`-A datascience -q workq -l filesystems=flare:home`, data list
  `books`). Contract:
  `NHOSTS_TRAIN` + `select=active+spare`, `--spare-nodes auto`. 2B/20B
  default to `_real` cos_sin RoPE (compile-lowerable) with the validator
  on; the 80B script defaults to the confirmed-stable
  **TP=4/LBS=1/AdamW LR=1e-6** corner (supersedes the legacy TP=2
  default, which NaNs at production GBS) and **warns when
  `dp_degree=NGPUS/TP > 186`** (the grad-path NaN trigger; safe corner
  validated only to ~62N). See `docs/guides/bad-node-failover.md`.
- **Legacy (retained, not for new runs):** the bash `failover_lib.sh`
  wrapper scripts `scripts/submit_agpt_{2b,20b,80b}_aurora_venv_failover.sh`
  (torch 2.13 `.venv/`, LBS=2, fp32 master, bash preflight + spare-swap
  via `NHOSTS_TRAIN`/`FAILOVER_MAX_RETRIES`/...). Kept for reproducing
  earlier Aurora production runs and as the fixture-tested reference for
  the failure taxonomy; superseded by native auto-retry 2026-06-24. The
  non-failover `submit_agpt_{2b,20b}_aurora_venv.sh` were REMOVED
  2026-06-24 (recover from git history if ever needed).
- **Legacy torch-2.10 v1 scripts** live under `submit/{aurora,sunspot}/*.sh`
  with their own README. They produced every v1 (bf16-tainted)
  trajectory and are kept only for reproducing v1 numbers — nothing
  live calls into that directory anymore.
- **`scripts/train_agpt_{2b,20b,80b}_venv.sh`** are interactive
  launchers (run from a compute node with an existing allocation),
  not PBS submitters.
- **yeet-env tarball mode (`.venv.tar.gz` + `ezpz yeet --src ...`) is
  the default at scale.** Per-file rsync mode is the fallback. Tarball
  broadcast is sub-linear: 8N=70s → 256N=133s → 512N=175s → 1024N=255s
  → 4096N=750s. The old 1-2h "saturate flare for hours" warning was
  from per-file rsync mode and no longer applies — but still don't
  start multiple 512N+ jobs in the same minute.
- Always chain continuations: `qsub -W depend=afterany:<jobid> <script>`

### yeet-env Scaling

Copies `.venv/` (8.6GB) from flare to `/tmp` on every compute node.
**Tarball mode (`.venv.tar.gz` broadcast) is the default at scale**
and is sub-linear:

| Nodes | yeet (s) | Per-node (ms) |
| ----: | -------: | ------------: |
|     8 |     70   |         8,700 |
|    64 |     91   |         1,425 |
|   256 |    133   |           519 |
|   512 |    175   |           341 |
|  1024 |    255   |           249 |
|  4096 |    751   |           183 |

Two regimes: < 128 nodes is dominated by the one-time local extract
(~70-91s flat); ≥ 128 nodes the broadcast tree depth dominates, with
each 2× in nodes adding 1.5-1.8× wall-clock. Even at 4096N the
pre-launch overhead is under 13 minutes.

The "1–2+ hours per-file rsync DO NOT run concurrently" pattern still
applies if you fall back to rsync mode (no `.venv.tar.gz` present),
so always build the tarball first via `ezpz tar-env`. See
[`docs/guides/running-with-newer-pytorch.md`](../docs/guides/running-with-newer-pytorch.md).

### Optimizer Constraints at Scale

- **Muon broken at 80B:** bf16 overflow in Newton-Schulz at dim=9216 (NaN
  from step 7 regardless of LR; not relieved by batch size). Do NOT use Muon
  at 80B.
- **SophiaG broken at 80B ONLY at small batch.** It NaN'd at GBS=192, but at
  the production batch GBS=6144 it runs clean with a real LR-finder minimum
  at lr~2.5e-6 (larger batch smooths the Hessian grad*grad estimate below the
  bf16 overflow threshold). It is a viable 80B optimizer at production batch.
- **80B optimizer choice at GBS~6144 (measured 2026-06-27):** prefer **mano**
  (safest: clean broad U-min, suggested LR ~3e-6) or **sophiag** (lowest
  loss but narrower band, suggested LR ~1e-6). **AdamW is the worst of the
  three at this batch.**
- **80B AdamW at production GBS is on a NaN cliff.** At GBS=6144 the usable LR
  ceiling is only ~7e-7; LR=1e-6 sits PAST the last stable point (NaN onset
  1.36e-6), which is why GBS~6000 AdamW runs NaN nondeterministically. The
  old "use LR=1e-6" guidance is unsafe -- if staying on AdamW use ~5e-7, but
  prefer mano/sophiag. The small-batch "AdamW 1.1e-5" finder number does NOT
  transfer (batch-dependent: ~14x lower ceiling at production). See
  [`docs/experiments/lr-finder/agpt/README.md`](../docs/experiments/lr-finder/agpt/README.md#2026-06-27----80b-at-the-production-batch-gbs6144-sunspot).
- **torch.compile OOM at 512N:** 2B OOMs on GPU, 80B OOMs on CPU.
  Use `--compile.no-enable` for 512N jobs.
- **torch.compile time:** ~7–15 min at 256N depending on model size.
  4+ hours at TP=4 for 80B.

### Evaluation Pipeline

- **Checkpoint conversion:** `eval/convert_to_hf.py` converts DCP → HF safetensors.
  Use `--model_name experiments.ezpz.agpt --model_flavor 2b`.
  2B takes ~4 min, 20B takes ~20–30 min.
- **HF config:** Must copy `eval/configs/agpt_{2b,20b}_config.json` + tokenizer
  files from `assets/hf/gemma-7b/` into the HF checkpoint dir.
- **lm-eval:** Use bare `module load frameworks/2025.3.1` (NOT the user venv).
  The user venv has transformers 5.6.2 which breaks lm-eval's HF backend
  with `TypeError: LlamaForCausalLM.__init__() got unexpected kwarg 'dtype'`.
  Must also set `HF_HUB_ENABLE_HF_TRANSFER=0`.
- **Device:** `--device xpu` for lm-eval HF backend. 2B fits on 1 tile.
  20B (~40GB bf16) may need multi-tile.
- **Tokenizer:** google/gemma-7b, vocab_size=256128, bos=2, eos=1.

## Key Paths

```
torchtitan/experiments/ezpz/
├── agpt/                    # AuroraGPT model configs and registry
├── competition/             # Loss speedrun competition
├── datasets.py              # Generic HF dataset streaming
├── docs/                    # All documentation
│   ├── journal.md           # Day-by-day development log
│   ├── upstream-sync.md     # Upstream merge tracking
│   ├── competitions/        # Competition leaderboard + tracking
│   ├── configs/             # Model config docs
│   ├── evals/               # lm-eval results per model (v1 vs v2)
│   ├── experiments/         # Per-machine smoke / benchmark / LR-finder reports
│   ├── guides/              # Known issues, pytorch setup, XPU attention,
│   │                        # bf16 norm freeze, TP loss-reporting bug
│   ├── meeting-notes/       # AuroraGPT sync agendas + action items
│   ├── production/          # Live production training tracking (per-model, per-N)
│   ├── scaling/             # Per-model scaling study results
│   ├── summaries/           # Periodic 2-week / monthly retrospectives
│   └── upstream-issues/     # Upstream PR drafts + repro scripts
├── moe/                     # MoE model (DeepSeek-style) + parallelize
├── optimizer/               # Custom optimizers (Mano, SPAM, Muon, SophiaG, ADOPT)
├── rl/                      # GRPO reinforcement learning (task registry)
├── scripts/                 # Training, benchmark, LR-finder scripts
├── submit/aurora/           # Production submission scripts (torch 2.10 conda)
├── trainer.py               # FaultTolerantTrainer (overrides Trainer)
├── validator.py             # EzpzValidator (overrides Validator with TP loss fix)
├── blendcorpus/             # Blendcorpus dataloader (with serve_validation flag)
└── train.py                 # Main training entry point
```

## Running Training

```bash
# Interactive (from compute node with allocation)
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.agpt --config ezpz_agpt_2b

# PBS submission (from login node)
qsub -l select=2 -N my_job -v CONFIG=speedrun_2b_muon \
    torchtitan/experiments/ezpz/competition/submit_run.sh
```

## Available Optimizers

| CLI name | Container | Implementation |
|----------|-----------|----------------|
| `adamw` | `OptimizersContainer` | `torch.optim.AdamW` |
| `muon` | `MuonOptimizersContainer` | Custom Newton-Schulz (slow) |
| `torchmuon` | `TorchMuonOptimizersContainer` | `torch.optim.Muon` (fast) |
| `mano` | `ManoOptimizersContainer` | Manifold-normalized (arxiv 2601.23000) |
| `spam` | `SPAMOptimizersContainer` | Spike-aware Adam (arxiv 2501.06842) |
| `sophiag` | `SophiaGOptimizersContainer` | Second-order Hessian |
| `adopt` | `ADOPTOptimizersContainer` | Adaptive clipping |

## HF Dataset Streaming

Any HF dataset works as `--dataloader.dataset <path>`:
```bash
--dataloader.dataset HuggingFaceFW/fineweb-edu
--dataloader.dataset stanfordnlp/imdb
```

Auto-registered at runtime via `datasets.py`. No core changes needed.

**Caution:** Submitting many streaming jobs simultaneously hits HF rate limits
(1000 API requests / 5 min). Stagger submissions or cache locally.

## Competitions

**Tracking:** `docs/competitions/`
**W&B:** https://api.wandb.ai/links/aurora_gpt/hda3milo

| Competition | Winner | Loss |
|-------------|--------|------|
| 1000 steps, 2N, GBS=48 | Muon 3.557 / AdamW+QK-Norm 3.569 | 3.557 |
| 10B tokens, 8N, GBS=384 | AdamW 2.711 | 2.711 |
| 1000 steps, 2N, GAS=8, GBS=384 | AdamW+QK-Norm 3.205 | 3.205 |

**Key pattern:** Mano/Muon win short runs, AdamW wins in cosine decay phase.

## Recent Findings (2026-04-29 → 2026-05-04)

Big-ticket items from the last week. Each links to the canonical writeup
under `docs/`. Always check the relevant guide before suggesting work
that touches one of these areas.

- **bf16-master RMSNorm-freeze (2026-04-29).** All v1 production runs
  (2B / 20B / 80B) had `training.dtype = bfloat16`, keeping the master
  param copy in bf16. RMSNorm.weight starts at 1.0; the bf16 ULP at 1.0
  is ~7.8e-3 and per-step optimizer updates are ~1.6e-5 — every update
  rounds to zero. **All v1 RMSNorm.weights stayed at 1.0 for the whole
  run.** Fixed by flipping the default to `float32`. v2 production was
  restarted from scratch (path 2). End-to-end smoking gun:
  - **2B**: v2 ARC-Easy hit 0.429 at 100B tokens; v1 hovered at 0.272
    across 450B tokens (+19.8pp ARC-Easy / +15.4pp HellaSwag).
  - **20B**: v2 ARC-Easy `acc` lifts cleanly 0.271 → **0.444** across
    steps 100-800 (10-80B tokens); v1 256N flat at ~0.27 across all of
    0-63B tokens. HellaSwag `acc_norm` 0.254 → 0.284 (+3pp above v1).
  - **Direct weight verification**: v1 ckpts have RMSNorm.weight
    variance ≡ 0 (every value exactly 1.0); v2 step-5000 has
    mean(var)=1.2e-4, std=0.011, range [0.926, 1.102] across 25 norm
    layers. Final-norm channels scaled up uniformly to ~10% above init.
  See [`docs/guides/training-dtype-bf16-norm-freeze.md`](../docs/guides/training-dtype-bf16-norm-freeze.md).

- **TP > 1 loss reporting was off by `dp_world_size`** in the window
  2026-04-27 (upstream commit `1786292d`) through 2026-05-18 (upstream
  commit `d64eabcce`, [PR #3159](https://github.com/pytorch/torchtitan/pull/3159)).
  `_dist_reduce` short-circuited DTensor inputs with `full_tensor()`
  and skipped the requested mesh `all_reduce` when the meshes were
  orthogonal. Loss is a Replicated DTensor on the TP mesh; reductions
  are requested across `loss_mesh = batch × cp`. Reported `loss =
  true / dp_world_size` (gradients/optimizer steps unaffected). Fix:
  PR #3159 switched the DTensor branch to `x = x.to_local()` followed
  by the unconditional mesh `all_reduce`. Our local workarounds
  (`loss.full_tensor()` in `trainer.py` + `EzpzValidator(Validator)`
  in `validator.py`) were removed once the upstream fix had been in
  our `ezpz` branch via sync long enough to validate. **No current
  production runs use TP > 1**, so no live dashboard is wrong — but
  historical 80B v1 W&B traces from the affected window show
  `loss / 1536`. See
  [`docs/guides/loss-reporting-tp-dist-reduce.md`](../docs/guides/loss-reporting-tp-dist-reduce.md).

- **`compile + AC + TP=2` crashes on torch 2.13 for the entire
  agpt 80B family** with the
  `tensors_saved_with_vc_check`/`tensors_saved_for_backwards_with_vc_check_slice`
  AOT autograd assertion (`DeviceMesh` leaks into saved-for-backward
  tensors). Toy repro using legacy `parallelize_module` does NOT fire
  — needs the new `Module.parallelize` + `LocalMapConfig` path.
  **Bisect on 2026-05-05 with `agpt_{50b_wide, 70b_wide, 80b}` jobs
  12465952 (4N) and 12465962 (2N) showed all three crash on torch 2.13;
  the May 3 "depth-sensitive — works at 48 layers" claim was a
  torch-2.10-only artifact and is wrong.** Bug is **torch-2.13-sensitive**,
  fires on every config we've tried (smallest tested:
  `agpt_50b_wide` ~48B params, 2N, ~30s to crash).
  Workaround: `compile=OFF` for any 80B-family config on torch 2.13.
  Toy repro:
  [`docs/upstream-issues/repro_devicemesh_in_saved_tensors.py`](../docs/upstream-issues/repro_devicemesh_in_saved_tensors.py).

- **HSDP (`dp_replicate × dp_shard > 1`) hits an `aten.normal_.default`
  failure during init** (2026-05-04). Param init goes through
  `nn.init.trunc_normal_(param)` → `tensor.normal_(mean, std)` in-place,
  but DTensor sharding-prop has no strategy for in-place `normal_` on
  HSDP-style 2D meshes with mixed Replicate/Shard placements:
  `RuntimeError: aten.normal_.default: in-place operations that require
  placement changes are not supported`. Workaround until the
  materialize-then-init-then-redistribute fix lands: just use pure FSDP
  (`--data-parallel-shard-degree=<world_size>` and
  `--data-parallel-replicate-degree=1`).

- **Recurring `signal 9` Aurora NODE_FAIL pattern.** Four production
  jobs (8459818, 8460301, 8460302, 8463659) killed mid-run with
  `shepherd died from signal 9` on four different nodes. step-N
  ckpts saved cleanly so trajectories are resumable. Open question
  whether to file an ALCF support ticket — see
  [`docs/meeting-notes/agpt-sync.md`](../docs/meeting-notes/agpt-sync.md).

- **1024N init OOM/SIGSEGV (2026-05-04).** First-ever 1024N attempts
  on the v2 stack (8463182 2B, 8463183 20B) both crashed at startup
  in `set_determinism` distributed-init at 12,288 ranks
  (std::bad_alloc / SIGSEGV respectively). 256N/512N work fine. Bracket
  with 768N / 896N before resubmitting. See
  [`memory/project_1024n_init_crash.md`](.).

- **`EzpzValidator` at 80B TP=4: the "CCL deadlock" was a phantom.** The
  prior "validator deadlocks on a collective at 80B TP=4" framing is not
  backed by any log -- it conflated two unrelated NON-collective failures
  (a validator dataloader cold-cache `mmap`-race CRASH in job 12469584, and a
  training-side `_build_index_mappings` barrier stall in job 12469597) plus a
  genuine `loss_fn` tuple-unpack crash in `validator.py`. All now fixed
  (tuple-unpack; validator inherits the warm `data_cache_path`; prewarm builds
  the validation index). 2B/2N validator smoke is green (12469561); **80B TP=4
  validation CONFIRMED working 2026-06-28 (job 12469784, 4N/dp=12, warm cache):
  `validate()` completed at step 1, finite loss, no mmap/AttributeError/CCL
  hang.** (Confirmed at dp=12; a 62N pass would be belt-and-suspenders for
  production dp=186.) `VALIDATOR_ENABLE=0` remains the escape hatch. Full
  evidence: [`docs/guides/known-bugs/validator-tp4-at-80b.md`](../docs/guides/known-bugs/validator-tp4-at-80b.md).

## Common Pitfalls

- **Optimizer name casing:** Config objects use `"AdamW"` (capital), CLI uses
  `"adamw"` (lower). Use container Config classes in programmatic configs.
- **QK-Norm RMSNorm:** Needs `param_init=_NORM_INIT`, and lowercase alias
  (`2b_qknorm`) in `agpt_configs`.
- **Checkpoint conflicts:** Concurrent jobs must use different `checkpoint.folder`.
- **FSDP empty model parts:** Custom optimizer containers must handle model parts
  with zero parameters after sharding.
- **HF rate limits:** 24 ranks × multiple jobs = hundreds of API requests.
  Stagger PBS submissions by 5+ minutes.
- **HSDP with `dp_replicate > 1` will crash at init** — see "Recent
  Findings" above. Use pure FSDP.
- **Don't trust `loss:` from TP > 1 runs** without the `EzpzValidator`/
  `trainer.py` workaround in place — multiply by `dp_world_size` to
  recover the true value. See "Recent Findings" above.
- **`--dataloader.dataset=user/repo` (HF hub path) silently overrides
  `--dataloader.dataset-path`.** A warning is now emitted; if you want
  a local file dataset use `--dataloader.dataset blendcorpus
  --dataloader.dataset-path <txt-file>` instead.

## Production Training Status (Aurora)

Tracking in `docs/production/`. All v2 (post-bf16-fix) runs use the
torch 2.13 venv stack. New runs use the native auto-retry scripts
(`scripts/submit_agpt_*_autoretry.sh`); earlier Aurora v2 trajectories
ran under the legacy `*_aurora_venv_failover.sh` wrapper (see Job
Submission above). Default dtype is now `float32` (see Recent Findings).

### v2 — 2B 512N canonical chain (`8460301 → 8463626 → 8463627 → 8466847`)

- **Cumulative steps:** 5,073 (as of 2026-05-04)
- **Loss:** 2.97
- **Tokens consumed:** 510B (10.9% of 4.67T target)
- **Throughput:** ~2,700 TPS/GPU, ~10% MFU
- **Checkpoint dir:** `outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288/`
- **Status:** 8463626 walltime-finished cleanly (50 ckpts saved every
  100 steps); 8463627 (continuation) and 8466847 (held behind it)
  are both Q for a 512N slot.
- **Trajectory page:** [`docs/production/agpt/2b/n512/`](../docs/production/agpt/2b/n512/README.md)

### v2 — 20B 512N canonical chain (`8460302 → 8463628 → 8466848`)

- **Cumulative steps:** 863 (as of 2026-05-04)
- **Loss:** 3.46
- **Tokens consumed:** 87B (1.9% of 4.67T target)
- **Throughput:** ~358 TPS/GPU, ~17.8% MFU
- **Status:** 8463628 walltime-finished cleanly at step 863
  (step-100..step-800 ckpts saved); 8466848 (continuation, auto-released
  from hold) is Q for a 512N slot.
- **Eval (8 ckpts, steps 100-800):** ARC-Easy `acc` lifts cleanly
  **0.271 → 0.444** above v1's flat ~0.27 baseline; HellaSwag `acc_norm`
  breaks out **0.254 → 0.284** (+3pp above v1). ARC-C / Winogrande
  still in noise at this token count. The fp32-master fix is
  smoking-gun-validated at 20B.
- **Trajectory page:** [`docs/production/agpt/20b/n512/`](../docs/production/agpt/20b/n512/README.md)

### v2 — 20B 256N (`8463659`, NODE_FAIL after step 364)

- step 364, loss 4.61, 18.3B tokens. Killed by recurring `signal 9`
  Aurora NODE_FAIL — same pattern as 8459818 / 8460301.
- step-300 ckpt saved cleanly; resumable. No continuation chained
  (production is consolidated on the 512N chain; this trajectory was
  a per-token comparator scaling experiment).
- **Trajectory page:** [`docs/production/agpt/20b/n256/`](../docs/production/agpt/20b/n256/README.md)

### v2 — 1024N first attempts (`8463182` 2B, `8463183` 20B) — both crashed at startup

- **2B 1024N (8463182)**: `MemoryError: std::bad_alloc` in
  `torch.distributed.broadcast` during `set_determinism` init. Died
  in 211s.
- **20B 1024N (8463183)**: rank 4732 died from signal 11 (SIGSEGV)
  during the same init phase. Died in 211s.
- 12,288 ranks (1024N × 12 GPU/node) appears to hit an init-time
  memory/comm scaling issue that 256N and 512N don't see. Bracket
  with 768N / 896N before resubmitting.
- See [`memory/project_1024n_init_crash.md`](.).

### v2 — √2-LR fork (`8467141 → 8467142`, 2B 512N)

Experimental: tests whether scaling LR by √2 (3.22e-5 vs 2.28e-5) at
the doubled GBS=12,288 closes the per-token gap to 256N. Writes to
its own ckpt dir (`gbs12288-lr3.22e-5`), independent of the canonical
chain. Both jobs Q/H for 4+ days now.

### v2 — 80B

**Working path identified 2026-05-05** (job 12466025, 4N smoke, 20
steps, see `logs/agpt-80b-no-compile-t213-12466025/run.log`):

- `agpt_80b @ TP=2, AC=full, compile=OFF, AdamW LR=1e-6, fp32-master`
  on torch 2.13.
- Loss descended cleanly **12.98 → 10.46** (-2.52 nats) over 20 steps.
- MFU steady at **~17.8%** (matches what compile-on used to give v1
  on torch 2.10).
- Memory **88.94%** at peak (4N gives ~7 GiB per-tile headroom).
- Grad-norm bumped to ~34 around steps 15-16 then recovered to ~14
  by step 20 — production should add the 200-step linear warmup the
  2B/20B v2 configs use.

`compile=ON` is currently broken for the 80B family on **both** torch
versions (torch 2.10: step-1 hang regression since Apr 16-23 upstream
changes; torch 2.13: DeviceMesh-in-saved-tensors AOT autograd
assertion). `compile=OFF` is the only viable v2 path until either
upstream bug is fixed.

Not yet restarted as production — needs warmup added + a long-running
script. Job 12466025 was a smoke validation only.

### v1 (bf16-tainted, historical)

All v1 trajectories are kept under each per-trajectory page in
`docs/production/agpt/{2b,20b,80b}/n*/` for v1-vs-v2 comparison
purposes. Don't add tokens to v1 chains — they're frozen reference
points. Confirmed bf16 freeze: every v1 RMSNorm.weight is exactly
1.0 (variance ≡ 0); v2 step-5000 has mean(var)=1.2e-4, std=0.011,
range [0.926, 1.102] across 25 norm layers — i.e. norms are actually
training in v2. See `docs/guides/training-dtype-bf16-norm-freeze.md`.

## Scaling Study Results (Aurora)

### 2B Scaling (torch 2.10, compiled)
| Nodes | TPS/GPU | MFU |
|-------|---------|-----|
| 2 | 5,500 | 22% |
| 4 | 5,380 | 20% |
| 16 | 5,553 | 21% |
| 32 | 4,500 | 18% |
| 128 | 3,950 | 15% |
| 256 | 2,500 | 9.5% |

### 2B Scaling (torch 2.13, compiled, Sunspot)
| Nodes | TPS/GPU | MFU |
|-------|---------|-----|
| 2 | 7,142 | 27.6% |
| 4 | 7,068 | 27.3% |
| 16 | 6,995 | 27.0% |
| 64 | 6,702 | 25.9% |

## Known Bugs / Open Issues

Active issues with workarounds in place. For the full diagnosis +
empirical evidence, follow the doc link.

- **HSDP init crashes** with `aten.normal_.default: in-place operations
  that require placement changes are not supported`. Workaround: pure
  FSDP only.

- **`compile + AC + TP=2` AOT autograd crash on the agpt 80B family
  (torch 2.13).** `tensors_saved_with_vc_check` AssertionError —
  `DeviceMesh` leaks into saved-for-backward tensors. Bisect on
  2026-05-05 (jobs 12465952 + 12465962) showed the bug fires on every
  config in the family (smallest tested: `agpt_50b_wide` ~48B params,
  2N, ~30s to crash). The May 3 "depth-sensitive — works at 48 layers"
  claim was a torch-2.10-only artifact. **Bug is torch-version-sensitive:
  fires on torch 2.13, did not fire on torch 2.10.** Workaround:
  `compile=OFF` for 80B-family on torch 2.13, OR stay on torch 2.10
  for these configs. Toy repro:
  [`docs/upstream-issues/repro_devicemesh_in_saved_tensors.py`](../docs/upstream-issues/repro_devicemesh_in_saved_tensors.py).

- **80B TP=2 regression on torch 2.10:** Hangs at step 1 since upstream
  changes April 16-23. Works on torch 2.13. See `project_80b_bisect`
  memory.

- **MoE SIGABRT:** MOE models crash after upstream `00b7f569` refactor —
  `edp_mesh=None` when EP disabled. See `project_moe_sigabrt` memory.

- **TorchMuon integration:** `_CompositeOptimizer` wrapper needed for
  `OptimizersContainer`'s one-optimizer-per-model-part constraint.

- **IPEX import steals 60 MiB/rank:** Removing
  `import intel_extension_for_pytorch` fixed the 80B TP=2 OOM
  regression. Don't re-add it.

- **tyro Callable crash:** `trace_post_processor: Function.Config | None`
  in Profiler.Config. Fixed with `tyro.conf.Suppress`. PR
  pytorch/torchtitan#3038.

- **User venv transformers conflict:** The user venv at
  `venvs/aurora/torchtitan-ezpz-aurora_frameworks-2025.3.1/` has
  transformers 5.6.2 which breaks lm-eval's HF backend. Use bare
  `module load frameworks` for evals.

- **Stale checkpoint resume crash:** Trying to resume from a checkpoint
  built with a different parallelism config crashes silently. Fix:
  rename old checkpoint dir to `.bak-YYYYMMDD`.

- **Recurring Aurora `signal 9` NODE_FAIL** on long-walltime jobs (4
  jobs killed across 4 different nodes). Cause unclear; resume from
  most recent ckpt is the operational workaround. Possibly worth an
  ALCF support ticket.

- **1024N init OOM/SIGSEGV** at 12,288 ranks during `set_determinism`
  broadcast. First-ever attempts on v2 stack (8463182 / 8463183) both
  died in 211s. 256N/512N unaffected. See
  [`memory/project_1024n_init_crash.md`](.).

## User Preferences (operational)

These are *preferences* (style/formatting), not rules. Hard rules
moved up to "Golden Rules".

- **Always document experiments** — every run needs a markdown report
  under `docs/experiments/<module>/<machine>/<date>-<purpose>.md`,
  linked from a parent README.
- **Don't modify project-level `.gitignore`.**
- **Don't add `.ezpz-interactive-launch.sh`** to git — use
  `scripts/interactive-launch.sh`.
- **Append-only tables** — production training progress tables should
  append, not replace.
- **Use `scripts/submit_agpt_*_autoretry.sh`** (native
  `ezpz launch --auto-retry`) for current production (torch 2.13 venv,
  fp32-master). The bash `*_aurora_venv_failover.sh` wrapper is legacy
  but retained (reproducing earlier Aurora runs + failure-taxonomy
  reference). The non-failover `submit_agpt_{2b,20b}_aurora_venv.sh`
  were removed 2026-06-24 (recover from git history if needed). The
  legacy `submit/{aurora,sunspot}/*.sh` (torch 2.10 conda) are kept only
  for reproducing v1 numbers — see `submit/README.md`.
- **Date filenames as `YYYY-MM-DD`** for any per-day artifacts.
  Per-recurring-meeting docs use a stable filename with `## YYYY-MM-DD`
  sections inside (see `docs/meeting-notes/agpt-sync.md`).
- **Cross-link related docs.** Production READMEs link to eval READMEs
  and vice versa; the bf16-norm-freeze guide links to both training
  overlays and lm-eval figures.
