# TorchTitan + 🍋 `ezpz`

Pre-training AuroraGPT (dense + MoE) on ALCF systems
(Aurora / Sunspot / Polaris) with PyTorch ≥ 2.10. This folder is
an opinionated experiment harness on top of upstream `torchtitan`
that adds:

- Fault-tolerant training (bad-node failover wrapper)
- Per-machine launch scripts + venv broadcast (`ezpz yeet-env`)
- Custom optimizers (Mano, SPAM, Muon, SophiaG, ADOPT)
- A `BlendCorpusDataLoader` for olmo-mix-1124 + arbitrary HF datasets
- Eval pipeline (DCP → HF safetensors → lm-eval)
- Detailed production / eval / scaling tracking under [`docs/`](docs/)

This file is the **landing page** — quickstart + an index of where
everything lives. For day-to-day work, jump straight to the more
specific pages linked below.

> [!NOTE]
> This is the [`saforem2/torchtitan@ezpz`](https://github.com/saforem2/torchtitan/tree/ezpz)
> fork. Upstream is [`pytorch/torchtitan`](https://github.com/pytorch/torchtitan)
> and we [resync against it regularly](docs/upstream-sync.md).

## Quickstart (2B dense training, 2 nodes)

Full setup details — module loads, venv install, large-scale yeet-env
broadcast — are in
[`docs/guides/running-with-newer-pytorch.md`](docs/guides/running-with-newer-pytorch.md).
Minimum viable path on Aurora:

```bash
# 1. Allocate two nodes
qsub -q prod -A AuroraGPT -l walltime=06:00:00,filesystems=flare:home -l select=2 -I

# 2. Clone + enter
git clone https://github.com/saforem2/torchtitan --branch ezpz
cd torchtitan

# 3. Setup environment (loads modules + ezpz helper functions)
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env

# 4. Launch 2B training
MODEL=2b bash torchtitan/experiments/ezpz/run_train.sh
```

For other models / machines / data files, see
[`run_train.sh`](run_train.sh) and the detailed setup guide above.

> [!TIP]
> To suppress the `UserWarning: Torchinductor` error seen when using
> `--compile.enable` on Aurora:
> ```bash
> export SYCL_DISABLE_FSYCL_SYCLHPP_WARNING=1
> ```

## Documentation index

The full prioritized landing page (with last-modified dates and a
sentence per entry) is at [`docs/README.md`](docs/README.md). For a
structural map of every file under `docs/`, see
[`docs/TREE.md`](docs/TREE.md). The headline pages by topic:

### Live status

| Page | What's there |
|------|--------------|
| [Production index](docs/production/README.md) | Snapshot of every active training trajectory — 2B / 20B / 80B at 256N / 512N / 1024N+ |
| [Eval index](docs/evals/README.md) | lm-eval scores per model with v1-vs-v2 plots (the bf16-master fix is decisively validated) |
| [Journal](docs/journal.md) | Day-by-day session log |

### Setup + running

| Page | What's there |
|------|--------------|
| [Running with newer PyTorch](docs/guides/running-with-newer-pytorch.md) | Module loads, venv install, tokenizer download, large-scale (>512 nodes) workflow |
| [`scripts/submit_agpt_{2b,20b,80b}_aurora_venv*.sh`](scripts/) | Current (torch 2.13 venv) PBS production submitters |
| [`submit/README.md`](submit/README.md) | Legacy torch-2.10-conda submit scripts (kept for v1 reproduction only) |

### Big findings + workarounds

| Page | What's there |
|------|--------------|
| [Known issues](docs/guides/known-issues.md) | Operational notes + workarounds for active bugs |
| [bf16 RMSNorm freeze](docs/guides/training-dtype-bf16-norm-freeze.md) | The headline v1 bug — why we restarted as v2 with `dtype=float32` |
| [Bad-node failover wrapper](docs/guides/bad-node-failover.md) | How the `failover_lib.sh` wrapper detects + swaps bad nodes mid-training |
| [TP loss-reporting bug](docs/guides/loss-reporting-tp-dist-reduce.md) | Why TP > 1 loss is off by `dp_world_size` and how `EzpzValidator` fixes it |
| [XPU attention issues](docs/guides/xpu-attention-issues.md) | No flash-attn, selective AC quirks, SDPA fallback |

### Per-feature subdirectories

| Folder | Contents |
|--------|----------|
| [`agpt/`](agpt/) | AuroraGPT dense model configs (2B / 20B / 80B) + parallelism |
| [`moe/`](moe/) | DeepSeek-style MoE model + custom `EzpzGroupedExperts` |
| [`moe_runs/`](moe_runs/) | JSON override configs + launcher for MoE experiments |
| [`optimizer/`](optimizer/) | Mano, SPAM, Muon, SophiaG, ADOPT |
| [`blendcorpus/`](blendcorpus/) | olmo-mix-1124 dataloader with train/validation splits |
| [`eval/`](eval/) | DCP → HF converter + lm-eval pipeline |
| [`rl/`](rl/) | GRPO experimental task registry |
| [`scripts/`](scripts/) | Production submission scripts + interactive launchers + benchmarks |
| [`competition/`](competition/) | Loss-speedrun harness for the optimizer competitions |
| [`tests/`](tests/) | CPU/XPU unit tests (run with `python3 -m unittest`) |

### Long-form documentation

| Folder | Contents |
|--------|----------|
| [`docs/production/`](docs/production/) | Live per-model / per-node-count training trackers |
| [`docs/evals/`](docs/evals/) | Per-model eval results + plots |
| [`docs/guides/`](docs/guides/) | Big-finding writeups, operational notes, how-tos |
| [`docs/experiments/`](docs/experiments/) | Per-machine smoke / benchmark / LR-finder reports |
| [`docs/scaling/`](docs/scaling/) | Per-model scaling-study results (TPS / MFU vs N) |
| [`docs/competitions/`](docs/competitions/) | Optimizer speedrun leaderboards |
| [`docs/meeting-notes/`](docs/meeting-notes/) | AuroraGPT sync agendas + action items |
| [`docs/summaries/`](docs/summaries/) | 2-week / monthly retrospectives |
| [`docs/upstream-issues/`](docs/upstream-issues/) | Repros + drafts for PRs we're filing back to `pytorch/torchtitan` |
| [`docs/configs/`](docs/configs/) | Model config docs (architecture, registered names) |
| [`docs/baselines/`](docs/baselines/) | Reference training curves + benchmarks |

## MoE training

The MoE harness is documented at [`moe_runs/README.md`](moe_runs/README.md)
(JSON override configs, launchers, per-machine smoke + perf + prod-sim
recipes for Aurora and Polaris). The MoE model + the
`EzpzGroupedExperts` compute-backend selector live in
[`moe/`](moe/).

## References

- 🍋 `ezpz`:
  - Documentation: [ezpz.cool](https://ezpz.cool)
  - GitHub: [saforem2/ezpz](https://github.com/saforem2/ezpz)
- Upstream torchtitan: [pytorch/torchtitan](https://github.com/pytorch/torchtitan)
- Datasets:
  - [olmo-mix-1124](https://huggingface.co/datasets/allenai/olmo-mix-1124) (production training)
  - [google/gemma-7b](https://huggingface.co/google/gemma-7b) (tokenizer, vocab_size=256128)
