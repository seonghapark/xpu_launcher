# Learning Rate Finder

## How It Works

The learning rate finder identifies the optimal learning rate for a given
model and optimizer by running a short sweep of exponentially increasing
learning rates while monitoring training loss.

### Algorithm

Based on [Smith 2015](https://arxiv.org/abs/1506.01186) and
[Gugger's implementation](https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html),
ported from [argonne-lcf/Megatron-DeepSpeed](https://github.com/argonne-lcf/Megatron-DeepSpeed).

1. Set `N = fraction * training.steps` (default: 10% of total steps)
2. Compute multiplicative factor: `mult = (max_lr / init_lr) ^ (1/N)`
3. For each step `i` in `1..N`:
   - Run one full training step (forward + backward + optimizer)
   - Record the loss
   - Compute EMA-smoothed loss: `avg = beta * avg + (1 - beta) * loss`
   - Apply bias correction: `smoothed = avg / (1 - beta^i)`
   - Update learning rate: `lr *= mult`
4. Analyze curve with derivative-based blow-up detection (`find_optimal_lr()`)
5. Save results (CSV, NPZ, plot) and exit

### The LR-vs-Loss Curve

The sweep produces a characteristic curve. Loss stays flat while the LR is
too small, descends through the useful range, bottoms out at a minimum, then
diverges (blows up) once the LR is too large -- so the curve ends HIGH on the
right, not low:

```
Loss
  |
  |"""""""\                                            .
  |        \                                         .     (4) blow-up:
  | flat    \                                      .          LR too large
  | (1)(2)   \                                   .            -> loss
  |           \                                .              diverges (NaN)
  |            \         (3) descent          .
  |             \      (optimal region)      .
  |              \                          .
  |               \________________________/
  |                      (5) minimum
  |                suggested LR ~= blow-up / 10
  +---|----------------------|--------------|--------> LR (log scale)
   init_lr               optimal          max_lr
```

```
(1) warmup   - LR held at init_lr for warmup_fraction steps; model settles
(2) flat     - LR still too small to matter; loss ~ constant
(3) descent  - useful range; the optimal LR is on this slope (toward the min)
(4) blow-up  - LR too large; loss diverges, NaN at the extreme (curve ends HIGH)
(5) minimum  - lowest loss; suggested LR ~= blow-up point / 10 (conservative)
```

### Selecting the Optimal LR

Two heuristics (both implemented):

1. **Blow-up point / 10**: `find_optimal_lr()` detects where the smoothed
   loss derivative crosses from negative to positive, then divides by 10.
   Conservative, safe for production.
2. **Steepest descent**: Pick the LR at the steepest part of the loss curve.
   More aggressive, potentially faster convergence.

### Knobs to Tune

| Parameter | Effect | When to change |
|-----------|--------|---------------|
| `init_lr` | Starting LR | Lower if the flat region is too short |
| `max_lr` | Ending LR | Raise if blow-up isn't reached |
| `fraction` | Sweep length | Increase for smoother curves (more steps) |
| `beta` | EMA smoothing | Lower (0.9) for noisier curves, higher (0.99) for smoother |
| `warmup_fraction` | Hold at init_lr before sweep | Set 0.05-0.1 to let model settle |
| `smooth_frac` | Derivative smoothing window | Increase (0.1+) for noisy/short curves |
| `training.steps` | Base for fraction | Set to 1000+ for 100+ finder steps |

### Important Notes

- The finder **bypasses the normal LR scheduler** — it directly sets
  `param_group["lr"]` on all optimizer param groups after each step
- The model is **not saved** after the sweep — it's meant for LR
  selection, not training
- Results are **distributed-aware** — loss is reduced across all ranks
  before recording
- The sweep runs the **full training step** including gradient clipping,
  so the curve reflects realistic training dynamics
- **The sweep is cumulative, not per-LR-independent.** Each step takes one
  real optimizer update at the current LR, *then* raises the LR -- the model
  keeps training throughout. So the absolute loss depth depends on the number
  of sweep steps: a 100-step sweep reaches a lower minimum than a 15-step
  sweep simply because it has done ~6x more updates by the time it hits the
  optimal-LR region, even at the same model/init/LR. **Only the LR at the
  minimum is comparable across sweeps of different length; the loss value is
  not.** (Concretely: the Apr-2026 2B finders ran 100 steps and bottom out
  near loss ~9-10, while the 2026-06-28 GBS-trend sweeps ran 15 steps and
  bottom out near ~11.3 -- both at the same min LR ~8.6e-3. The shorter sweep
  was chosen because high-GBS / 80B steps are expensive and the finder only
  needs to locate the LR, not converge.)

---

## Results

Results are split by model family, then by model size (each size has its own
page + `figures/`):

- **agpt (dense)** -- [index](agpt/README.md) (master LR table + cross-model
  findings), with per-size pages: [2B](agpt/2b/README.md) /
  [20B](agpt/20b/README.md) /
  [80B](agpt/80b/README.md) (incl. the GBS=6144 production-batch finding +
  LR-ceiling-vs-GBS trend).
- **moe (sparse)** -- [index](moe/README.md), with per-config pages:
  [debugmodel](moe/debugmodel/README.md) / [500M](moe/500m/README.md) /
  [2B](moe/2b/README.md) / [4B](moe/4b/README.md) / [7B](moe/7b/README.md).

### Headline recommended LRs (agpt, small-batch)

| Model | AdamW   | Muon    | SophiaG |
|-------|---------|---------|---------|
| 2B    | **1.3e-3**| **2.4e-3**| **3.1e-4**|
| 20B   | **4.0e-4**| **1.7e-4**| **1.8e-5**|
| 80B (small batch, GBS=192) | **1.1e-5** | N/A[1] | N/A[1] |

[1] 80B Muon/SophiaG broken at dim=9216 (bf16 overflow). **Important:** at the
80B *production* batch (GBS=6144) the AdamW number above does NOT hold -- its
usable LR collapses to ~7e-7 (a NaN cliff) and mano becomes the best-behaved
optimizer. See the
[agpt 80B page](agpt/80b/README.md).

### Cross-family key findings

1. **Optimizer sensitivity:** `AdamW (most tolerant) > Muon > SophiaG (most sensitive)`
2. **Model scaling:** Larger models need lower LRs. Muon/SophiaG scale more
   aggressively (~N^-0.5) than AdamW (~N^-0.25).
3. **Blow-up severity:** SophiaG diverges catastrophically (loss 7,000+) vs
   gradual blow-up for AdamW (loss ~60).
4. **Cross-hardware consistency:** Suggested LRs match within 2x across Intel
   XPU (Aurora, Sunspot) and NVIDIA A100 (Polaris).
5. **Batch dependence:** the optimal/ceiling LR shifts strongly with batch size
   -- a small-batch sweep cannot calibrate a large-batch production LR (80B).

---

## Comparison with Megatron-DeepSpeed

Our LR finder is ported from
[argonne-lcf/Megatron-DeepSpeed](https://github.com/saforem2/Megatron-DeepSpeed/blob/updates-and-model-card/ALCF/notes/large_batch_optimizers_settings.md).

### Methodology — same approach

Both implementations follow Smith 2015 / Gugger:
- Exponential LR sweep with power-law increase
- EMA-smoothed loss tracking
- "Blow-up point / 10" heuristic
- Same output format (CSV + NPZ + plots)

### What we adopted from Megatron-DeepSpeed

| Feature | Megatron-DeepSpeed | torchtitan-ezpz |
|---------|-------------------|-----------------|
| `find_all_minima_lrs()` | Derivative-based analysis with smoothing | Ported as `find_optimal_lr()` with closest-to-min selection |
| Weight init | `std = sqrt(2/(5*d))` | Adopted in agpt `_linear_init(dim)` |
| GAS support | `GRAD_ACC_STEPS=16` | Configurable via `--training.gradient_accumulation_steps` |
| Warmup | Not implemented | Added `warmup_fraction` config |

### Key differences

| | Megatron-DeepSpeed | torchtitan-ezpz |
|---|---|---|
| Default LR | 0.0002 (Muon) | 8e-4 (config default) |
| Sequence length | 4096 | 8192 (agpt), 4096 (moe) |
| Weight init | `sqrt(2/(5*d))` | `sqrt(2/(5*d))` (adopted) |
| Optimizers tested | AdamW, Muon, dShampoo, LAMB | AdamW, Muon, SophiaG |
| Schedulers | constant+cooldown, cosine, infinite | cosine, linear |
| Machines | Aurora, Sunspot | Aurora, Sunspot, Polaris |
| Models | AuroraGPT (Megatron arch) | AuroraGPT (torchtitan arch) |

### LR values — consistent

Megatron-DeepSpeed uses `LR=0.0002` as the default for Muon training,
which aligns with our 20B Muon finding of ~2-4e-5 (suggested, i.e.
conservative). The blow-up point itself is ~2e-4, matching their default.

Our data on optimizer sensitivity ordering (AdamW > Muon > SophiaG) and
the scaling law (larger models need lower LRs) extends beyond what's
documented in the Megatron-DeepSpeed notes.

---

## Reports

Per-model pages (each holds all dates / machines / batch sizes for that model):

- **agpt** -- [index](agpt/README.md) | [2B](agpt/2b/README.md) |
  [20B](agpt/20b/README.md) | [80B](agpt/80b/README.md)
- **moe** -- [index](moe/README.md) | [debugmodel](moe/debugmodel/README.md) |
  [500M](moe/500m/README.md) | [2B](moe/2b/README.md) | [4B](moe/4b/README.md) |
  [7B](moe/7b/README.md)

---

## Usage

### Running the LR finder

Add `--lr_finder.enable` to any training command:

```bash
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module ezpz.agpt --config agpt_2b \
  --training.steps 1000 \
  --checkpoint.no_enable \
  --lr_finder.enable
```

The finder runs for 10% of `training.steps` (e.g. 100 steps for 1000), sweeping
LR exponentially from `init_lr` to `max_lr`.

### Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--lr_finder.enable` | false | Run LR finder instead of training |
| `--lr_finder.init_lr` | 1e-6 | Starting learning rate |
| `--lr_finder.max_lr` | 1.0 | Maximum learning rate |
| `--lr_finder.fraction` | 0.1 | Fraction of training.steps to sweep |
| `--lr_finder.beta` | 0.98 | EMA smoothing factor |
| `--lr_finder.warmup_fraction` | 0.0 | Hold at init_lr before sweep |
| `--lr_finder.smooth_frac` | 0.05 | Derivative smoothing window |

### With different optimizers

```bash
# AdamW (default)
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module ezpz.agpt --config agpt_20b \
  --training.steps 1000 --checkpoint.no_enable --lr_finder.enable

# Muon
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module ezpz.agpt --config agpt_20b \
  --training.steps 1000 --checkpoint.no_enable --lr_finder.enable \
  --optimizer muon

# SophiaG
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module ezpz.agpt --config agpt_20b \
  --training.steps 1000 --checkpoint.no_enable --lr_finder.enable \
  --optimizer sophiag
```

### Output

Results are saved to `outputs/lr_finder/ezpz/<module>/<flavor>/<optimizer>/`:

- `lr_finder_data.csv` — two columns: learning_rate, loss
- `lr_finder_data.npz` — numpy arrays for programmatic analysis
- `lr_vs_loss.png` — log-scale plot with minimum and suggested LR marked

### Reproducing the experiments

```bash
# All 6 sweeps (2B+20B x 3 optimizers):
LRF_MODELS="2b 20b" LRF_OPTIMIZERS="adamw muon sophiag" \
    bash torchtitan/experiments/ezpz/scripts/run_lr_finder_sweep.sh
```

### Generating plots

```bash
# From repo root (figures are now in a shared per-family dir; prefix the
# filenames by machine, e.g. sunspot_2b.png, when committing):
python3 torchtitan/experiments/ezpz/utils/plot_lr_finder.py \
    --data-dir outputs/lr_finder/ezpz/ezpz.agpt \
    --output-dir torchtitan/experiments/ezpz/docs/experiments/lr-finder/agpt/figures
```

## References

- Smith, L.N. (2015). [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)
- Gugger, S. [How Do You Find A Good Learning Rate](https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html)
- [Megatron-DeepSpeed LR Finder](https://github.com/saforem2/Megatron-DeepSpeed/blob/updates-and-model-card/ALCF/notes/large_batch_optimizers_settings.md#learning-rate) (argonne-lcf)
- Yang et al. (2023). [Tensor Programs VI: Feature Learning in Infinite-Depth Neural Networks](https://arxiv.org/abs/2312.16903) — weight init `sqrt(2/(5*d))`
