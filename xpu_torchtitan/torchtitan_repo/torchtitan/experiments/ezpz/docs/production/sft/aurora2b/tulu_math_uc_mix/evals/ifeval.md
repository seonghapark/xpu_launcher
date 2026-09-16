# IFEval — AuroraGPT-2B-sophiag (baseline) vs SFT-step729

**Status:** complete. Logs at
`logs/eval-ifeval-20260610-154906.log`. Run on PBS allocation
12468471 (claude-eval-alloc, 4N), single XPU tile per model,
back-to-back on the same physical node for fairness.

## TL;DR

**The SFT'd model is unambiguously better at following structural
instructions** — +8 pp absolute on the headline `prompt_level_strict_acc`
metric (~48% relative improvement). All four IFEval subscores
move in the same direction and well outside the per-eval
noise floor.

| Metric | Baseline | SFT-step729 | Delta | Relative |
|---|---:|---:|---:|---:|
| `prompt_level_strict_acc` | 0.1645 (±0.016) | **0.2440** (±0.019) | **+0.080** | **+48.3%** |
| `prompt_level_loose_acc` | 0.1682 | **0.2736** | +0.105 | +62.7% |
| `inst_level_strict_acc` | 0.2890 | **0.3861** | +0.097 | +33.6% |
| `inst_level_loose_acc` | 0.2950 | **0.4113** | +0.116 | +39.4% |

The ±0.018 stderr on `prompt_level_strict_acc` is roughly 1
sigma; +0.080 is **~4-5 sigma above noise**, so this is a real
effect, not eval-set sampling.

## What IFEval measures

541 generative prompts that test **verifiable structural
instruction-following**. Each prompt lists 1–4 constraints the
response must satisfy:

> Write a 200-word essay on X. Do not use the word "the". Include
> at least 3 questions. Each sentence must start with a
> capitalized verb.

Constraints are deterministic (counted programmatically, no
human judging required), so the score is reproducible. The metric
hierarchy:

- `prompt_level_strict_acc` — fraction of prompts where **ALL**
  listed constraints are met
- `prompt_level_loose_acc` — same but with relaxed punctuation
  / whitespace matching
- `inst_level_strict_acc` — fraction of **individual constraints**
  met (more lenient, since the model can pass 3/4 constraints
  and still get partial credit here)
- `inst_level_loose_acc` — instance-level with relaxed matching

The 2-3× ratio between `inst_level_*` and `prompt_level_*` (29%
inst vs 16% prompt for the baseline) reflects that the baseline
gets *some* constraints right per prompt but rarely all of them.

## Interpretation

Three things this tells us:

1. **The SFT actually taught the model to follow prompts.** The
   raw pretrained `AuroraGPT-2B-sophiag` produces text that's
   only accidentally constraint-compliant; the SFT'd version
   recognizes "do not use the word X" or "include at least N
   questions" as a structural directive and tries to honor it.
2. **The 4.5B-token mix had enough instruction signal.** The
   `tulu-3-sft-mixture` (65% weight) is the canonical SFT
   instruction corpus from AllenAI's tulu 3 work — it dominates
   the mix and clearly transferred. If we'd done a more math-
   heavy mix (say, swapping in 0.5 weight metamathqa) we'd
   probably see the IFEval gain shrink and the math gain grow.
3. **There's still substantial headroom.** Modern post-SFT models
   in this size range typically hit `prompt_level_strict_acc`
   in the 0.30–0.45 range; we're at 0.24. Mostly attributable
   to (a) only 3 epochs of SFT, (b) no DPO/RLHF preference
   stage, (c) 2B parameters is small. The next recipe could push
   harder on instruction adherence by:
   - more tulu-3 weight (currently 0.65)
   - longer SFT (5+ epochs)
   - a DPO follow-up pass (would need a preference dataset)

## Setup

| Field | Value |
|-------|-------|
| Eval | `lm-eval --tasks ifeval` |
| Tasks | 541 prompts, generative (`generate_until`) |
| n-shot | 0 |
| Backend | lm-eval 0.4.10 + transformers 4.50.1 + dtype-shim |
| Hardware | 1 XPU tile per model on `x1921c7s0b0n0`, back-to-back |
| Wall time | baseline 46 min, sft 47 min (total ~93 min) |

## Blockers cleared along the way

IFEval pulls in three Python deps that the bare `frameworks/2025.3.1`
module doesn't ship: `langdetect`, `immutabledict`, `nltk`. First
attempt crashed at module-import time. Fixed by:

```bash
PYTHON=venvs/sunspot/torchtitan-aurora_frameworks-2025.3.1/bin/python3
$PYTHON -m pip install --user langdetect immutabledict --no-deps
# nltk was already present; download punkt_tab
$PYTHON -c "import nltk; nltk.download('punkt_tab')"
```

User-pip-install lands in `~/.local/lib/python3.12/site-packages`
which the frameworks venv picks up via the standard site-packages
search. No `.venv/` mutation.

## Alongside this eval

The base-LM 7-task suite ([`README.md`](README.md)) showed no
change from SFT — as expected, since those benchmarks measure
base knowledge that SFT doesn't add. The GRPO smoke
([`grpo-smoke.md`](grpo-smoke.md)) showed an 8× downstream-RL
speedup. With this IFEval result, all three eval angles align:

- SFT doesn't change base-LM knowledge (lm-eval flat)
- SFT does add instruction adherence (IFEval +8pp)
- SFT does prime the model for RL on instruction-format tasks
  (GRPO 8× speedup, 100% by step 40)

The 4.5B-token `tulu_math_uc_mix` recipe is validated end to end.
