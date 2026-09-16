# Upstream Sync — Loss Baselines

Each upstream sync replay should produce a loss curve that matches the
prior sync's curve within data-shuffle noise. This catches regressions
that pass at import-time and parallelize-time but silently break
gradients (wrong placement, broken reductions, NaN drift, etc.).

## Workflow per sync

```bash
# 1. Merge upstream and replay any breaking changes (see ../upstream-sync.md).

# 2. Run both smoke tests on 2 nodes.
qsub -l select=2 -l walltime=00:45:00 -N smoke_2b_vNN \
    -v CONFIG=smoke_2b_50steps \
    torchtitan/experiments/ezpz/competition/submit_run.sh

qsub -l select=2 -l walltime=00:45:00 -N smoke_moe_vNN \
    -v CONFIG=smoke_moe_500m_50steps \
    torchtitan/experiments/ezpz/scripts/submit_moe_smoke.sh

# 3. Diff against the saved baselines.
python3 -m torchtitan.experiments.ezpz.eval.loss_baseline check \
    --log smoke_2b_vNN.o<jobid> \
    --baseline torchtitan/experiments/ezpz/docs/baselines/agpt_2b_50.json

python3 -m torchtitan.experiments.ezpz.eval.loss_baseline check \
    --log smoke_moe_vNN.o<jobid> \
    --baseline torchtitan/experiments/ezpz/docs/baselines/moe_500m_50.json

# 4. If both PASS, refresh the baselines from this run so future drift
#    is measured against the latest code:
python3 -m torchtitan.experiments.ezpz.eval.loss_baseline save \
    --log smoke_2b_vNN.o<jobid> \
    --baseline torchtitan/experiments/ezpz/docs/baselines/agpt_2b_50.json \
    --note "vNN: <one-line summary of what changed>"

python3 -m torchtitan.experiments.ezpz.eval.loss_baseline save \
    --log smoke_moe_vNN.o<jobid> \
    --baseline torchtitan/experiments/ezpz/docs/baselines/moe_500m_50.json \
    --note "vNN: <one-line summary of what changed>"

# 5. Commit the updated baselines + the journal entry.
```

## What the check enforces

The script compares two scalars:

- **Final loss** (step 50) — should match within `±tol` (default 0.10)
- **Mean of last 10 steps** — same tolerance

`±0.10` is set so it catches gross regressions (wrong gradient
placement, broken reductions, NaN drift) without false-positiving on
the streaming-data shuffle noise that adds ~0.05 per run. Tighten with
`--tol` if needed for a specific check.

## When to update vs. investigate

- **PASS** → save the new run as the baseline (step 4 above). The new
  curve is now the reference for the next sync.
- **FAIL within ~0.15** → likely shuffle variance interacting with a
  small numerical change. Re-run once to confirm; if still failing,
  bisect.
- **FAIL > 0.20 or final NaN** → almost certainly a regression. Stop
  rolling out; bisect against the prior baseline.

## What is NOT a substitute

This is a smoke check, not bit-exactness validation. It will not catch:

- Subtle numerical changes that stay within ±0.1 over 50 steps but
  diverge over thousands.
- Throughput regressions (TPS / MFU). Watch the TPS column manually for
  significant deviations from the previous round's numbers in the
  journal.
- Anything specific to TP/EP/CP — the smoke configs run TP=1, EP=0,
  CP=1 to keep the verification fast.

For exact-numerics validation use `--debug.seed=42 --debug.deterministic`
on a fixed-seed pinned-data config (see `CLAUDE.md` "Validating
Numerics"). That's much slower on XPU and not part of the per-sync
workflow.

## Baselines

| File | Config | Hardware | Last refresh |
|---|---|---|---|
| `baselines/agpt_2b_50.json` | `smoke_2b_50steps` | Sunspot 2N | v22 (2026-04-28) |
| `baselines/moe_500m_50.json` | `smoke_moe_500m_50steps` | Sunspot 2N | v22 (2026-04-28) |
