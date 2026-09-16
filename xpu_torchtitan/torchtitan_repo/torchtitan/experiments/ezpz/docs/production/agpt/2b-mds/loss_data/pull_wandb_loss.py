#!/usr/bin/env python3
"""Pull train history (loss + grad_norm + tflops + TPS) and val loss
for the MDS AuroraGPT-2B SophiaG run.

The Megatron-DeepSpeed run was restarted ~113 times — every PBS job became
its own W&B run under aurora_gpt/AuroraGPT — so we filter to the production
config (nl=12, hs=2048, seq=8192, gbs=6144, optimizer=sophiag) and stitch
all per-run histories together by the `iteration` axis (deduping on
collision; later starts win).

Output:
    train_metrics.csv   columns: iteration, lm_loss, grad_norm, tflops, tps_per_gpu, run_id
    val_loss.csv        columns: iteration, val_loss, run_id

Usage:
    python3 pull_wandb_loss.py [--limit N]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import wandb

ENTITY = "aurora_gpt"
PROJECT = "AuroraGPT"

TRAIN_ITER_KEY = "loss/iteration"
TRAIN_LOSS_KEY = "loss/lm loss"
GRAD_NORM_KEY = "loss/grad_norm"
TFLOPS_KEY = "throughput/tflops"
TPS_KEY = "throughput/tokens_per_gpu_per_sec"

VAL_KEY = "val/lm loss"
VAL_ITER_KEY = "val/iteration"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap on number of runs (debug)."
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(__file__).parent,
        help="Directory to write CSVs.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api(timeout=60)
    runs = api.runs(
        f"{ENTITY}/{PROJECT}",
        filters={
            "config.args.optimizer": "sophiag",
            "config.args.hidden_size": 2048,
            "config.args.num_layers": 12,
            "config.args.seq_length": 8192,
            "config.args.global_batch_size": 6144,
        },
        per_page=100,
        order="+created_at",
    )

    # Per-iteration train metrics dict: it -> (lm_loss, grad_norm, tflops, tps, run_id)
    train_rows: dict[int, tuple[float | None, float | None, float | None, float | None, str]] = {}
    val_rows: dict[int, tuple[float, str]] = {}
    n = 0
    for r in runs:
        n += 1
        if args.limit and n > args.limit:
            break
        added_train = 0
        try:
            for row in r.scan_history(
                keys=[TRAIN_ITER_KEY, TRAIN_LOSS_KEY, GRAD_NORM_KEY, TFLOPS_KEY, TPS_KEY],
            ):
                it = row.get(TRAIN_ITER_KEY)
                if it is None:
                    continue
                it = int(it)
                lm = row.get(TRAIN_LOSS_KEY)
                gn = row.get(GRAD_NORM_KEY)
                tf = row.get(TFLOPS_KEY)
                tps = row.get(TPS_KEY)
                # Require at least one metric beyond the iteration anchor.
                if lm is None and gn is None and tf is None and tps is None:
                    continue
                train_rows[it] = (
                    float(lm) if lm is not None else None,
                    float(gn) if gn is not None else None,
                    float(tf) if tf is not None else None,
                    float(tps) if tps is not None else None,
                    r.id,
                )
                added_train += 1
        except Exception as e:
            print(f"  [{n}] {r.name} ({r.id}): train scan err {e}")
        added_val = 0
        try:
            for row in r.scan_history(keys=[VAL_KEY, VAL_ITER_KEY]):
                it = row.get(VAL_ITER_KEY)
                vl = row.get(VAL_KEY)
                if it is not None and vl is not None:
                    val_rows[int(it)] = (float(vl), r.id)
                    added_val += 1
        except Exception as e:
            print(f"  [{n}] {r.name} ({r.id}): val scan err {e}")
        print(
            f"  [{n}] {r.name} ({r.id}) {str(r.created_at)[:10]} "
            f"+{added_train} train, +{added_val} val "
            f"-> totals {len(train_rows)} train / {len(val_rows)} val"
        )

    # Write CSVs sorted by iteration
    train_csv = args.out_dir / "train_metrics.csv"
    with train_csv.open("w") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "lm_loss", "grad_norm", "tflops", "tps_per_gpu", "run_id"])
        for it in sorted(train_rows):
            lm, gn, tf, tps, rid = train_rows[it]
            w.writerow([
                it,
                "" if lm is None else lm,
                "" if gn is None else gn,
                "" if tf is None else tf,
                "" if tps is None else tps,
                rid,
            ])
    print(f"wrote {train_csv} ({len(train_rows)} rows)")

    val_csv = args.out_dir / "val_loss.csv"
    with val_csv.open("w") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "val_loss", "run_id"])
        for it in sorted(val_rows):
            vl, rid = val_rows[it]
            w.writerow([it, vl, rid])
    print(f"wrote {val_csv} ({len(val_rows)} rows)")


if __name__ == "__main__":
    main()
