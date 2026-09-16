"""Loss-trajectory baseline check for post-upstream-sync smoke tests.

Each upstream sync replay should produce a loss curve that matches the
prior sync's curve within data-shuffle noise. This script extracts the
loss trajectory from a PBS .o file and either saves it as a baseline or
diffs it against the saved baseline for the same config.

Usage:

    # After running smoke_2b_50steps and confirming results look reasonable:
    python3 -m torchtitan.experiments.ezpz.eval.loss_baseline save \\
        --log smoke_2b_v22.o12465527 \\
        --baseline torchtitan/experiments/ezpz/docs/baselines/agpt_2b_50.json \\
        --note "v22 — quantize-on-config + LocalMapInnerAttention removal"

    # On the next sync's smoke run:
    python3 -m torchtitan.experiments.ezpz.eval.loss_baseline check \\
        --log smoke_2b_v23.o12465999 \\
        --baseline torchtitan/experiments/ezpz/docs/baselines/agpt_2b_50.json

The check passes if:
    - The new run reached the same number of steps as the baseline
    - |new_final_loss - baseline_final_loss| <= tol  (default 0.10)
    - The mean of the last 10 steps is within tol of the baseline mean

Streaming-data shuffle adds ~0.05 noise per run; tol=0.10 catches gross
regressions (wrong gradient placement, broken reductions, NaN drift)
without false-positiving on normal noise.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

# step:  N  loss: F.FFFFF  grad_norm: ...
# The ANSI escapes around the numbers are stripped before regex.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_STEP_RE = re.compile(r"step:\s*(\d+)\s+loss:\s*([0-9.]+)\s+grad_norm:\s*([0-9.]+)")


@dataclass
class Trajectory:
    config: str
    steps: list[int]
    losses: list[float]
    grad_norms: list[float]

    @property
    def final_loss(self) -> float:
        return self.losses[-1]

    @property
    def tail_mean(self) -> float:
        tail = self.losses[-10:] if len(self.losses) >= 10 else self.losses
        return mean(tail)


def parse_log(path: Path) -> Trajectory:
    """Extract per-step (step, loss, grad_norm) from a PBS .o log."""
    steps, losses, grad_norms = [], [], []
    with path.open() as f:
        for line in f:
            clean = _ANSI_RE.sub("", line)
            m = _STEP_RE.search(clean)
            if not m:
                continue
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            grad_norms.append(float(m.group(3)))
    if not steps:
        raise SystemExit(f"no step:/loss: lines found in {path}")
    return Trajectory(
        config=path.stem,
        steps=steps,
        losses=losses,
        grad_norms=grad_norms,
    )


def save_baseline(traj: Trajectory, baseline_path: Path, note: str) -> None:
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": note,
        "source_log": traj.config,
        "num_steps": len(traj.steps),
        "final_loss": traj.final_loss,
        "tail10_mean": traj.tail_mean,
        "steps": traj.steps,
        "losses": traj.losses,
        "grad_norms": traj.grad_norms,
    }
    baseline_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"saved {baseline_path}")
    print(f"  source: {traj.config}")
    print(f"  steps: {len(traj.steps)}  final loss: {traj.final_loss:.5f}  "
          f"tail10 mean: {traj.tail_mean:.5f}")


def check_against(traj: Trajectory, baseline_path: Path, tol: float) -> int:
    payload = json.loads(baseline_path.read_text())

    expected_steps = payload["num_steps"]
    if len(traj.steps) != expected_steps:
        print(
            f"FAIL  step-count mismatch  "
            f"baseline={expected_steps} new={len(traj.steps)}"
        )
        return 1

    base_final = payload["final_loss"]
    base_tail = payload["tail10_mean"]
    new_final = traj.final_loss
    new_tail = traj.tail_mean

    delta_final = new_final - base_final
    delta_tail = new_tail - base_tail

    print(f"baseline note: {payload.get('note', '(none)')}")
    print(f"baseline source: {payload.get('source_log', '(unknown)')}")
    print()
    print(f"             baseline   new        delta   tol")
    print(f"  final     {base_final:>8.5f}  {new_final:>8.5f}  {delta_final:>+7.4f}  ±{tol}")
    print(f"  tail10    {base_tail:>8.5f}  {new_tail:>8.5f}  {delta_tail:>+7.4f}  ±{tol}")

    failed = abs(delta_final) > tol or abs(delta_tail) > tol
    if failed:
        print()
        print(f"FAIL  loss drift exceeds ±{tol}")
        return 1
    print()
    print(f"PASS  within ±{tol} of baseline")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    save = sub.add_parser("save", help="save a new baseline from a log")
    save.add_argument("--log", required=True, type=Path)
    save.add_argument("--baseline", required=True, type=Path)
    save.add_argument("--note", required=True, help="what changed since last baseline")

    check = sub.add_parser("check", help="diff a log against a saved baseline")
    check.add_argument("--log", required=True, type=Path)
    check.add_argument("--baseline", required=True, type=Path)
    check.add_argument("--tol", type=float, default=0.10,
                       help="abs loss tolerance for final + tail10 (default 0.10)")

    args = parser.parse_args()
    traj = parse_log(args.log)

    if args.cmd == "save":
        save_baseline(traj, args.baseline, args.note)
        return 0
    return check_against(traj, args.baseline, args.tol)


if __name__ == "__main__":
    sys.exit(main())
