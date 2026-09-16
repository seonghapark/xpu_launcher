"""Compare three approaches for ``num_tokens_per_expert`` under
``torch.use_deterministic_algorithms(True)`` on Intel XPU:

1. ``torch.histc`` — the current upstream implementation.
2. ``torch.bincount`` — what PR #3146's commit message promised but
   the merged diff didn't ship. PR author benchmarked it as ~20%
   slower than histc at the routing step on XPU.
3. ``torch.scatter`` (zeros + scatter_(... 1) then sum) — the
   alternative @acisseJZhong suggested on PR #3146 and the PR author
   benchmarked at +1.5% (noise) vs histc.

For each: (a) does it raise under
``torch.use_deterministic_algorithms(True)`` on XPU? (b) do all
three produce equal counts? (c) cheap microbenchmark.

This script is the basis for choosing between PR #3436 options:

- Unconditionally swap histc -> bincount (current PR #3436 — has the
  ~20% perf cost the PR #3146 author was concerned about).
- Conditional swap (use bincount only when
  ``torch.are_deterministic_algorithms_enabled()`` — zero cost
  default, only pays for determinism when the user opts in).
- Swap histc -> scatter (matches histc perf, may avoid the issue
  entirely if it's deterministic on XPU).
"""

from __future__ import annotations

import sys
import time

import torch


NUM_EXPERTS = 32
NUM_TOKENS = 8192
TOP_K = 2
WARMUP = 5
ITERS = 50


def histc_counts(idx, num_experts):
    return torch.histc(
        idx.view(-1), bins=num_experts, min=0, max=num_experts
    )


def bincount_counts(idx, num_experts):
    return torch.bincount(idx.view(-1), minlength=num_experts)


def scatter_counts(idx, num_experts):
    # zeros[expert_idx] += 1 across all (token, top_k) entries.
    counts = torch.zeros(
        num_experts, dtype=torch.int64, device=idx.device
    )
    counts.scatter_add_(
        0, idx.view(-1), torch.ones_like(idx.view(-1))
    )
    return counts


def time_kernel(fn, idx, num_experts):
    # warmup
    for _ in range(WARMUP):
        fn(idx, num_experts)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn(idx, num_experts)
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1e6  # microseconds


def main() -> int:
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("XPU not available.")
        return 0

    device = torch.device("xpu:0")
    torch.xpu.set_device(device)
    torch.manual_seed(0)

    idx = torch.randint(
        0, NUM_EXPERTS, (NUM_TOKENS, TOP_K), device=device, dtype=torch.int64
    )

    # 1) Equivalence — non-deterministic mode.
    h = histc_counts(idx, NUM_EXPERTS).long()
    b = bincount_counts(idx, NUM_EXPERTS)
    s = scatter_counts(idx, NUM_EXPERTS)

    print(f"histc:    {h[:6].tolist()} ... sum={h.sum().item()}")
    print(f"bincount: {b[:6].tolist()} ... sum={b.sum().item()}")
    print(f"scatter:  {s[:6].tolist()} ... sum={s.sum().item()}")

    if not (torch.equal(h, b) and torch.equal(h, s)):
        print("FAIL: not all three equal")
        return 1
    print("PASS: all three produce equal counts.")

    # 2) Determinism behavior.
    print("\n=== under torch.use_deterministic_algorithms(True) ===")
    torch.use_deterministic_algorithms(True)
    for name, fn in [
        ("histc", histc_counts),
        ("bincount", bincount_counts),
        ("scatter", scatter_counts),
    ]:
        try:
            r = fn(idx, NUM_EXPERTS)
            torch.xpu.synchronize()
            print(f"{name:9s}: OK (no raise)")
        except RuntimeError as e:
            msg = str(e).splitlines()[0]
            print(f"{name:9s}: RAISES — {msg}")
    torch.use_deterministic_algorithms(False)

    # 3) Microbenchmark (non-deterministic mode for apples-to-apples
    # vs the PR #3146 author's numbers).
    print("\n=== microbenchmark (per-kernel microseconds) ===")
    times = {}
    for name, fn in [
        ("histc", histc_counts),
        ("bincount", bincount_counts),
        ("scatter", scatter_counts),
    ]:
        try:
            times[name] = time_kernel(fn, idx, NUM_EXPERTS)
            print(f"{name:9s}: {times[name]:.2f} us")
        except RuntimeError as e:
            print(f"{name:9s}: SKIPPED ({e})")
    if "histc" in times:
        base = times["histc"]
        print(f"\nRelative to histc:")
        for n, t in times.items():
            pct = (t - base) / base * 100.0
            print(f"  {n:9s}: {pct:+.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
