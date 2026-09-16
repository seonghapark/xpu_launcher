"""Verify ``torch.bincount`` is XPU-deterministic and equivalent to
``torch.histc`` for the MoE router/dispatcher usage pattern.

Context: ``torchtitan/models/common/moe.py:262`` and
``torchtitan/models/common/token_dispatcher.py:92`` both call

    num_tokens_per_expert = torch.histc(
        selected_experts_indices.view(-1),
        bins=num_experts,
        min=0,
        max=num_experts,
    )

which raises ``_histc_xpu does not have a deterministic
implementation`` when ``torch.use_deterministic_algorithms(True)``.

PR pytorch/torchtitan#3146's commit message says it replaces these
with ``torch.bincount``, but the merged diff actually only adds
``aten.topk.default`` to the SAC save list — the swap itself is
missing. Before filing the follow-up upstream, confirm that:

1. ``torch.bincount`` does NOT raise on XPU under
   ``use_deterministic_algorithms(True)``.
2. ``torch.bincount(x, minlength=N)`` and
   ``torch.histc(x.float(), bins=N, min=0, max=N).long()`` produce
   identical integer counts for ``x`` in ``[0, N)``.

Exit 0 on PASS, 1 on any mismatch.
"""

from __future__ import annotations

import sys

import torch


NUM_EXPERTS = 32
NUM_TOKENS = 8192
TOP_K = 2


def main() -> int:
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("XPU not available; this verifier only makes sense on XPU.")
        return 0

    device = torch.device("xpu:0")
    torch.xpu.set_device(device)
    torch.manual_seed(0)

    # Shape and dtype match what ``TokenChoiceTopKRouter`` produces:
    # int64 indices in [0, num_experts) from ``torch.topk`` over
    # routing scores.
    selected_experts_indices = torch.randint(
        0, NUM_EXPERTS, (NUM_TOKENS, TOP_K), device=device, dtype=torch.int64
    )

    # 1) baseline: histc (the current implementation)
    histc_result = torch.histc(
        selected_experts_indices.view(-1).float(),
        bins=NUM_EXPERTS,
        min=0,
        max=NUM_EXPERTS,
    ).long()
    print(f"histc result: shape={tuple(histc_result.shape)} sum={histc_result.sum().item()}")

    # 2) the proposed replacement: bincount
    bincount_result = torch.bincount(
        selected_experts_indices.view(-1),
        minlength=NUM_EXPERTS,
    )
    print(f"bincount result: shape={tuple(bincount_result.shape)} sum={bincount_result.sum().item()}")

    # Sanity: same total (== num_tokens * top_k).
    assert histc_result.sum().item() == NUM_TOKENS * TOP_K, "histc total wrong"
    assert bincount_result.sum().item() == NUM_TOKENS * TOP_K, "bincount total wrong"

    # Equivalence check (the contract PR #3146 claims).
    if not torch.equal(histc_result, bincount_result):
        diff = (histc_result - bincount_result).abs().max().item()
        print(
            f"FAIL: bincount and histc disagree. max |diff|={diff}"
        )
        print(f"histc[:8]={histc_result[:8].tolist()}")
        print(f"bincount[:8]={bincount_result[:8].tolist()}")
        return 1
    print("PASS: bincount and histc produce identical counts.")

    # Determinism check — the actual reason for the swap.
    torch.use_deterministic_algorithms(True)
    try:
        # bincount under deterministic algorithms
        bincount_under_det = torch.bincount(
            selected_experts_indices.view(-1),
            minlength=NUM_EXPERTS,
        )
        torch.xpu.synchronize()
        print(
            "PASS: torch.bincount does not raise under "
            "use_deterministic_algorithms(True) on XPU."
        )
    except RuntimeError as exc:
        print(
            f"FAIL: bincount raised under deterministic algorithms: {exc}"
        )
        return 1
    finally:
        torch.use_deterministic_algorithms(False)

    if not torch.equal(bincount_result, bincount_under_det):
        print(
            "FAIL: bincount results differ between non-deterministic and "
            "deterministic modes."
        )
        return 1
    print("PASS: bincount results are stable across determinism modes.")

    # Also verify histc DOES raise (sanity that we're actually testing
    # the same regime PR #3146 was trying to fix).
    torch.use_deterministic_algorithms(True)
    try:
        torch.histc(
            selected_experts_indices.view(-1).float(),
            bins=NUM_EXPERTS,
            min=0,
            max=NUM_EXPERTS,
        )
        torch.xpu.synchronize()
        print(
            "UNEXPECTED: histc did NOT raise under "
            "use_deterministic_algorithms(True) — XPU may have gained a "
            "deterministic _histc_xpu in this torch build. Re-test the "
            "full training path."
        )
    except RuntimeError as exc:
        if "_histc_xpu" in str(exc) and "deterministic" in str(exc):
            print(
                "CONFIRMED: histc raises the expected "
                "_histc_xpu non-deterministic error."
            )
        else:
            print(f"UNEXPECTED histc error: {exc}")
            return 1
    finally:
        torch.use_deterministic_algorithms(False)

    print("\nAll checks passed — bincount is a safe deterministic "
          "replacement for histc in this usage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
