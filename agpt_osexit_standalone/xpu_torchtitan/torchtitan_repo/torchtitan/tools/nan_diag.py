"""NaN/Inf origin finder for training on flaky hardware/kernels (AGPT_NAN_DIAG=1).

Every check is a 0-dim device bool appended to a per-step list; nothing syncs
until ``report()`` reads the list once per step. Only when something is
non-finite does it build a per-rank report of WHERE it first appeared:

  fwd       module outputs during the original forward
  fwd_param module parameters right after FSDP all-gather (pre-forward); a
            failure here with clean upstream outputs points at the all-gather
  recompute module outputs during the activation-checkpoint recompute
            (forward hooks firing inside backward)
  bwd       gradient w.r.t. each module output, in backward execution order
  grad      local parameter-gradient shards after FSDP reduce-scatter,
            before clipping (clip_grad_norm_ with a NaN norm makes every
            grad NaN, which would hide the origin)

For each phase the first failing tensor in execution order is reported, plus
whether it held NaN or only Inf (Inf-only suggests bf16/fp32 overflow).
"""

from __future__ import annotations

import os
import socket

import torch

_LEAF_TYPES = {"Linear", "RMSNorm", "Embedding", "ScaledDotProductAttention"}
_BLOCK_CHILDREN = {"attention", "feed_forward", "attention_norm", "ffn_norm"}


def enabled() -> bool:
    return os.environ.get("AGPT_NAN_DIAG", "0") == "1"


def _local(t):
    return t.to_local() if hasattr(t, "to_local") else t


def _first_tensor(out):
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        for o in out:
            if isinstance(o, torch.Tensor):
                return o
    return None


def _clean(name: str) -> str:
    return name.replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")


class NanDiag:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, torch.Tensor, torch.Tensor]] = []
        self.handles = []
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.host = socket.gethostname().split(".")[0]
        self.events = 0

    # -- recording ----------------------------------------------------------
    def _rec(self, phase: str, name: str, t: torch.Tensor) -> None:
        x = _local(t.detach())
        if x.numel() == 0 or not (x.is_floating_point() or x.is_complex()):
            return
        self.records.append((phase, name, torch.isfinite(x).all(), torch.isnan(x).any()))

    def install(self, model_parts) -> int:
        n = 0
        for part in model_parts:
            for raw_name, mod in part.named_modules():
                name = _clean(raw_name)
                leaf = type(mod).__name__ in _LEAF_TYPES
                child = name.split(".")[-1] in _BLOCK_CHILDREN
                if not (leaf or child):
                    continue
                self.handles.append(mod.register_forward_hook(self._make_fwd(name)))
                if child:
                    self.handles.append(mod.register_forward_pre_hook(self._make_pre(name, mod)))
                n += 1
        return n

    def _make_fwd(self, name: str):
        def hook(mod, args, out):
            t = _first_tensor(out)
            if t is None:
                return
            in_backward = torch._C._current_graph_task_id() != -1
            self._rec("recompute" if in_backward else "fwd", name, t)
            if not in_backward and t.requires_grad:
                t.register_hook(lambda g, _n=name: self._rec("bwd", _n, g) if g is not None else None)
        return hook

    def _make_pre(self, name: str, mod):
        def hook(m, args):
            if torch._C._current_graph_task_id() != -1:
                return
            for pn, p in mod.named_parameters(recurse=True):
                self._rec("fwd_param", f"{name}.{pn}", p)
        return hook

    def record_param_grads(self, model_parts) -> None:
        for part in model_parts:
            for name, p in part.named_parameters():
                if p.grad is not None:
                    self._rec("grad", _clean(name), p.grad)

    # -- reporting ----------------------------------------------------------
    def report(self, step: int, loss: torch.Tensor | None = None) -> str | None:
        recs, self.records = self.records, []
        if not recs:
            return None
        extra = [] if loss is None else [torch.isfinite(_local(loss.detach())).all()]
        flags = torch.stack([r[2] for r in recs] + [r[3] for r in recs] + extra).tolist()
        k = len(recs)
        finite, has_nan = flags[:k], flags[k : 2 * k]
        loss_finite = flags[2 * k] if extra else None
        if all(finite):
            return None
        self.events += 1
        parts = []
        for phase in ("fwd_param", "fwd", "recompute", "bwd", "grad"):
            idx = [i for i, r in enumerate(recs) if r[0] == phase]
            bad = [i for i in idx if not finite[i]]
            if not idx:
                continue
            if not bad:
                parts.append(f"{phase}=clean({len(idx)})")
                continue
            i = bad[0]
            kind = "nan" if has_nan[i] else "inf"
            parts.append(f"{phase}=FIRST:{recs[i][1]}[{kind}] bad={len(bad)}/{len(idx)}")
        return (
            f"NANDIAG step={step} rank={self.rank} host={self.host} "
            f"local_loss_finite={loss_finite} event={self.events} " + " | ".join(parts)
        )
