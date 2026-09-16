from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.experiments.ezpz.optimizer.adopt import ADOPT
from torchtitan.experiments.ezpz.optimizer.mano import Mano
from torchtitan.experiments.ezpz.optimizer.muon import Muon, MuonClip, QKInputRecorder
from torchtitan.experiments.ezpz.optimizer.schedule_free import AdamWScheduleFree
from torchtitan.experiments.ezpz.optimizer.sophia import SophiaG
from torchtitan.experiments.ezpz.optimizer.spam import SPAM

__all__ = [
    "ADOPTOptimizersContainer",
    "ManoOptimizersContainer",
    "MuonClipOptimizersContainer",
    "MuonOptimizersContainer",
    "SPAMOptimizersContainer",
    "ScheduleFreeOptimizersContainer",
    "SophiaGOptimizersContainer",
    "TorchMuonOptimizersContainer",
    "default_adopt",
    "default_mano",
    "default_muon",
    "default_muon_clip",
    "default_schedule_free",
    "default_sophiag",
    "default_spam",
    "register_muonclip_qk_pairs",
]


# ---------------------------------------------------------------------------
# Container subclasses — thin wrappers that register the optimizer class name.
#
# Post upstream PR #3269 ("[optimizer] support mixed optimizers"), the
# Config base lives on OptimizersContainer.Config and just carries
# ``param_groups: list[ParamGroupConfig]`` + ``implementation``. Each
# container's only job is to extend ``_resolve_optimizer_cls`` so the
# pattern-based grouping inside OptimizersContainer.__init__ can dispatch
# the registered optimizer name to its concrete class. The old per-Config
# flat ``lr / beta1 / beta2 / eps / weight_decay`` fields are gone —
# users supply those through ``ParamGroupConfig.optimizer_kwargs`` (see
# the ``default_<name>(lr=..., **kwargs)`` factories below for the common
# single-group case, mirroring ``default_adamw``).
# ---------------------------------------------------------------------------


class ADOPTOptimizersContainer(OptimizersContainer):
    # Empty Config subclass so OptimizersContainer.Config.build() instantiates
    # THIS class (which has the ADOPT-registering _resolve_optimizer_cls)
    # instead of the base. Without this override, .build() builds the base
    # OptimizersContainer, whose _resolve_optimizer_cls only knows Adam/AdamW.
    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        pass

    @staticmethod
    def _resolve_optimizer_cls(name: str) -> type:
        if name == "ADOPT":
            return ADOPT
        return OptimizersContainer._resolve_optimizer_cls(name)


class SophiaGOptimizersContainer(OptimizersContainer):
    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        pass

    @staticmethod
    def _resolve_optimizer_cls(name: str) -> type:
        if name == "SophiaG":
            return SophiaG
        return OptimizersContainer._resolve_optimizer_cls(name)

    def update_hessian(self) -> None:
        """Delegate hessian update to each inner SophiaG optimizer."""
        for optimizer in self.optimizers:
            if isinstance(optimizer, SophiaG):
                optimizer.update_hessian()


class MuonOptimizersContainer(OptimizersContainer):
    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        pass

    @staticmethod
    def _resolve_optimizer_cls(name: str) -> type:
        if name == "Muon":
            return Muon
        return OptimizersContainer._resolve_optimizer_cls(name)


class MuonClipOptimizersContainer(MuonOptimizersContainer):
    @dataclass(kw_only=True, slots=True)
    class Config(MuonOptimizersContainer.Config):
        pass

    @staticmethod
    def _resolve_optimizer_cls(name: str) -> type:
        if name == "MuonClip":
            return MuonClip
        return MuonOptimizersContainer._resolve_optimizer_cls(name)


class ManoOptimizersContainer(OptimizersContainer):
    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        pass

    @staticmethod
    def _resolve_optimizer_cls(name: str) -> type:
        if name == "Mano":
            return Mano
        return OptimizersContainer._resolve_optimizer_cls(name)


class ScheduleFreeOptimizersContainer(OptimizersContainer):
    """Schedule-Free AdamW — no LR schedule needed.

    Requires .train() before training and .eval() before evaluation/checkpointing.
    Based on: https://github.com/facebookresearch/schedule_free
    """

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        pass

    @staticmethod
    def _resolve_optimizer_cls(name: str) -> type:
        if name == "AdamWScheduleFree":
            return AdamWScheduleFree
        return OptimizersContainer._resolve_optimizer_cls(name)

    def train_mode(self) -> None:
        """Switch all optimizers to train mode."""
        for optimizer in self.optimizers:
            optimizer.train()

    def eval_mode(self) -> None:
        """Switch all optimizers to eval mode."""
        for optimizer in self.optimizers:
            optimizer.eval()


class SPAMOptimizersContainer(OptimizersContainer):
    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        pass

    @staticmethod
    def _resolve_optimizer_cls(name: str) -> type:
        if name == "SPAM":
            return SPAM
        return OptimizersContainer._resolve_optimizer_cls(name)


# ---------------------------------------------------------------------------
# TorchMuon — composite (Muon for 2D, AdamW for the rest) per model part.
#
# Doesn't fit the pattern-based grouping model of the base container because
# the param split is shape-based, not name-based. Kept as a bespoke
# __init__. The Config still extends OptimizersContainer.Config so it can
# carry param_groups (used for the lr lookup) and ``implementation``.
# ---------------------------------------------------------------------------


class _CompositeOptimizer(torch.optim.Optimizer):
    """Wraps multiple optimizers into a single Optimizer interface.

    Needed because OptimizersContainer expects one optimizer per model part,
    but torch.optim.Muon only handles 2D params (need a separate AdamW for
    embeddings/head).
    """

    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        # Skip Optimizer.__init__ — it rejects empty param lists.
        # We manage param_groups and state via the inner optimizers.
        self._optimizers = optimizers
        self.defaults = {}
        self.state = {}
        self.param_groups = []
        for opt in optimizers:
            self.param_groups.extend(opt.param_groups)
            self.state.update(opt.state)

    def step(self, closure=None):
        loss = None
        for opt in self._optimizers:
            result = opt.step(closure)
            if result is not None:
                loss = result
        return loss

    def zero_grad(self, *args, **kwargs):
        for opt in self._optimizers:
            opt.zero_grad(*args, **kwargs)

    def state_dict(self):
        return {"optimizers": [opt.state_dict() for opt in self._optimizers]}

    def load_state_dict(self, state_dict):
        for opt, sd in zip(self._optimizers, state_dict["optimizers"]):
            opt.load_state_dict(sd)


class TorchMuonOptimizersContainer(OptimizersContainer):
    """Uses torch.optim.Muon (built-in, optimized) for 2D hidden layers
    and torch.optim.AdamW for embeddings/head/1D params.

    Much faster than the custom Muon implementation — benefits from
    PyTorch's fused kernels and Gram Newton-Schulz optimizations.

    Post PR #3269 we still hand-split params by shape (ndim==2 and
    max_shape<=10000 → Muon; rest → AdamW). The upstream pattern-based
    grouping doesn't help here because the split is shape-based, not
    name-based. We read lr / weight_decay / betas / eps from the first
    catch-all ParamGroupConfig in the Config; ``muon_*`` and
    ``adamw_lr_factor`` knobs are pulled out of that param group's
    optimizer_kwargs.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        pass

    def __init__(
        self, config: Config, *, model_parts: list[nn.Module]
    ) -> None:
        import torch.optim

        # Pull the single-group settings from the first ParamGroupConfig.
        # Callers should use ``default_torch_muon(lr=...)`` below; the
        # explicit-multi-group form isn't supported (shape-based split).
        if not config.param_groups:
            raise ValueError(
                "TorchMuonOptimizersContainer requires a non-empty param_groups "
                "(use default_torch_muon(lr=..., **kwargs))"
            )
        kw: dict[str, Any] = dict(config.param_groups[0].optimizer_kwargs)
        lr = kw.pop("lr")
        weight_decay = kw.pop("weight_decay", 0.0)
        adamw_lr_factor = kw.pop("adamw_lr_factor", 1.0)
        momentum = kw.pop("momentum", 0.95)
        nesterov = kw.pop("nesterov", True)
        ns_steps = kw.pop("ns_steps", 5)
        betas = kw.pop("betas", (0.9, 0.95))
        eps = kw.pop("eps", 1e-8)

        all_params: list[nn.Parameter] = []
        self.optimizers = []
        self.model_parts = model_parts
        # Track which model_part each optimizer belongs to, so state_dict /
        # load_state_dict can route correctly (upstream now requires this).
        self._model_part_indices: list[int] = []

        for part_idx, model in enumerate(model_parts):
            muon_params: list[nn.Parameter] = []
            adamw_params: list[nn.Parameter] = []
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                if p.ndim == 2 and max(p.shape) <= 10000:
                    muon_params.append(p)
                else:
                    adamw_params.append(p)

            inner_opts: list[torch.optim.Optimizer] = []
            if muon_params:
                inner_opts.append(
                    torch.optim.Muon(
                        muon_params,
                        lr=lr,
                        weight_decay=weight_decay,
                        momentum=momentum,
                        nesterov=nesterov,
                        ns_steps=ns_steps,
                    )
                )
            if adamw_params:
                inner_opts.append(
                    torch.optim.AdamW(
                        adamw_params,
                        lr=lr * adamw_lr_factor,
                        weight_decay=weight_decay,
                        betas=betas,
                        eps=eps,
                    )
                )

            if not inner_opts:
                # Empty model part (FSDP sharding) — use a no-op AdamW.
                inner_opts.append(
                    torch.optim.AdamW([{"params": []}], lr=lr)
                )

            self.optimizers.append(_CompositeOptimizer(inner_opts))
            self._model_part_indices.append(part_idx)
            all_params.extend(muon_params)
            all_params.extend(adamw_params)

        self._post_init(all_params)


# ---------------------------------------------------------------------------
# default_<name>(lr=..., **kwargs) factories — mirror upstream's
# default_adamw. Each returns an OptimizersContainer.Config with a single
# catch-all ParamGroupConfig naming the registered optimizer name. Use
# these for the simple single-optimizer-everywhere case; for mixed
# optimizer setups, build the Config explicitly with multiple
# ParamGroupConfig entries.
# ---------------------------------------------------------------------------


def default_adopt(
    lr: float = 1e-3, *, clip_lambda_power: float = 0.25, decouple: bool = False,
    **kwargs: Any,
) -> OptimizersContainer.Config:
    """One-group ADOPT config. ADOPT only supports foreach / for-loop, not fused."""
    clip_lambda = None
    if clip_lambda_power > 0:
        power = clip_lambda_power

        def clip_lambda(step: int, _power: float = power) -> float:  # noqa: F811
            return step**_power

    return ADOPTOptimizersContainer.Config(
        param_groups=[
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="ADOPT",
                optimizer_kwargs={
                    "lr": lr,
                    "betas": (0.9, 0.999),
                    "eps": 1e-6,
                    "weight_decay": 0.0,
                    "decouple": decouple,
                    "clip_lambda": clip_lambda,
                    **kwargs,
                },
            )
        ],
        implementation="foreach",
    )


def default_sophiag(lr: float = 3e-4, **kwargs: Any) -> OptimizersContainer.Config:
    """One-group SophiaG config."""
    return SophiaGOptimizersContainer.Config(
        param_groups=[
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="SophiaG",
                optimizer_kwargs={
                    "lr": lr,
                    "betas": (0.965, 0.99),
                    "rho": 0.04,
                    "weight_decay": 0.1,
                    **kwargs,
                },
            )
        ],
    )


def default_muon(lr: float = 2.4e-3, **kwargs: Any) -> OptimizersContainer.Config:
    """One-group Muon config."""
    return MuonOptimizersContainer.Config(
        param_groups=[
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="Muon",
                optimizer_kwargs={
                    "lr": lr,
                    "wd": 0.0,
                    "momentum": 0.95,
                    "nesterov": True,
                    "ns_steps": 5,
                    "adamw_betas": (0.95, 0.95),
                    "adamw_eps": 1e-8,
                    **kwargs,
                },
            )
        ],
    )


def default_muon_clip(
    lr: float = 2.4e-3,
    *,
    clip_t: float = 100.0,
    clip_alpha: float = 0.5,
    use_sqrt_d: bool = True,
    **kwargs: Any,
) -> OptimizersContainer.Config:
    """One-group MuonClip config."""
    return MuonClipOptimizersContainer.Config(
        param_groups=[
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="MuonClip",
                optimizer_kwargs={
                    "lr": lr,
                    "wd": 0.0,
                    "momentum": 0.95,
                    "nesterov": True,
                    "ns_steps": 5,
                    "adamw_betas": (0.95, 0.95),
                    "adamw_eps": 1e-8,
                    "qk_clip": True,
                    "clip_t": clip_t,
                    "alpha": clip_alpha,
                    "use_sqrt_d": use_sqrt_d,
                    **kwargs,
                },
            )
        ],
    )


def default_mano(lr: float = 3e-4, **kwargs: Any) -> OptimizersContainer.Config:
    """One-group Mano config."""
    return ManoOptimizersContainer.Config(
        param_groups=[
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="Mano",
                optimizer_kwargs={
                    "lr": lr,
                    "momentum": 0.95,
                    "weight_decay": 0.0,
                    "adamw_betas": (0.9, 0.95),
                    "adamw_eps": 1e-8,
                    **kwargs,
                },
            )
        ],
    )


def default_schedule_free(
    lr: float = 3e-4, *, warmup_steps: int = 200, r: float = 0.0,
    weight_lr_power: float = 2.0, **kwargs: Any,
) -> OptimizersContainer.Config:
    """One-group AdamWScheduleFree config."""
    return ScheduleFreeOptimizersContainer.Config(
        param_groups=[
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="AdamWScheduleFree",
                optimizer_kwargs={
                    "lr": lr,
                    "betas": (0.9, 0.95),
                    "eps": 1e-8,
                    "weight_decay": 0.1,
                    "warmup_steps": warmup_steps,
                    "r": r,
                    "weight_lr_power": weight_lr_power,
                    **kwargs,
                },
            )
        ],
    )


def default_spam(
    lr: float = 1.3e-3,
    *,
    spike_threshold: float = 2.0,
    delta_t: int = 100,
    ema_beta: float = 0.999,
    **kwargs: Any,
) -> OptimizersContainer.Config:
    """One-group SPAM config."""
    return SPAMOptimizersContainer.Config(
        param_groups=[
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="SPAM",
                optimizer_kwargs={
                    "lr": lr,
                    "betas": (0.9, 0.95),
                    "eps": 1e-8,
                    "weight_decay": 0.1,
                    "spike_threshold": spike_threshold,
                    "delta_t": delta_t,
                    "ema_beta": ema_beta,
                    **kwargs,
                },
            )
        ],
    )


def default_torch_muon(
    lr: float = 2.4e-3, *, adamw_lr_factor: float = 1.0,
    momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5,
    betas: tuple[float, float] = (0.9, 0.95), eps: float = 1e-8,
    weight_decay: float = 0.0, **kwargs: Any,
) -> OptimizersContainer.Config:
    """Single-config for TorchMuonOptimizersContainer.

    The container hand-splits params by shape (ndim==2 → Muon; rest →
    AdamW), so the pattern in this ParamGroupConfig is effectively
    ignored — it exists only to carry the kwargs into the container's
    bespoke __init__.
    """
    return TorchMuonOptimizersContainer.Config(
        param_groups=[
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="TorchMuon",  # ignored by bespoke __init__
                optimizer_kwargs={
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "adamw_lr_factor": adamw_lr_factor,
                    "momentum": momentum,
                    "nesterov": nesterov,
                    "ns_steps": ns_steps,
                    "betas": betas,
                    "eps": eps,
                    **kwargs,
                },
            )
        ],
    )


def register_muonclip_qk_pairs(
    optimizer: MuonClipOptimizersContainer,
    model_parts: list[nn.Module],
    *,
    q_attr: str = "wq",
    k_attr: str = "wk",
    d_head: int | None = None,
) -> list[QKInputRecorder]:
    """Register QK pairs for MuonClip post-step clipping.

    Call this as a ``post_optimizer_build_fn`` in model specs. Iterates over
    all attention modules in ``model_parts`` and registers Q/K weight pairs
    with the MuonClip optimizers.

    Returns the list of QKInputRecorder instances (keep them alive).
    """
    recorders: list[QKInputRecorder] = []
    for model in model_parts:
        for inner_opt in optimizer.optimizers:
            if not isinstance(inner_opt, MuonClip):
                continue
            for module in model.modules():
                if hasattr(module, q_attr) and hasattr(module, k_attr):
                    recorder = inner_opt.attach_to_attention(
                        module,
                        q_attr=q_attr,
                        k_attr=k_attr,
                        d_head=d_head,
                    )
                    recorders.append(recorder)
    return recorders
