"""ezpz-free TorchTitan training entry with FT trainer, optimizer swap, XPU workarounds.

Standalone replacement for ``torchtitan.experiments.ezpz.train`` that does not
import the external ``ezpz`` package. It provides:

- XPU workarounds:
  * IPEX import on torch<2.11 (XPU operator overrides for TP collectives;
    without it TP=2+ hangs in the first forward pass)
  * xccl split-group monkeypatch (ProcessGroupXCCL lacks ``supportsSplitting``,
    which breaks nested DeviceMesh construction on XPU)
- ``FaultTolerantTrainer`` upgrade for any ``--module/--config`` (set
  ``FT_TRAINER=0`` to keep the registry's own trainer class)
- ``--optimizer <name> [--optimizer.field=value ...]`` swap (adamw, adopt,
  mano, muon, muonclip, schedulefree, sophiag, spam, torchmuon)

Usage: python titan_train.py --module llama3 --config llama3_debugmodel ...
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_TORCHTITAN_ROOT = Path(
    os.environ.get("TORCHTITAN_ROOT", Path(__file__).resolve().parent / "torchtitan_repo")
)
if str(_TORCHTITAN_ROOT) not in sys.path:
    sys.path.insert(0, str(_TORCHTITAN_ROOT))

import torch

# --- XPU workaround 1: IPEX operator overrides for TP collectives ---
if torch.__version__ < "2.11":
    try:
        import intel_extension_for_pytorch  # noqa: F401
    except Exception:
        pass

from torchtitan.components.optimizer import OptimizersContainer, default_adamw
from torchtitan.config import ConfigManager
from torchtitan.experiments.ezpz.optimizer import (
    SophiaGOptimizersContainer,
    default_adopt,
    default_mano,
    default_muon,
    default_muon_clip,
    default_schedule_free,
    default_sophiag,
    default_spam,
    default_torch_muon,
)
from torchtitan.experiments.ezpz.trainer import FaultTolerantTrainer
from torchtitan.experiments.ezpz.xccl_split_group_workaround import (
    maybe_install_xccl_split_group_workaround,
)
from torchtitan.tools.logging import init_logger, logger

_OPTIMIZER_FACTORIES: dict[str, Any] = {
    "adamw": default_adamw,
    "adam": default_adamw,
    "adopt": default_adopt,
    "mano": default_mano,
    "muon": default_muon,
    "muonclip": default_muon_clip,
    "schedulefree": default_schedule_free,
    "sophiag": default_sophiag,
    "spam": default_spam,
    "torchmuon": default_torch_muon,
}


def _coerce_override(raw: str) -> Any:
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _extract_optimizer_args(
    args: list[str],
) -> tuple[str | None, dict[str, str], list[str]]:
    """Extract ``--optimizer name`` and ``--optimizer.*`` overrides from args."""
    if "--optimizer" not in args:
        return None, {}, list(args)

    optimizer_name: str | None = None
    overrides: dict[str, str] = {}
    remaining: list[str] = []
    i = 0
    while i < len(args):
        token = args[i]
        if token == "--optimizer":
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                optimizer_name = args[i + 1].strip().lower()
                i += 2
                continue
            raise ValueError("--optimizer requires a name (e.g. --optimizer muon)")
        if token.startswith("--optimizer."):
            if "=" in token:
                key_part, value = token.split("=", 1)
                overrides[key_part.removeprefix("--optimizer.")] = value
                i += 1
            else:
                field_name = token.removeprefix("--optimizer.")
                if i + 1 < len(args) and not args[i + 1].startswith("--"):
                    overrides[field_name] = args[i + 1]
                    i += 2
                else:
                    overrides[field_name] = "true"
                    i += 1
            continue
        remaining.append(token)
        i += 1

    if optimizer_name is not None and optimizer_name not in _OPTIMIZER_FACTORIES:
        available = ", ".join(sorted(_OPTIMIZER_FACTORIES))
        raise ValueError(f"unknown optimizer {optimizer_name!r}; available: {available}")
    return optimizer_name, overrides, remaining


def _build_optimizer_config(
    name: str, overrides: dict[str, str]
) -> OptimizersContainer.Config:
    factory = _OPTIMIZER_FACTORIES[name]
    kwargs = {k.replace("-", "_"): _coerce_override(v) for k, v in overrides.items()}
    return factory(**kwargs)


def _upgrade_to_fault_tolerant(config: Any) -> Any:
    """Retype a Trainer.Config as FaultTolerantTrainer.Config (field-copying)."""
    if isinstance(config, FaultTolerantTrainer.Config):
        return config
    from dataclasses import fields

    ft_init_fields = {f.name for f in fields(FaultTolerantTrainer.Config) if f.init}
    try:
        kwargs = {
            f.name: getattr(config, f.name)
            for f in fields(type(config))
            if f.init and f.name in ft_init_fields
        }
        upgraded = FaultTolerantTrainer.Config(**kwargs)
        logger.info(
            "Upgraded %s -> FaultTolerantTrainer.Config", type(config).__name__
        )
        return upgraded
    except Exception as exc:
        logger.warning(
            "FaultTolerantTrainer upgrade failed (%s); using original config", exc
        )
        return config


def main(args: list[str] | None = None) -> None:
    init_logger()

    # --- XPU workaround 2: nested DeviceMesh split_group on xccl ---
    maybe_install_xccl_split_group_workaround()

    raw_args = sys.argv[1:] if args is None else args
    optimizer_name, optimizer_overrides, parsed_args = _extract_optimizer_args(raw_args)

    config_manager = ConfigManager()
    config: Any = config_manager.parse_args(parsed_args)

    if os.environ.get("FT_TRAINER", "1") == "1":
        config = _upgrade_to_fault_tolerant(config)

    if optimizer_name is not None:
        config.optimizer = _build_optimizer_config(optimizer_name, optimizer_overrides)
        logger.info(
            "Using optimizer: %s (%s)", optimizer_name, type(config.optimizer).__name__
        )

    trainer = None
    try:
        trainer = config.build()

        # SophiaG needs a hessian EMA update before each param update
        if isinstance(trainer.optimizers, SophiaGOptimizersContainer):
            trainer.optimizers.register_step_pre_hook(
                lambda *_a, **_k: trainer.optimizers.update_hessian()
            )

        if config.checkpoint.create_seed_checkpoint:
            assert int(os.environ.get("WORLD_SIZE", "1")) == 1, (
                "Must create seed checkpoint using a single device."
            )
            assert config.checkpoint.enable, (
                "Must enable checkpointing when creating a seed checkpoint."
            )
            trainer.checkpointer.save(curr_step=0, last_step=True)
            logger.info("Created seed checkpoint")
        else:
            trainer.train()
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        logger.info("Process group destroyed")


if __name__ == "__main__":
    main()
