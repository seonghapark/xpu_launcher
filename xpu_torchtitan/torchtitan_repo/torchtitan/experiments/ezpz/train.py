# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import json
import os
import sys

from pathlib import Path

from typing import Any

import ezpz
import ezpz.distributed
import ezpz.utils
import torch
import torch.distributed
from torch.distributed import get_rank, get_world_size, is_initialized

from torchtitan.components.optimizer import default_adamw, OptimizersContainer
from torchtitan.config import ConfigManager
from torchtitan.experiments.ezpz.logging import init_logger
from torchtitan.experiments.ezpz.optimizer import (
    ADOPTOptimizersContainer,
    ManoOptimizersContainer,
    MuonClipOptimizersContainer,
    MuonOptimizersContainer,
    ScheduleFreeOptimizersContainer,
    SPAMOptimizersContainer,
    SophiaGOptimizersContainer,
    TorchMuonOptimizersContainer,
    default_adopt,
    default_mano,
    default_muon,
    default_muon_clip,
    default_schedule_free,
    default_sophiag,
    default_spam,
    default_torch_muon,
)
from torchtitan.tools.logging import logger

import torchtitan.experiments.ezpz.datasets  # noqa: F401 — enable arbitrary HF datasets

DEFAULT_MODULE = "ezpz.agpt"
DEFAULT_CONFIG = "ezpz_agpt_2b"

fp = Path(__file__)
WBPROJ_NAME = f"torchtitan.{fp.parent.stem}.{fp.stem}"
os.environ.setdefault("WANDB_PROJECT", f"{WBPROJ_NAME}")

# IPEX provides XPU operator overrides needed for TP collectives on
# torch <=2.10. Without it, TP=2+ hangs during the first forward pass.
import torch as _torch

if _torch.__version__ < "2.11":
    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
    except Exception:
        pass


_LEGACY_KEY_REMAP = {
    "job.dump-folder": "dump-folder",
    "job.print-config": "debug.print-config",
    "job.print-args": "debug.print-config",
    "job.no-print-config": "debug.no-print-config",
    "job.save-config-file": "debug.save-config-file",
    "model.hf-assets-path": "hf-assets-path",
    "model.tokenizer-path": "hf-assets-path",
    "training.dataset": "dataloader.dataset",
    "training.dataset-path": "dataloader.dataset-path",
    "validation.enable": "validator.enable",
    "validation.no-enable": "validator.no-enable",
    "validation.freq": "validator.freq",
    "validation.steps": "validator.steps",
    "validation.dataset": "validator.dataloader.dataset",
    "validation.dataset-path": "validator.dataloader.dataset-path",
}

_FLAVOR_TO_CONFIG = {
    "debug": "ezpz_agpt_debugmodel",
    "debugmodel": "ezpz_agpt_debugmodel",
    "2b": "ezpz_agpt_2b",
    "7b": "ezpz_agpt_7b",
    "8b": "ezpz_agpt_8b",
    "auroragpt-2b": "ezpz_agpt_2b",
    "auroragpt2b": "ezpz_agpt_2b",
    "auroragpt-7b": "ezpz_agpt_7b",
    "auroragpt7b": "ezpz_agpt_7b",
    "llama3-8b": "ezpz_agpt_8b",
}

# PR #3269 ("[optimizer] support mixed optimizers") removed the flat
# `lr` / `beta1` / etc. fields from OptimizersContainer.Config and
# moved them into ParamGroupConfig.optimizer_kwargs. The
# ``--optimizer name --optimizer.lr=...`` CLI swap path now goes
# through one of these default_<name>(lr=..., **kwargs) factories,
# each of which returns an OptimizersContainer.Config with a single
# catch-all ParamGroupConfig naming the right optimizer.
_OPTIMIZER_FACTORIES: dict[str, Any] = {
    "adamw": default_adamw,
    "adam": default_adamw,  # base only registers Adam + AdamW
    "adopt": default_adopt,
    "mano": default_mano,
    "muon": default_muon,
    "muonclip": default_muon_clip,
    "schedulefree": default_schedule_free,
    "sophiag": default_sophiag,
    "spam": default_spam,
    "torchmuon": default_torch_muon,
}


def _update_env() -> dict:
    now = datetime.datetime.now()
    dstr = now.strftime("%Y-%m-%d-%H%M%S")
    env_dict: dict[str, Any] = {
        f"env.{k}": v
        for k, v in dict(os.environ).items()
        if not k.startswith("_") and "API" not in k and "LS_" not in k
    }
    env_dict |= {
        "created_at": dstr,
        "day": ezpz.utils.get_timestamp("%d"),
        "DIST_INFO": ezpz.distributed.get_dist_info(),
        "ezpz_file": ezpz.__file__,
        "ezpz_version": getattr(ezpz, "__version__", "0.0"),
        "hostname": ezpz.distributed.get_hostname(),
        "month": ezpz.utils.get_timestamp("%m"),
        "machine": ezpz.distributed.get_machine(),
        "pytorch_backend": str(ezpz.distributed.get_torch_backend()).lower(),
        "project": WBPROJ_NAME,
        "torch_version": torch.__version__,
        "torch_file": torch.__file__,
        "world_size": str(ezpz.distributed.get_world_size()),
        "year": ezpz.utils.get_timestamp("%Y"),
        "working_directory": os.getcwd(),
    }
    _ = env_dict.pop("LS_COLORS", None)
    _ = env_dict.pop("PS1", None)
    logger.info(f"Running on {ezpz.distributed.get_machine()=}")
    # logger.info(f"environment={json.dumps(env_dict, indent=4, sort_keys=True)}")

    return env_dict


def _has_flag(args: list[str], name: str) -> bool:
    key = f"--{name}"
    return any(arg == key or arg.startswith(f"{key}=") for arg in args)


def _inject_default_module_and_config(args: list[str]) -> list[str]:
    merged = list(args)
    if not _has_flag(merged, "module"):
        merged = ["--module", DEFAULT_MODULE, *merged]
    if not _has_flag(merged, "config"):
        merged = ["--config", DEFAULT_CONFIG, *merged]
    return merged


def _extract_optimizer_args(
    args: list[str],
) -> tuple[str | None, dict[str, str], list[str]]:
    """Extract ``--optimizer name`` and ``--optimizer.*`` overrides from args.

    Only activates when ``--optimizer`` (bare) is present. If absent,
    all args pass through to tyro unchanged.

    Returns:
        (optimizer_name, overrides_dict, remaining_args)
    """
    # Quick check: is --optimizer present as a standalone flag?
    has_bare_optimizer = False
    for i, arg in enumerate(args):
        if arg == "--optimizer":
            has_bare_optimizer = True
            break
    if not has_bare_optimizer:
        return None, {}, list(args)

    optimizer_name: str | None = None
    overrides: dict[str, str] = {}
    remaining: list[str] = []
    i = 0
    while i < len(args):
        token = args[i]

        # --optimizer <name> (bare, no dot)
        if token == "--optimizer":
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                optimizer_name = args[i + 1].strip().lower()
                i += 2
                continue
            else:
                raise ValueError("--optimizer requires a name (e.g. --optimizer muon)")

        # --optimizer.field value  or  --optimizer.field=value
        if token.startswith("--optimizer."):
            if "=" in token:
                key_part, value = token.split("=", 1)
                field_name = key_part.removeprefix("--optimizer.")
                overrides[field_name] = value
                i += 1
            else:
                field_name = token.removeprefix("--optimizer.")
                if i + 1 < len(args) and not args[i + 1].startswith("--"):
                    overrides[field_name] = args[i + 1]
                    i += 2
                else:
                    # Boolean flag with no value — treat as "true"
                    overrides[field_name] = "true"
                    i += 1
            continue

        remaining.append(token)
        i += 1

    if optimizer_name is None:
        raise ValueError("--optimizer flag found but no name provided")

    if optimizer_name not in _OPTIMIZER_FACTORIES:
        available = ", ".join(sorted(_OPTIMIZER_FACTORIES.keys()))
        raise ValueError(
            f"Unknown optimizer '{optimizer_name}'. Available: {available}"
        )

    return optimizer_name, overrides, remaining


def _coerce_override(raw: str) -> Any:
    """Coerce a string CLI override to bool / int / float / str.

    The old PR #3269 code walked dataclass fields for type info; the new
    factories accept ``**kwargs`` so types are inferred here from the
    raw string. Common cases:
      - ``"true"`` / ``"false"`` → bool
      - ``"42"`` → int
      - ``"2.4e-3"`` → float
      - everything else → raw string
    """
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


def _build_optimizer_config(
    name: str,
    base: OptimizersContainer.Config,
    overrides: dict[str, str],
) -> OptimizersContainer.Config:
    """Build optimizer Config from name + CLI overrides via the default_<name> factory.

    Post PR #3269, OptimizersContainer.Config no longer has flat
    ``lr`` / ``beta1`` / etc. fields — instead it holds
    ``param_groups: list[ParamGroupConfig]``. The ``--optimizer name``
    CLI flag now dispatches to a ``default_<name>(lr=..., **kwargs)``
    factory that builds a single-ParamGroupConfig setup naming the
    requested optimizer. ``base`` (the value already in the registry)
    is ignored — the user asked to switch optimizers, so we build a
    fresh setup keyed off the registered ``default_<name>`` defaults
    plus their ``--optimizer.*`` overrides.
    """
    factory = _OPTIMIZER_FACTORIES[name]
    kwargs: dict[str, Any] = {}
    for raw_key, raw_value in overrides.items():
        kwargs[raw_key.replace("-", "_")] = _coerce_override(raw_value)
    return factory(**kwargs)


def _canonicalize_option(option: str) -> str:
    return option.removeprefix("--").replace("_", "-")


def _config_name_from_flavor(flavor: str) -> str:
    normalized = flavor.strip().lower()
    return _FLAVOR_TO_CONFIG.get(normalized, f"ezpz_agpt_{normalized}")


def _translate_legacy_args(args: list[str]) -> list[str]:
    translated: list[str] = []
    legacy_tokenizer_backend: str | None = None
    i = 0

    while i < len(args):
        token = args[i]
        if not token.startswith("--"):
            translated.append(token)
            i += 1
            continue

        inline_value = "=" in token
        if inline_value:
            option, value = token.split("=", 1)
            consume_next = False
        else:
            option = token
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                value = args[i + 1]
                consume_next = True
            else:
                value = None
                consume_next = False

        key = _canonicalize_option(option)

        if key == "job.config-file":
            raise ValueError(
                "`--job.config-file` is no longer supported for ezpz. "
                "Use `--module ezpz.agpt --config ezpz_agpt_debugmodel` and CLI overrides instead."
            )

        if key in {"experimental.custom-args-module", "experimental.custom-import"}:
            logger.warning("Ignoring deprecated --experimental.* flag for ezpz.")
            i += 2 if consume_next else 1
            continue

        if key == "model.name":
            if value is not None:
                module_name = value
                if value.strip().lower() == "blendcorpus":
                    module_name = DEFAULT_MODULE
                translated.extend(["--module", module_name])
            i += 2 if consume_next else 1
            continue

        if key == "model.flavor":
            if value is not None:
                translated.extend(["--config", _config_name_from_flavor(value)])
            i += 2 if consume_next else 1
            continue

        if key == "model.tokenizer-backend":
            if value is not None:
                legacy_tokenizer_backend = value
            i += 2 if consume_next else 1
            continue

        if key.startswith("blendcorpus."):
            remapped = f"dataloader.{key.removeprefix('blendcorpus.')}"
        else:
            remapped = _LEGACY_KEY_REMAP.get(key, key)

        if value is None:
            translated.append(f"--{remapped}")
        else:
            translated.extend([f"--{remapped}", value])

        i += 2 if consume_next else 1

    if legacy_tokenizer_backend is not None:
        translated.extend(
            ["tokenizer:config", "--tokenizer.backend", legacy_tokenizer_backend]
        )

    return translated


def _ensure_rank_env() -> None:
    os.environ.setdefault("LOCAL_RANK", str(ezpz.distributed.get_local_rank()))
    if is_initialized():
        os.environ.setdefault("RANK", str(get_rank()))
        os.environ.setdefault("WORLD_SIZE", str(get_world_size()))


def _log_rank0_abort_chain(phase: str, exc: BaseException) -> None:
    """Emit a single rank-0 ABORT line with the chained cause of ``exc``.

    Startup failures (import errors, missing tokenizer/dataset deps,
    config errors) come from inside library code whose default exception
    formatting is opaque. ``ConfigManager._load_config`` in particular
    catches ``ImportError`` and re-raises with its own generic
    ``"Cannot import config_registry"`` message, hiding the real
    ``ModuleNotFoundError`` chain. mpiexec's per-rank stderr is
    interleaved across 24+ ranks, so even when a useful chain is
    propagated, the root cause typically gets buried by `rank N exited
    with code 1` lines.

    Walk ``__cause__`` then ``__context__`` and log a single rank-0
    summary so the launcher tail makes the cause discoverable.
    """
    if ezpz.distributed.get_rank() != 0:
        return
    chain: list[str] = []
    cur: BaseException | None = exc
    while cur is not None:
        chain.append(f"{type(cur).__name__}: {cur}")
        cur = cur.__cause__ or cur.__context__
    logger.error(
        "RANK 0 ABORT during %s:\n  %s",
        phase,
        "\n  caused by: ".join(chain),
    )


def main(args: list[str] | None = None) -> None:
    init_logger()

    import torchtitan

    logger.info(
        "torchtitan version: %s (0.0.0 means __version__ is not defined correctly).",
        torchtitan.__version__,
    )

    raw_args = sys.argv[1:] if args is None else args
    parsed_args = _inject_default_module_and_config(_translate_legacy_args(raw_args))

    # Extract --optimizer before tyro sees it (tyro only knows the base Config)
    optimizer_name, optimizer_overrides, parsed_args = _extract_optimizer_args(
        parsed_args
    )

    logger.info(f"\n{json.dumps(parsed_args, indent=4, sort_keys=True)}")
    config_manager = ConfigManager()
    try:
        config: Any = config_manager.parse_args(parsed_args)
    except Exception as parse_exc:
        # ``ConfigManager._load_config`` catches ``ImportError`` from
        # ``importlib.import_module(...config_registry)`` and re-raises a
        # generic "Cannot import config_registry for module 'X'" message
        # that hides the real ``ModuleNotFoundError`` chain. Surface the
        # chain on rank 0 so the launcher tail names the actually missing
        # module (e.g. ``ezpz``) instead of just the symptom.
        _log_rank0_abort_chain("config_manager.parse_args()", parse_exc)
        raise

    # Swap in the correct optimizer Config subclass if --optimizer was specified
    if optimizer_name is not None:
        config.optimizer = _build_optimizer_config(
            optimizer_name,
            config.optimizer,
            optimizer_overrides,
        )
        logger.info(
            "Using optimizer: %s (%s)", optimizer_name, type(config.optimizer).__name__
        )

    trainer = None

    try:
        if config.comm.mode == "local_tensor":
            logger.info("Local tensor mode enabled - skipping training execution")
            return

        try:
            trainer = config.build()
        except Exception as build_exc:
            # Build-time failures (import errors, config errors, missing
            # tokenizer/dataset deps, etc.) come from inside config.build
            # before the trainer's own exception handler runs.
            _log_rank0_abort_chain("config.build()", build_exc)
            raise

        # SophiaG requires a hessian EMA update each step before the param update
        if isinstance(trainer.optimizers, SophiaGOptimizersContainer):
            trainer.optimizers.register_step_pre_hook(
                lambda *_args, **_kwargs: trainer.optimizers.update_hessian()
            )

        if ezpz.distributed.get_rank() == 0 and ezpz.distributed.verify_wandb():
            try:
                run = ezpz.distributed.setup_wandb(
                    project_name=WBPROJ_NAME,
                    settings={"console": "wrap"},
                )
                wbconfig = {}
                wbconfig |= {"env": _update_env()}
                wbconfig |= config.to_dict()
                # wbconfig |= {"config": asdict(config)}
                wbconfig |= {"dist": ezpz.distributed.get_dist_info()}
                if run is not None:
                    run.config.update(wbconfig)
            except Exception as e:
                logger.warning("Unable to update `wandb.run.config`, continuing!")
                if ezpz.distributed.get_rank() == 0:
                    logger.exception(e)

        if config.checkpoint.create_seed_checkpoint:
            assert (
                int(os.environ["WORLD_SIZE"]) == 1
            ), "Must create seed checkpoint using a single device, to disable sharding."
            assert (
                config.checkpoint.enable
            ), "Must enable checkpointing when creating a seed checkpoint."
            trainer.checkpointer.save(curr_step=0, last_step=True)
            logger.info("Created seed checkpoint")
        elif config.lr_finder.enable:
            from torchtitan.experiments.ezpz.lr_finder import run_lr_finder

            run_lr_finder(trainer)
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
    ezpz.distributed.setup_torch()
    _ensure_rank_env()
    main()
    # Hard-exit after main() returns. Without this, mpiexec hangs
    # post-training waiting on a wedged C++ thread on most ranks (kernel
    # stack: one thread in __do_sys_pause + a non-daemon torch signal
    # handler that never returns). Python's normal shutdown can't finish
    # while a non-daemon thread is alive. os._exit bypasses the cleanup
    # chain; we've already destroy_process_group()'d and wandb has
    # flushed by this point, so there's nothing important left to run.
    # Kill the mp resource_tracker daemon first so it can't print
    # "leaked semaphore" warnings on shutdown (the semaphores are
    # kernel-cleaned anyway when the process group dies).
    try:
        from multiprocessing.resource_tracker import _resource_tracker as _rt
        import signal
        if getattr(_rt, "_pid", None) is not None:
            os.kill(_rt._pid, signal.SIGKILL)
    except Exception:
        pass
    os._exit(0)
