"""ezpz-free distributed runtime helpers for the compat experiment.

Replaces the subset of `ezpz.distributed` / `ezpz.utils` that train.py and
trainer.py used: rank/topology discovery from MPI/PALS/SLURM env, torch
distributed init with XPU/CUDA backend selection, and wandb gating.
"""

from __future__ import annotations

import datetime
import os
import socket
from datetime import timedelta
from typing import Any, Optional

import torch
import torch.distributed

_RANK_ENVS = (
    "RANK",
    "PMI_RANK",
    "PMIX_RANK",
    "PALS_RANKID",
    "OMPI_COMM_WORLD_RANK",
    "SLURM_PROCID",
)
_WORLD_ENVS = (
    "WORLD_SIZE",
    "PMI_SIZE",
    "PMIX_SIZE",
    "OMPI_COMM_WORLD_SIZE",
    "SLURM_NTASKS",
)
_LOCAL_RANK_ENVS = (
    "LOCAL_RANK",
    "PALS_LOCAL_RANKID",
    "PMI_LOCAL_RANK",
    "MPI_LOCALRANKID",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "SLURM_LOCALID",
)


def _env_int(names: tuple[str, ...], default: int) -> int:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            try:
                return int(value)
            except ValueError:
                continue
    return default


def get_rank() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return _env_int(_RANK_ENVS, 0)


def get_world_size() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return _env_int(_WORLD_ENVS, 1)


def get_local_rank() -> int:
    return _env_int(_LOCAL_RANK_ENVS, 0)


def get_hostname() -> str:
    return socket.gethostname()


def get_machine() -> str:
    hn = socket.getfqdn().lower()
    if "aurora" in hn or "sunspot" in hn:
        return "aurora"
    if "polaris" in hn:
        return "polaris"
    if "frontier" in hn:
        return "frontier"
    if "perlmutter" in hn:
        return "perlmutter"
    return "unknown"


def get_torch_backend() -> str:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xccl"
    if torch.cuda.is_available():
        return "nccl"
    return "gloo"


def get_timestamp(fmt: str = "%Y-%m-%d-%H%M%S") -> str:
    return datetime.datetime.now().strftime(fmt)


def get_dist_info() -> dict[str, Any]:
    return {
        "rank": get_rank(),
        "world_size": get_world_size(),
        "local_rank": get_local_rank(),
        "hostname": get_hostname(),
        "machine": get_machine(),
        "backend": get_torch_backend(),
        "master_addr": os.environ.get("MASTER_ADDR"),
        "master_port": os.environ.get("MASTER_PORT"),
    }


def verify_wandb() -> bool:
    """True when wandb is importable and credentialed (env or netrc)."""
    if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true"):
        return False
    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        return False
    try:
        import wandb

        return bool(wandb.api.api_key)
    except Exception:
        return False


def setup_wandb(project_name: str, settings: Optional[dict] = None):
    import wandb

    kwargs: dict[str, Any] = {"project": project_name}
    if settings:
        kwargs["settings"] = wandb.Settings(**settings)
    return wandb.init(**kwargs)


def _resolve_master_addr() -> str:
    addr = os.environ.get("MASTER_ADDR")
    if addr:
        return addr
    nodefile = os.environ.get("PBS_NODEFILE")
    if nodefile and os.path.isfile(nodefile):
        with open(nodefile, encoding="utf-8") as fh:
            for line in fh:
                host = line.strip()
                if host:
                    return host
    return socket.gethostname()


def _resolve_master_port() -> str:
    port = os.environ.get("MASTER_PORT")
    if port:
        return port
    # Deterministic per-job port so all ranks agree without communication
    jobid = os.environ.get("PBS_JOBID") or os.environ.get("SLURM_JOB_ID") or "0"
    digits = "".join(ch for ch in jobid if ch.isdigit()) or "0"
    return str(29500 + int(digits[-4:]) % 1000)


def setup_torch(timeout_minutes: int = 30) -> None:
    """Bind the local accelerator and initialize torch.distributed from env."""
    rank = get_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()

    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ["MASTER_ADDR"] = _resolve_master_addr()
    os.environ["MASTER_PORT"] = _resolve_master_port()

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.set_device(local_rank % max(1, torch.xpu.device_count()))
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank % max(1, torch.cuda.device_count()))

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend=get_torch_backend(),
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=timeout_minutes),
        )
