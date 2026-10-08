"""TorchTitan entry point for MPI/PBS launches on XPU, CUDA, and CPU."""

from __future__ import annotations

import os
import socket
import sys
import warnings
from pathlib import Path

# Silence import-time warnings on every rank (set XPU_SHOW_WARNINGS=1 to keep them)
if os.environ.get("XPU_SHOW_WARNINGS", "0") != "1":
    warnings.filterwarnings("ignore")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")  # inherited by worker subprocesses
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

_TORCHTITAN_ROOT = Path(
    os.environ.get("TORCHTITAN_ROOT", Path(__file__).resolve().parent / "torchtitan_repo")
)
if str(_TORCHTITAN_ROOT) not in sys.path:
    sys.path.insert(0, str(_TORCHTITAN_ROOT))


def _extend_torchtitan_path() -> None:
    """Let the installed torchtitan backfill submodules missing from the repo.

    ``torchtitan_repo/torchtitan`` is a regular package, so putting
    TORCHTITAN_ROOT first on sys.path makes it *fully* shadow the
    ``torchtitan`` installed in site-packages. The two trees have
    complementary gaps: the repo carries ``models/agpt`` but no
    ``hf_datasets`` / ``experiments``, while the installed 0.3.0 wheel has
    those two but no ``models/agpt``. Since ``components/validate.py`` and
    every model registry import ``torchtitan.hf_datasets`` unguarded, a
    shadowed-but-incomplete repo fails at import time.

    Appending the installed package directory to ``torchtitan.__path__``
    keeps repo modules winning (its entry stays first) while missing
    submodules fall through to the wheel. Set TORCHTITAN_NO_PATH_FALLBACK=1
    to disable.
    """
    if os.environ.get("TORCHTITAN_NO_PATH_FALLBACK", "0") == "1":
        return

    import importlib
    import importlib.util

    repo_pkg = _TORCHTITAN_ROOT / "torchtitan"
    if not repo_pkg.is_dir():
        return  # not the shadowing layout; nothing to repair

    # Locate the installed copy without importing torchtitan first.
    installed: str | None = None
    for entry in sys.path:
        if not entry or Path(entry) == _TORCHTITAN_ROOT:
            continue
        candidate = Path(entry) / "torchtitan" / "__init__.py"
        if candidate.is_file():
            installed = str(candidate.parent)
            break

    if installed is None:
        return  # only one copy present; nothing to backfill

    pkg = importlib.import_module("torchtitan")
    if installed not in pkg.__path__:
        pkg.__path__.append(installed)


_extend_torchtitan_path()

import torch

# --- XPU workaround 1: IPEX operator overrides for TP collectives ---
if torch.__version__ < "2.11":
    try:
        import intel_extension_for_pytorch  # noqa: F401
    except Exception:
        pass

# torch 2.10+ natively supports Enum in torch.compile; torchao (read-only
# system site-packages) still calls register_constant() on Enums, emitting a
# deprecation warning per rank. Make that call a no-op for Enum subclasses.
if torch.__version__ >= "2.10":
    import enum

    import torch.utils._pytree as _pytree

    _orig_register_constant = _pytree.register_constant

    def _register_constant_skip_enums(cls, *args, **kwargs):
        if isinstance(cls, type) and issubclass(cls, enum.Enum):
            return cls
        return _orig_register_constant(cls, *args, **kwargs)

    _pytree.register_constant = _register_constant_skip_enums

_RANK_ENVS = (
    "RANK", "PMI_RANK", "PMIX_RANK", "PALS_RANKID",
    "OMPI_COMM_WORLD_RANK", "SLURM_PROCID",
)
_WORLD_SIZE_ENVS = (
    "WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "OMPI_COMM_WORLD_SIZE", "SLURM_NTASKS",
)
_LOCAL_RANK_ENVS = (
    "LOCAL_RANK", "PALS_LOCAL_RANKID", "PMI_LOCAL_RANK", "MPI_LOCALRANKID",
    "OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID",
)


def _first_env_int(names: tuple[str, ...], default: int = 0) -> int:
    for name in names:
        value = os.environ.get(name)
        if value:
            try:
                return int(value)
            except ValueError:
                continue
    return default


def _world_size() -> int:
    world_size = _first_env_int(_WORLD_SIZE_ENVS)
    if world_size > 0:
        return world_size
    local_size = _first_env_int(("PALS_LOCAL_SIZE", "PMI_LOCAL_SIZE"))
    nodefile = os.environ.get("PBS_NODEFILE")
    if local_size > 0 and nodefile and os.path.isfile(nodefile):
        with open(nodefile, encoding="utf-8") as file:
            node_count = len({line.strip() for line in file if line.strip()})
        if node_count:
            return local_size * node_count
    return 1


def _master_addr() -> str:
    nodefile = os.environ.get("PBS_NODEFILE")
    if nodefile and os.path.isfile(nodefile):
        with open(nodefile, encoding="utf-8") as file:
            for line in file:
                if host := line.strip():
                    return host
    return socket.gethostname()


def _master_port() -> str:
    job_id = os.environ.get("PBS_JOBID") or os.environ.get("SLURM_JOB_ID") or "0"
    digits = "".join(character for character in job_id if character.isdigit()) or "0"
    return str(29500 + int(digits[-4:]) % 1000)


def _normalize_distributed_env() -> None:
    os.environ.setdefault("RANK", str(_first_env_int(_RANK_ENVS)))
    os.environ.setdefault("WORLD_SIZE", str(_world_size()))
    os.environ.setdefault("LOCAL_RANK", str(_first_env_int(_LOCAL_RANK_ENVS)))
    os.environ.setdefault("MASTER_ADDR", _master_addr())
    os.environ.setdefault("MASTER_PORT", _master_port())


def main() -> None:
    _normalize_distributed_env()
    from torchtitan.distributed.xccl_split_group_workaround import (
        maybe_install_xccl_split_group_workaround,
    )
    from torchtitan.train import main as torchtitan_main

    maybe_install_xccl_split_group_workaround()
    torchtitan_main()


if __name__ == "__main__":
    main()
