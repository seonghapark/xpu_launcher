"""CUDA (NVIDIA) backend functionality for xpu launch.

Every accelerator backend module (cuda, xpu, rocm) exposes the same public
functions so callers can write `cuda.func(...)`, `xpu.func(...)`,
`rocm.func(...)` interchangeably.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Optional

from cli.accelerators import _common

NAME = "cuda"


def name() -> str:
    return NAME


def visible_devices_env() -> str:
    return "CUDA_VISIBLE_DEVICES"


def extra_visible_devices_envs() -> tuple[str, ...]:
    return ("NVIDIA_VISIBLE_DEVICES",)


def distributed_backend() -> str:
    """torch.distributed backend name."""
    return "nccl"


def collective_library() -> str:
    return "nccl"


def smi_binary() -> str:
    return "nvidia-smi"


def smi_query_command() -> list[str]:
    return [
        smi_binary(),
        "--query-gpu=index,utilization.gpu,power.draw,memory.used",
        "--format=csv,noheader,nounits",
    ]


def env_hints() -> tuple[str, ...]:
    return (
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "CUDA_HOME",
        "CUDA_ROOT",
    )


def is_available(env: Optional[Mapping[str, str]] = None) -> bool:
    torch = _common.torch_module()
    if torch is not None:
        try:
            # torch.version.hip is not None means ROCm, not CUDA.
            if torch.cuda.is_available() and torch.version.hip is None:
                return True
        except Exception:
            pass
    if _common.which(smi_binary()):
        return True
    if _common.any_dev_nodes("/dev/nvidia[0-9]*", "/dev/nvidiactl"):
        return True
    return _common.any_env(env_hints(), env=env)


def device_count(env: Optional[Mapping[str, str]] = None) -> int:
    torch = _common.torch_module()
    if torch is not None:
        try:
            if torch.cuda.is_available() and torch.version.hip is None:
                return int(torch.cuda.device_count())
        except Exception:
            pass
    counted = _common.run_and_count([smi_binary(), "--list-gpus"])
    if counted is not None:
        return counted
    env_map = os.environ if env is None else env
    devices = _common.parse_device_list(env_map.get(visible_devices_env()))
    if devices is not None:
        return len(devices)
    return 0


def visible_devices(env: Optional[Mapping[str, str]] = None) -> Optional[list[str]]:
    env_map = os.environ if env is None else env
    for key in (visible_devices_env(), *extra_visible_devices_envs()):
        devices = _common.parse_device_list(env_map.get(key))
        if devices is not None:
            return devices
    return None


def set_visible_devices(
    devices: Sequence[int | str],
    env: Optional[MutableMapping[str, str]] = None,
) -> dict[str, str]:
    value = _common.format_device_list(devices)
    return _common.assign_env(env, {visible_devices_env(): value})


def crash_patterns() -> str:
    """Regex alternation fragment for CUDA/NCCL failure signatures."""
    return (
        r"CUDA error"
        r"|CUDA out of memory"
        r"|torch\.cuda\.OutOfMemoryError"
        r"|cudaError(?:_t)?"
        r"|CUDA_ERROR_[A-Z_]+"
        r"|NCCL (?:error|WARN)"
        r"|nccl(?:SystemError|InternalError|UnhandledCudaError|RemoteError)"
        r"|NVIDIA-SMI has failed"
        r"|Xid \d+"
    )


def doctor_payload(env: Optional[Mapping[str, str]] = None) -> dict[str, object]:
    env_map = os.environ if env is None else env
    present_hints = [key for key in env_hints() if env_map.get(key)]
    return {
        "name": NAME,
        "available": is_available(env=env),
        "device_count": device_count(env=env),
        "smi_binary": smi_binary(),
        "smi_on_path": _common.which(smi_binary()),
        "distributed_backend": distributed_backend(),
        "collective_library": collective_library(),
        "visible_devices_env": visible_devices_env(),
        "visible_devices": visible_devices(env=env),
        "env_hints_present": present_hints,
    }
