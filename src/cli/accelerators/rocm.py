"""ROCm (AMD) backend functionality for xpu launch.

Every accelerator backend module (cuda, xpu, rocm) exposes the same public
functions so callers can write `cuda.func(...)`, `xpu.func(...)`,
`rocm.func(...)` interchangeably.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Optional

from cli.accelerators import _common

NAME = "rocm"


def name() -> str:
    return NAME


def visible_devices_env() -> str:
    return "ROCR_VISIBLE_DEVICES"


def extra_visible_devices_envs() -> tuple[str, ...]:
    return ("HIP_VISIBLE_DEVICES",)


def distributed_backend() -> str:
    """torch.distributed backend name (RCCL registers as nccl in PyTorch)."""
    return "nccl"


def collective_library() -> str:
    return "rccl"


def smi_binary() -> str:
    return "rocm-smi"


def smi_query_command() -> list[str]:
    return [smi_binary(), "--showuse", "--showpower", "--showmemuse", "--csv"]


def env_hints() -> tuple[str, ...]:
    return (
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCM_PATH",
        "ROCM_HOME",
        "HSA_OVERRIDE_GFX_VERSION",
    )


def is_available(env: Optional[Mapping[str, str]] = None) -> bool:
    torch = _common.torch_module()
    if torch is not None:
        try:
            # ROCm builds expose torch.cuda with torch.version.hip set.
            if torch.cuda.is_available() and torch.version.hip is not None:
                return True
        except Exception:
            pass
    if _common.which(smi_binary()):
        return True
    if _common.any_dev_nodes("/dev/kfd"):
        return True
    return _common.any_env(env_hints(), env=env)


def device_count(env: Optional[Mapping[str, str]] = None) -> int:
    torch = _common.torch_module()
    if torch is not None:
        try:
            if torch.cuda.is_available() and torch.version.hip is not None:
                return int(torch.cuda.device_count())
        except Exception:
            pass
    counted = _common.run_and_count(
        [smi_binary(), "--showid"], line_rx=r"GPU\[\d+\]"
    )
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
    assignments = {
        visible_devices_env(): value,
        **{key: value for key in extra_visible_devices_envs()},
    }
    return _common.assign_env(env, assignments)


def crash_patterns() -> str:
    """Regex alternation fragment for ROCm/HIP/RCCL failure signatures."""
    return (
        r"HIP error"
        r"|hipError(?:_t)?"
        r"|hipErrorOutOfMemory"
        r"|HSA_STATUS_ERROR_[A-Z_]+"
        r"|RCCL (?:error|WARN)"
        r"|rccl(?:SystemError|InternalError)"
        r"|Memory access fault by GPU"
        r"|amdgpu.*(?:hang|reset)"
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
