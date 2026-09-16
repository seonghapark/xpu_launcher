"""Intel XPU backend functionality for xpu launch.

Every accelerator backend module (cuda, xpu, rocm) exposes the same public
functions so callers can write `cuda.func(...)`, `xpu.func(...)`,
`rocm.func(...)` interchangeably.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Optional

from cli.accelerators import _common

NAME = "xpu"


def name() -> str:
    return NAME


def visible_devices_env() -> str:
    return "ZE_AFFINITY_MASK"


def extra_visible_devices_envs() -> tuple[str, ...]:
    return ()


def distributed_backend() -> str:
    """torch.distributed backend name (native xccl; oneCCL registers as ccl)."""
    return "xccl"


def collective_library() -> str:
    return "oneccl"


def smi_binary() -> str:
    return "xpu-smi"


def smi_query_command() -> list[str]:
    # Metric IDs 0,1,8,18: GPU util, power, memory used, temperature.
    return [smi_binary(), "dump", "-d", "-1", "-m", "0,1,8,18", "-n", "1"]


def env_hints() -> tuple[str, ...]:
    return (
        "ZE_AFFINITY_MASK",
        "ZE_FLAT_DEVICE_HIERARCHY",
        "ONEAPI_ROOT",
        "CCL_ATL_TRANSPORT",
    )


def is_available(env: Optional[Mapping[str, str]] = None) -> bool:
    torch = _common.torch_module()
    if torch is not None:
        try:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return True
        except Exception:
            pass
    if _common.which(smi_binary()):
        return True
    return _common.any_env(env_hints(), env=env)


def device_count(env: Optional[Mapping[str, str]] = None) -> int:
    torch = _common.torch_module()
    if torch is not None:
        try:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return int(torch.xpu.device_count())
        except Exception:
            pass
    counted = _common.run_and_count(
        [smi_binary(), "discovery"], line_rx=r"Device ID:?\s*\S+"
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
    return _common.assign_env(env, {visible_devices_env(): value})


def crash_patterns() -> str:
    """Regex alternation fragment for XPU/Level Zero/oneCCL failure signatures."""
    return (
        r"UR_RESULT_ERROR_[A-Z_]+"
        r"|ZE_RESULT_ERROR_[A-Z_]+"
        r"|Level.?Zero error"
        r"|oneccl"
        r"|\bxccl\b"
        r"|ccl::exception"
        r"|SYCL exception"
        r"|PI_ERROR_[A-Z_]+"
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
