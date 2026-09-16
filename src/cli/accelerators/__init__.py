"""Accelerator backend registry: XPU (Intel), CUDA (NVIDIA), ROCm (AMD).

Backend-specific functionality lives in one module per backend and every
module exposes the exact same public functions (`PUBLIC_API`), so callers
use `cuda.func(...)`, `xpu.func(...)`, `rocm.func(...)` interchangeably.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from types import ModuleType
from typing import Optional

from cli.accelerators import cuda, rocm, xpu

# Detection priority: XPU first (Aurora), then CUDA, then ROCm.
BACKENDS: dict[str, ModuleType] = {
    "xpu": xpu,
    "cuda": cuda,
    "rocm": rocm,
}

ACCELERATOR_CHOICES = ("auto", *BACKENDS.keys(), "none")

# Common API every backend module must implement.
PUBLIC_API: tuple[str, ...] = (
    "name",
    "is_available",
    "device_count",
    "visible_devices_env",
    "extra_visible_devices_envs",
    "visible_devices",
    "set_visible_devices",
    "distributed_backend",
    "collective_library",
    "smi_binary",
    "smi_query_command",
    "env_hints",
    "crash_patterns",
    "doctor_payload",
)

_ENV_OVERRIDES = ("XPU_LAUNCH_ACCELERATOR", "XPU_ACCELERATOR")


def missing_api(module: ModuleType) -> list[str]:
    """Return PUBLIC_API functions absent from a backend module."""
    return [func for func in PUBLIC_API if not callable(getattr(module, func, None))]


def get_accelerator(name: str) -> Optional[ModuleType]:
    """Return the backend module for a name; None for 'none'/'auto'/unknown."""
    return BACKENDS.get(name)


def detect_accelerator(
    explicit: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> str:
    """Resolve the active accelerator: explicit > env override > probing."""
    if explicit and explicit != "auto":
        if explicit != "none" and explicit not in BACKENDS:
            raise ValueError(
                f"unknown accelerator {explicit!r}; expected one of {ACCELERATOR_CHOICES}"
            )
        return explicit

    env_map = os.environ if env is None else env
    for key in _ENV_OVERRIDES:
        override = (env_map.get(key) or "").strip().lower()
        if override in BACKENDS or override == "none":
            return override

    for name, module in BACKENDS.items():
        if module.is_available(env=env):
            return name
    return "none"


__all__ = [
    "ACCELERATOR_CHOICES",
    "BACKENDS",
    "PUBLIC_API",
    "cuda",
    "detect_accelerator",
    "get_accelerator",
    "missing_api",
    "rocm",
    "xpu",
]
