"""Accelerator backends (cuda/xpu/rocm) re-exported for library users."""

from __future__ import annotations

from cli.accelerators import (
    ACCELERATOR_CHOICES,
    BACKENDS,
    PUBLIC_API,
    cuda,
    detect_accelerator,
    get_accelerator,
    missing_api,
    rocm,
    xpu,
)

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
