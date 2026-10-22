"""Public Python package namespace for xpu_launch."""

from __future__ import annotations

from cli.__about__ import __version__
from xpu_launch import accelerators
from xpu_launch.cli import main

__all__ = [
    "__version__",
    "accelerators",
    "main",
]
