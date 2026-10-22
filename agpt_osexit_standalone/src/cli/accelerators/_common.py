"""Shared torch-free helpers for accelerator backend modules."""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Optional


def torch_module():
    """Return the torch module when importable, else None (torch is optional)."""
    try:
        import torch  # type: ignore

        return torch
    except Exception:
        return None


def which(binary: str) -> bool:
    return shutil.which(binary) is not None


def any_dev_nodes(*patterns: str) -> bool:
    return any(glob.glob(pattern) for pattern in patterns)


def any_env(names: Sequence[str], env: Optional[Mapping[str, str]] = None) -> bool:
    env_map = os.environ if env is None else env
    return any(env_map.get(name) for name in names)


def run_and_count(
    cmd: Sequence[str],
    line_rx: Optional[str] = None,
    *,
    timeout_s: float = 10.0,
) -> Optional[int]:
    """Run a query command and count matching output lines; None on failure."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        proc = subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    if line_rx is None:
        count = sum(1 for line in proc.stdout.splitlines() if line.strip())
    else:
        rx = re.compile(line_rx)
        count = len({m.group(0) for m in rx.finditer(proc.stdout)})
    return count or None


def parse_device_list(value: Optional[str]) -> Optional[list[str]]:
    if value is None:
        return None
    devices = [item.strip() for item in value.split(",") if item.strip()]
    return devices


def format_device_list(devices: Sequence[int | str]) -> str:
    return ",".join(str(device) for device in devices)


def assign_env(
    env: Optional[MutableMapping[str, str]],
    assignments: Mapping[str, str],
) -> dict[str, str]:
    env_map: MutableMapping[str, str] = os.environ if env is None else env
    for key, value in assignments.items():
        env_map[key] = value
    return dict(assignments)
