"""Compatibility adapters to reuse ezpz runtime conventions when available."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Optional


def _import_ezpz():
    try:
        import ezpz  # type: ignore

        return ezpz
    except Exception:
        return None


def get_machine_name() -> str:
    """Return machine name using ezpz heuristics when available."""
    ezpz = _import_ezpz()
    if ezpz is not None:
        try:
            return str(ezpz.get_machine()).lower()
        except Exception:
            pass

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


def get_scheduler_name(explicit: Optional[str] = None) -> str:
    ezpz = _import_ezpz()
    if ezpz is not None:
        try:
            from ezpz.configs import get_scheduler  # type: ignore

            return str(get_scheduler(_scheduler=explicit)).lower()
        except Exception:
            pass
    return (explicit or "unknown").lower()


def get_distributed_summary(hostfile: Optional[str] = None) -> dict[str, str]:
    """Expose selected distributed/runtime facts in a stable shape."""
    data = {
        "machine": get_machine_name(),
        "scheduler": "unknown",
        "backend": "unknown",
        "device": "unknown",
        "world_size_total": "unknown",
        "gpus_per_node": "unknown",
        "num_nodes": "unknown",
    }

    ezpz = _import_ezpz()
    if ezpz is None:
        return data

    try:
        from ezpz.configs import get_scheduler  # type: ignore

        data["scheduler"] = str(get_scheduler()).lower()
    except Exception:
        pass

    try:
        import ezpz.distributed as dist  # type: ignore

        data["backend"] = str(dist.get_torch_backend())
        data["device"] = str(dist.get_torch_device_type())
        data["world_size_total"] = str(dist.get_world_size_total())
        data["gpus_per_node"] = str(dist.get_gpus_per_node())
        if hostfile:
            data["num_nodes"] = str(dist.get_num_nodes(hostfile=hostfile))
        else:
            data["num_nodes"] = str(dist.get_num_nodes())
    except Exception:
        pass

    return data


def scrape_bad_nodes(log_path: Path, *, machine: Optional[str] = None) -> list[str]:
    """Use ezpz.failover.scrape_bad_nodes when installed; else return []."""
    ezpz = _import_ezpz()
    if ezpz is None:
        return []

    try:
        from ezpz.failover import scrape_bad_nodes as _scrape_bad_nodes  # type: ignore

        machine_name = machine or get_machine_name()
        return list(_scrape_bad_nodes(log_path, machine=machine_name))
    except Exception:
        return []
