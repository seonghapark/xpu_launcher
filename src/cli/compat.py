"""Standalone runtime compatibility helpers (machine, scheduler, summary)."""

from __future__ import annotations

import socket
from typing import Optional


def get_machine_name() -> str:
    """Return a coarse machine name from the local hostname."""
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
    return (explicit or "unknown").lower()


def get_distributed_summary(hostfile: Optional[str] = None) -> dict[str, str]:
    """Expose selected distributed/runtime facts in a stable shape."""
    from cli import accelerators
    from cli.scheduler_topology import (
        detect_scheduler,
        get_devices_per_node_hint,
        get_num_nodes_from_hostfile,
    )

    data = {
        "machine": get_machine_name(),
        "scheduler": "unknown",
        "backend": "unknown",
        "device": "unknown",
        "world_size_total": "unknown",
        "gpus_per_node": "unknown",
        "num_nodes": "unknown",
    }

    try:
        data["scheduler"] = detect_scheduler()
    except Exception:
        pass

    accelerator = accelerators.detect_accelerator()
    backend = accelerators.get_accelerator(accelerator)
    if backend is not None:
        data["device"] = accelerator
        data["backend"] = backend.distributed_backend()
        gpus_per_node = get_devices_per_node_hint() or backend.device_count()
        if gpus_per_node:
            data["gpus_per_node"] = str(gpus_per_node)

    num_nodes = get_num_nodes_from_hostfile(hostfile)
    if num_nodes is not None:
        data["num_nodes"] = str(num_nodes)
        if data["gpus_per_node"] != "unknown":
            data["world_size_total"] = str(num_nodes * int(data["gpus_per_node"]))

    return data
