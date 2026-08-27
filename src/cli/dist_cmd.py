"""Dist subsystem wrappers for topology/runtime introspection."""

from __future__ import annotations

import json
import os

import click

from cli.ezpz_compat import get_distributed_summary
from cli.scheduler_topology import detect_scheduler, infer_topology, resolve_hostfile


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


@click.group("dist")
def dist_cmd() -> None:
    """Distributed/runtime helpers."""


@dist_cmd.command("summary")
@click.option("--hostfile", default=None)
def dist_summary(hostfile: str | None) -> None:
    scheduler = detect_scheduler()
    resolved = resolve_hostfile(hostfile, scheduler)
    click.echo(json.dumps(get_distributed_summary(hostfile=resolved), indent=2, sort_keys=True))


@dist_cmd.command("topology")
@click.option("--hostfile", default=None)
@click.option("--nproc", type=int, default=None)
@click.option("--nhosts", type=int, default=None)
@click.option("--nproc-per-node", type=int, default=None)
def dist_topology(
    hostfile: str | None,
    nproc: int | None,
    nhosts: int | None,
    nproc_per_node: int | None,
) -> None:
    scheduler = detect_scheduler()
    resolved = resolve_hostfile(hostfile, scheduler)
    topo = infer_topology(
        requested_nproc=nproc,
        requested_nhosts=nhosts,
        requested_nproc_per_node=nproc_per_node,
        hostfile=resolved,
    )
    click.echo(
        json.dumps(
            {
                "nproc": topo.nproc,
                "nhosts": topo.nhosts,
                "nproc_per_node": topo.nproc_per_node,
                "max_nhosts": topo.max_nhosts,
                "max_nproc_per_node": topo.max_nproc_per_node,
            },
            indent=2,
            sort_keys=True,
        )
    )


@dist_cmd.command("validate")
@click.option("--hostfile", default=None)
@click.option("--nproc", type=int, default=None)
@click.option("--nhosts", type=int, default=None)
@click.option("--nproc-per-node", type=int, default=None)
def dist_validate(
    hostfile: str | None,
    nproc: int | None,
    nhosts: int | None,
    nproc_per_node: int | None,
) -> None:
    scheduler = detect_scheduler()
    resolved = resolve_hostfile(hostfile, scheduler)
    topo = infer_topology(
        requested_nproc=nproc,
        requested_nhosts=nhosts,
        requested_nproc_per_node=nproc_per_node,
        hostfile=resolved,
    )
    summary = get_distributed_summary(hostfile=resolved)

    issues: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []

    env_world_size = _env_int("WORLD_SIZE")
    if env_world_size is not None and env_world_size != topo.nproc:
        issues.append(
            {
                "name": "env.world_size_mismatch",
                "expected": topo.nproc,
                "actual": env_world_size,
            }
        )

    env_local_world_size = _env_int("LOCAL_WORLD_SIZE")
    if env_local_world_size is not None and env_local_world_size != topo.nproc_per_node:
        warnings.append(
            {
                "name": "env.local_world_size_mismatch",
                "expected": topo.nproc_per_node,
                "actual": env_local_world_size,
            }
        )

    summary_world_size = summary.get("world_size_total")
    if isinstance(summary_world_size, str) and summary_world_size.isdigit():
        sw = int(summary_world_size)
        if sw < topo.nproc:
            warnings.append(
                {
                    "name": "dist.world_size_total_smaller_than_requested",
                    "requested": topo.nproc,
                    "world_size_total": sw,
                }
            )

    payload = {
        "scheduler": scheduler,
        "hostfile": resolved,
        "topology": {
            "nproc": topo.nproc,
            "nhosts": topo.nhosts,
            "nproc_per_node": topo.nproc_per_node,
            "max_nhosts": topo.max_nhosts,
            "max_nproc_per_node": topo.max_nproc_per_node,
        },
        "distributed": summary,
        "issues": issues,
        "warnings": warnings,
        "ok": len(issues) == 0,
    }
    click.echo(json.dumps(payload, indent=2, sort_keys=True))
