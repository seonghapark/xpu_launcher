"""Integrations subsystem: optional runtime integrations inventory."""

from __future__ import annotations

import importlib
import json

import click


@click.command("integrations")
def integrations_cmd() -> None:
    checks = {}
    for name in ("torch", "deepspeed", "horovod", "wandb", "mlflow"):
        try:
            importlib.import_module(name)
            checks[name] = "available"
        except Exception:
            checks[name] = "missing"
    click.echo(json.dumps({"integrations": checks}, indent=2, sort_keys=True))
