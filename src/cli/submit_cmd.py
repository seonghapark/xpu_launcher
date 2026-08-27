"""Submit subsystem: scheduler-aware command materialization."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import os
from pathlib import Path

import click

from cli.ezpz_compat import get_machine_name
from cli.scheduler_topology import choose_launcher_binary, detect_scheduler, infer_topology, resolve_hostfile


def _build_launcher_command(
    scheduler: str,
    launcher: str,
    hostfile: str | None,
    nproc: int,
    nhosts: int,
    nproc_per_node: int,
) -> list[str]:
    if launcher in {"mpiexec", "mpirun"}:
        cmd = [launcher, "-n", str(nproc)]
        ppn_flag = "--ppn" if launcher == "mpiexec" else "-N"
        cmd.extend([ppn_flag, str(nproc_per_node)])
        if hostfile:
            cmd.extend(["--hostfile", hostfile])
        return cmd

    if launcher == "srun":
        cmd = [launcher, "-u", "--verbose", "-N", str(nhosts), "-n", str(nproc)]
        cmd.extend(["--ntasks-per-node", str(nproc_per_node)])
        if hostfile:
            cmd.append(f"--nodelist={Path(hostfile).resolve()}")
        return cmd

    if scheduler in {"pbs", "slurm"}:
        return ["<missing-launcher>"]
    return []


def _build_submit_script(
    *,
    scheduler: str,
    machine: str,
    job_name: str,
    time_limit: str,
    queue: str,
    account: str,
    filesystems: str,
    topology: dict[str, int],
    launch_command: list[str],
    command_text: str,
) -> str:
    if scheduler == "pbs":
        fs = filesystems.replace(",", ":")
        lines = [
            "#!/bin/bash --login",
            f"#PBS -l select={topology['nhosts']}",
            f"#PBS -l walltime={time_limit}",
            f"#PBS -l filesystems={fs}",
        ]
        if account:
            lines.append(f"#PBS -A {account}")

        # Aurora/Polaris are both ALCF PBS systems; keep explicit placement
        # and queue/job directives aligned with their common conventions.
        if machine in {"aurora", "polaris", "sunspot", "sophia"}:
            lines.append("#PBS -l place=scatter")

        lines += [
            "#PBS -k doe",
            "#PBS -j oe",
            f"#PBS -q {queue}",
            f"#PBS -N {job_name}",
            "",
            "set -eo pipefail",
            "cd \"${PBS_O_WORKDIR:-$PWD}\"",
        ]
    elif scheduler == "slurm":
        lines = [
            "#!/bin/bash --login",
            f"#SBATCH -J {job_name}",
            f"#SBATCH -N {topology['nhosts']}",
            f"#SBATCH -n {topology['nproc']}",
            f"#SBATCH -t {time_limit}",
        ]
        if account:
            lines.append(f"#SBATCH --account={account}")

        lines += [
            f"#SBATCH --partition={queue}",
            "",
            "set -eo pipefail",
            "cd \"${SLURM_SUBMIT_DIR:-$PWD}\"",
        ]
    else:
        lines = ["#!/bin/bash", "cd \"$PWD\""]

    launch_prefix = shlex.join(launch_command) if launch_command else ""
    run_line = f"{launch_prefix} {command_text}".strip()
    lines.append(run_line)
    return "\n".join(lines) + "\n"


def _submit_script(scheduler: str, script_path: Path) -> tuple[bool, str]:
    if scheduler == "pbs":
        if shutil.which("qsub") is None:
            return False, "qsub not found on PATH"
        proc = subprocess.run(["qsub", str(script_path)], check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout).strip()
        return True, (proc.stdout or "").strip()

    if scheduler == "slurm":
        if shutil.which("sbatch") is None:
            return False, "sbatch not found on PATH"
        proc = subprocess.run(["sbatch", str(script_path)], check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout).strip()
        return True, (proc.stdout or "").strip()

    return False, f"submit not supported for scheduler={scheduler}"


def _profile_defaults(machine: str, scheduler: str) -> dict[str, str]:
    if scheduler == "pbs" and machine in {"aurora", "sunspot", "sophia"}:
        return {
            "queue": "debug",
            "filesystems": "home:flare",
            "account": os.environ.get("PBS_ACCOUNT", os.environ.get("PROJECT", "")),
        }
    if scheduler == "pbs" and machine == "polaris":
        return {
            "queue": "debug",
            "filesystems": "home:eagle",
            "account": os.environ.get("PBS_ACCOUNT", os.environ.get("PROJECT", "")),
        }
    if scheduler == "slurm":
        return {
            "queue": "debug",
            "filesystems": "home",
            "account": os.environ.get("SLURM_ACCOUNT", os.environ.get("PROJECT", "")),
        }
    return {"queue": "debug", "filesystems": "home", "account": ""}


@click.command("submit")
@click.option("--hostfile", default=None)
@click.option("--nproc", type=int, default=None)
@click.option("--nhosts", type=int, default=None)
@click.option("--nproc-per-node", type=int, default=None)
@click.option("--command", "command_text", required=True, help="Command string to run")
@click.option("--job-name", default="xpu-job")
@click.option("--time", "time_limit", default="01:00:00")
@click.option("--queue", default=None, help="Scheduler queue/partition")
@click.option("--account", default=None, help="Scheduler account/project")
@click.option("--filesystems", default=None, help="PBS filesystems list (comma or colon separated)")
@click.option("--script-path", default=None, help="Output script path (default: ./xpu_submit_<scheduler>.sh)")
@click.option("--run/--no-run", default=False, help="Actually submit via qsub/sbatch")
def submit_cmd(
    hostfile: str | None,
    nproc: int | None,
    nhosts: int | None,
    nproc_per_node: int | None,
    command_text: str,
    job_name: str,
    time_limit: str,
    queue: str | None,
    account: str | None,
    filesystems: str | None,
    script_path: str | None,
    run: bool,
) -> None:
    scheduler = detect_scheduler()
    machine = get_machine_name()
    resolved = resolve_hostfile(hostfile, scheduler)
    topo = infer_topology(
        requested_nproc=nproc,
        requested_nhosts=nhosts,
        requested_nproc_per_node=nproc_per_node,
        hostfile=resolved,
    )
    launcher = choose_launcher_binary(scheduler) or "none"
    defaults = _profile_defaults(machine, scheduler)
    resolved_queue = queue or defaults["queue"]
    resolved_account = account if account is not None else defaults["account"]
    resolved_filesystems = filesystems or defaults["filesystems"]

    launch_command = _build_launcher_command(
        scheduler,
        launcher,
        resolved,
        topo.nproc,
        topo.nhosts,
        topo.nproc_per_node,
    )

    out_path = Path(script_path) if script_path else Path.cwd() / f"xpu_submit_{scheduler}.sh"
    script_text = _build_submit_script(
        scheduler=scheduler,
        machine=machine,
        job_name=job_name,
        time_limit=time_limit,
        queue=resolved_queue,
        account=resolved_account,
        filesystems=resolved_filesystems,
        topology={
            "nproc": topo.nproc,
            "nhosts": topo.nhosts,
            "nproc_per_node": topo.nproc_per_node,
        },
        launch_command=launch_command,
        command_text=command_text,
    )
    out_path.write_text(script_text, encoding="utf-8")

    submitted = False
    submit_message = "not submitted (--no-run)"
    if run:
        submitted, submit_message = _submit_script(scheduler, out_path)

    click.echo(
        json.dumps(
            {
                "scheduler": scheduler,
                "machine": machine,
                "launcher": launcher,
                "hostfile": resolved,
                "script_path": str(out_path),
                "queue": resolved_queue,
                "account": resolved_account,
                "filesystems": resolved_filesystems,
                "topology": {
                    "nproc": topo.nproc,
                    "nhosts": topo.nhosts,
                    "nproc_per_node": topo.nproc_per_node,
                },
                "launch_command": launch_command,
                "command": command_text,
                "submitted": submitted,
                "submit_message": submit_message,
            },
            indent=2,
            sort_keys=True,
        )
    )
