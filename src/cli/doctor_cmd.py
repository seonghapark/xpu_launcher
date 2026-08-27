"""Doctor subsystem: runtime diagnostics and compatibility checks."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess

import click

from cli.ezpz_compat import get_distributed_summary
from cli.scheduler_topology import choose_launcher_binary, detect_scheduler, infer_topology, resolve_hostfile


def _check(name: str, ok: bool, details: str) -> dict[str, object]:
    return {"name": name, "ok": ok, "details": details}


def _launcher_family(
    launcher_path: str,
    launcher_realpath: str,
    ldd_text: str,
    env: dict[str, str] | None = None,
) -> str:
    lower_path = launcher_path.lower()
    lower_realpath = launcher_realpath.lower()
    lower_ldd = ldd_text.lower()
    env = env or {}
    if "/pals/" in lower_path or "/pals/" in lower_realpath or "pals" in lower_ldd:
        return "pals"
    if "mpich" in lower_path or "mpich" in lower_realpath or "libmpich" in lower_ldd:
        return "mpich"
    if (
        "openmpi" in lower_path
        or "openmpi" in lower_realpath
        or "libopen-rte" in lower_ldd
    ):
        return "openmpi"
    if "srun" in lower_path or "srun" in lower_realpath:
        return "slurm"
    if env.get("PALS_PMI") or env.get("PALS_LOCAL_RANKID"):
        return "pals"
    if env.get("MPICH_ROOT"):
        return "mpich"
    if env.get("OMPI_COMM_WORLD_RANK"):
        return "openmpi"
    if env.get("SLURM_JOB_ID"):
        return "slurm"
    return "unknown"


def _ldd_summary(binary: str) -> dict[str, object]:
    """Return linked MPI/PMI sonames and presence flags for a binary."""
    if shutil.which(binary) is None:
        return {
            "ok": False,
            "details": f"{binary} not found",
            "launcher_path": "",
            "launcher_realpath": "",
            "launcher_family": "unknown",
            "mpi_sonames": [],
            "pmi_sonames": [],
            "has_mpi_lib": False,
            "has_pmi_lib": False,
        }
    launcher_path = shutil.which(binary) or binary
    launcher_realpath = os.path.realpath(launcher_path)
    try:
        proc = subprocess.run(
            ["ldd", launcher_realpath],
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
        )
    except OSError as exc:
        return {
            "ok": False,
            "details": f"ldd failed: {exc}",
            "launcher_path": launcher_path,
            "launcher_realpath": launcher_realpath,
            "launcher_family": "unknown",
            "mpi_sonames": [],
            "pmi_sonames": [],
            "has_mpi_lib": False,
            "has_pmi_lib": False,
        }

    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    sonames = sorted(set(re.findall(r"\b(lib[^\s/]+\.so(?:\.[0-9]+)*)\b", text)))
    mpi_sonames = [
        s
        for s in sonames
        if any(tok in s.lower() for tok in ("libmpi", "libmpich", "libopen-rte"))
    ]
    pmi_sonames = [
        s
        for s in sonames
        if any(tok in s.lower() for tok in ("libpmi", "libpmix"))
    ]
    has_mpi = len(mpi_sonames) > 0
    has_pmi = len(pmi_sonames) > 0
    family = _launcher_family(
        launcher_path,
        launcher_realpath,
        text,
        dict(os.environ),
    )
    return {
        "ok": proc.returncode == 0,
        "details": f"mpi={has_mpi}, pmi={has_pmi}",
        "launcher_path": launcher_path,
        "launcher_realpath": launcher_realpath,
        "launcher_family": family,
        "mpi_sonames": mpi_sonames,
        "pmi_sonames": pmi_sonames,
        "has_mpi_lib": has_mpi,
        "has_pmi_lib": has_pmi,
    }


@click.command("doctor")
@click.option("--hostfile", default=None, help="Optional hostfile to inspect.")
@click.option("--nproc", type=int, default=None)
@click.option("--nhosts", type=int, default=None)
@click.option("--nproc-per-node", type=int, default=None)
def doctor_cmd(
    hostfile: str | None,
    nproc: int | None,
    nhosts: int | None,
    nproc_per_node: int | None,
) -> None:
    scheduler = detect_scheduler()
    resolved = resolve_hostfile(hostfile, scheduler)
    summary = get_distributed_summary(hostfile=resolved)
    checks: list[dict[str, object]] = []

    checks.append(
        _check(
            "scheduler.detected",
            scheduler in {"pbs", "slurm", "none", "unknown"},
            f"scheduler={scheduler}",
        )
    )

    checks.append(
        _check(
            "hostfile.resolved",
            resolved is not None,
            f"hostfile={resolved}",
        )
    )

    if scheduler == "pbs":
        has_pbs_env = bool(os.environ.get("PBS_JOBID") or os.environ.get("PBS_NODEFILE"))
        has_qsub = shutil.which("qsub") is not None
        checks.append(
            _check(
                "pbs.environment",
                has_pbs_env or has_qsub,
                "expected PBS_JOBID/PBS_NODEFILE or qsub on PATH",
            )
        )

    if scheduler == "slurm":
        has_slurm_env = bool(
            os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_NODELIST")
        )
        has_sbatch = shutil.which("sbatch") is not None
        checks.append(
            _check(
                "slurm.environment",
                has_slurm_env or has_sbatch,
                "expected SLURM_JOB_ID/SLURM_NODELIST or sbatch on PATH",
            )
        )

    launcher_bins = {
        "mpiexec": shutil.which("mpiexec") is not None,
        "mpirun": shutil.which("mpirun") is not None,
        "srun": shutil.which("srun") is not None,
    }
    checks.append(
        _check(
            "launchers.available",
            any(launcher_bins.values()),
            json.dumps(launcher_bins, sort_keys=True),
        )
    )

    launcher = choose_launcher_binary(scheduler) or "none"
    ldd_info: dict[str, object] = {
        "ok": False,
        "details": "launcher missing",
        "launcher_path": "",
        "launcher_realpath": "",
        "launcher_family": "unknown",
        "mpi_sonames": [],
        "pmi_sonames": [],
        "has_mpi_lib": False,
        "has_pmi_lib": False,
    }
    if launcher != "none":
        ldd_info = _ldd_summary(launcher)

    has_mpi_lib = bool(ldd_info["has_mpi_lib"])
    has_pmi_lib = bool(ldd_info["has_pmi_lib"])

    checks.append(
        _check(
            "mpi.launcher.detected",
            launcher != "none",
            f"launcher={launcher}",
        )
    )
    checks.append(
        _check(
            "mpi.launcher.family",
            launcher != "none",
            str(ldd_info["launcher_family"]),
        )
    )
    checks.append(
        {
            "name": "mpi.libs.detected",
            "ok": has_mpi_lib,
            "details": str(ldd_info["details"]),
            "launcher_path": ldd_info["launcher_path"],
            "launcher_realpath": ldd_info["launcher_realpath"],
            "launcher_family": ldd_info["launcher_family"],
            "mpi_sonames": ldd_info["mpi_sonames"],
            "pmi_sonames": ldd_info["pmi_sonames"],
        }
    )

    pmi_env_keys = [
        "PMI_RANK",
        "PMI_SIZE",
        "PMI_LOCAL_RANK",
        "PMI_FD",
        "PMI_PORT",
        "PMI_CONTROL_PORT",
        "PMI_ID",
        "PMIX_RANK",
        "PMIX_SIZE",
        "PALS_LOCAL_RANKID",
        "PALS_PMI",
    ]
    present_pmi_env = [k for k in pmi_env_keys if os.environ.get(k) is not None]
    checks.append(
        _check(
            "pmi.env.detected",
            bool(present_pmi_env) or scheduler in {"none", "unknown"},
            (
                f"present={present_pmi_env}"
                if present_pmi_env
                else "no PMI/PMIx env vars detected"
            ),
        )
    )

    launcher_needs_pmi = launcher in {"mpiexec", "mpirun", "srun"}
    launcher_family = str(ldd_info["launcher_family"])
    pals_runtime_signal = bool(
        os.environ.get("PALS_PMI")
        or os.environ.get("MPICH_ROOT")
        or launcher_family == "pals"
    )

    pmi_compat_ok = (not launcher_needs_pmi) or has_pmi_lib or bool(present_pmi_env)
    if not pmi_compat_ok and launcher_family == "pals" and pals_runtime_signal:
        pmi_compat_ok = True

    checks.append(
        _check(
            "pmi.libs.compatibility",
            pmi_compat_ok,
            (
                "PMI library/env should be visible for launcher-based runs; "
                f"family={launcher_family}, pals_signal={pals_runtime_signal}"
            ),
        )
    )

    topology_payload: dict[str, object] = {}
    topo_ok = True
    topo_details = "ok"
    try:
        topo = infer_topology(
            requested_nproc=nproc,
            requested_nhosts=nhosts,
            requested_nproc_per_node=nproc_per_node,
            hostfile=resolved,
        )
        topology_payload = {
            "nproc": topo.nproc,
            "nhosts": topo.nhosts,
            "nproc_per_node": topo.nproc_per_node,
            "max_nhosts": topo.max_nhosts,
            "max_nproc_per_node": topo.max_nproc_per_node,
        }
    except Exception as exc:
        topo_ok = False
        topo_details = str(exc)
    checks.append(_check("topology.inference", topo_ok, topo_details))

    ok_count = sum(1 for c in checks if c["ok"])
    fail_count = len(checks) - ok_count
    payload = {
        "xpu": {
            "scheduler_detected": scheduler,
            "hostfile": resolved,
            "distributed": summary,
            "topology": topology_payload,
            "checks": checks,
            "ok_count": ok_count,
            "fail_count": fail_count,
        }
    }
    click.echo(json.dumps(payload, indent=2, sort_keys=True))
