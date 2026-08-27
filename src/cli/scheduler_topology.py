"""Scheduler detection and topology inference for xpu launch."""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import time
from getpass import getuser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass(frozen=True)
class Topology:
    nproc: int
    nhosts: int
    nproc_per_node: int
    max_nhosts: Optional[int]
    max_nproc_per_node: Optional[int]


_QSTAT_MAX_RETRIES = 5
_QSTAT_RETRY_DELAY_S = 2
_PBS_JOBS_CACHE_TTL_S = 30.0
_pbs_jobs_cache: tuple[float, dict[str, list[str]]] | None = None


def _parse_positive_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    m = re.match(r"^(\d+)", text)
    if not m:
        return None
    out = int(m.group(1))
    return out if out > 0 else None


def detect_scheduler(explicit: Optional[str] = None) -> str:
    """Detect active scheduler: pbs, slurm, none, or unknown."""
    if explicit:
        return explicit.strip().lower()

    env_override = os.environ.get("XPU_SCHEDULER") or os.environ.get("EZPZ_SCHEDULER")
    if env_override:
        return env_override.strip().lower()

    if os.environ.get("PBS_JOBID") or os.environ.get("COBALT_JOBID"):
        return "pbs"

    if os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID"):
        return "slurm"

    if shutil.which("qsub"):
        return "pbs"

    if shutil.which("sbatch"):
        return "slurm"

    return "none"


def _read_unique_hosts(path: Path) -> list[str]:
    hosts: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            host = raw.strip()
            if not host or host in seen:
                continue
            seen.add(host)
            hosts.append(host)
    return hosts


def expand_slurm_nodelist(nodelist_str: str) -> list[str]:
    """Expand SLURM bracket nodelist syntax into explicit hostnames."""
    s = nodelist_str.strip()
    if "[" not in s:
        return [s]

    open_idx = s.index("[")
    close_idx = s.rfind("]")
    if close_idx < open_idx:
        return [s]

    prefix = s[:open_idx]
    body = s[open_idx + 1 : close_idx]

    nodes: list[str] = []
    for part in body.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = token.split("-", 1)
            if not lo.isdigit() or not hi.isdigit():
                continue
            width = len(lo)
            for num in range(int(lo), int(hi) + 1):
                nodes.append(f"{prefix}{num:0{width}d}")
        else:
            nodes.append(f"{prefix}{token}")

    return nodes or [s]


def _run_command(cmd: Sequence[str]) -> Optional[str]:
    if not cmd:
        return None
    if shutil.which(cmd[0]) is None:
        return None
    proc = subprocess.run(
        list(cmd),
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def _short_hostname(host: str) -> str:
    return host.split(".", 1)[0]


def _run_qstat_with_retry(*args: str) -> Optional[str]:
    if shutil.which("qstat") is None:
        return None
    last_err = ""
    for attempt in range(_QSTAT_MAX_RETRIES):
        proc = subprocess.run(
            ["qstat", *args],
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
        )
        if proc.returncode == 0:
            return proc.stdout
        last_err = (proc.stderr or proc.stdout).strip()
        transient = (
            "Communication failure" in last_err
            or "cannot connect to server" in last_err
        )
        if transient and attempt + 1 < _QSTAT_MAX_RETRIES:
            time.sleep(_QSTAT_RETRY_DELAY_S)
            continue
        break
    return None


def get_pbs_running_jobs_for_user() -> dict[str, list[str]]:
    global _pbs_jobs_cache
    if _pbs_jobs_cache is not None:
        ts, cached = _pbs_jobs_cache
        if (time.monotonic() - ts) < _PBS_JOBS_CACHE_TTL_S:
            return cached

    output = _run_qstat_with_retry("-fn1wru", getuser())
    if not output:
        return {}

    jobs: dict[str, list[str]] = {}
    for line in output.splitlines():
        if " R " not in line:
            continue
        parts = [p for p in line.split(" ") if p]
        if len(parts) < 2:
            continue
        jobid = parts[0].split(".")[0]
        host_field = parts[-1]
        nodes = [_short_hostname(h.split("/")[0]) for h in host_field.split("+")]
        if nodes:
            jobs[jobid] = nodes

    _pbs_jobs_cache = (time.monotonic(), jobs)
    return jobs


def get_pbs_jobid_of_active_job() -> Optional[str]:
    pbs_jobid = os.environ.get("PBS_JOBID")
    if pbs_jobid:
        short_id = pbs_jobid.split(".")[0]
        pbs_nodefile = os.environ.get("PBS_NODEFILE")
        if pbs_nodefile and Path(pbs_nodefile).is_file():
            try:
                raw_nodes = Path(pbs_nodefile).read_text(encoding="utf-8").split()
            except OSError:
                raw_nodes = []
            if raw_nodes:
                nodes_short = {_short_hostname(n) for n in raw_nodes}
                local = _short_hostname(socket.getfqdn())
                if local in nodes_short:
                    return short_id
            else:
                return short_id
        else:
            return short_id

    jobs = get_pbs_running_jobs_for_user()
    local = _short_hostname(socket.getfqdn())
    for jobid, nodelist in jobs.items():
        if local in {_short_hostname(h) for h in nodelist}:
            return jobid
    return None


def get_pbs_nodefile_from_jobid(jobid: str | int) -> Optional[str]:
    pbs_parent = Path("/var/spool/pbs/aux")
    if not pbs_parent.is_dir():
        return None
    matches = [p for p in pbs_parent.iterdir() if str(jobid) in p.name]
    if len(matches) != 1:
        return None
    nodefile = matches[0]
    if not nodefile.is_file():
        return None
    return nodefile.resolve().as_posix()


def get_pbs_nodefile_of_active_job() -> Optional[str]:
    jobid = get_pbs_jobid_of_active_job()
    if not jobid:
        return None
    return get_pbs_nodefile_from_jobid(jobid)


def get_slurm_running_jobids() -> list[str]:
    """Best-effort running job IDs for current user."""
    output = _run_command(["sacct"])
    if output:
        jobs = {
            row.replace(".", " ").split(" ")[0]
            for row in output.splitlines()
            if " RUNNING " in row
        }
        if jobs:
            return sorted(jobs)

    output = _run_command(["squeue", "-h", "-u", getuser(), "-o", "%A %T"])
    if not output:
        return []
    out: list[str] = []
    for row in output.splitlines():
        parts = row.split()
        if len(parts) >= 2 and parts[1].upper() == "RUNNING":
            out.append(parts[0])
    return sorted(set(out))


def get_slurm_active_jobid() -> Optional[str]:
    jobid = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    if jobid:
        return str(jobid)

    local = _short_hostname(socket.getfqdn())
    jobids = get_slurm_running_jobids()
    for candidate in jobids:
        nodelist = get_nodelist_from_slurm_jobid(candidate)
        if local in {_short_hostname(n) for n in nodelist}:
            return candidate
    return jobids[0] if jobids else None


def get_nodelist_from_slurm_jobid(jobid: str | int) -> list[str]:
    output = _run_command(["scontrol", "show", "job", str(jobid)])
    if not output:
        return []
    for line in output.splitlines():
        m = re.search(r"NodeList=([^\s]+)", line)
        if m and m.group(1) != "(null)":
            return expand_slurm_nodelist(m.group(1))
    return []


def get_slurm_active_nodelist() -> list[str]:
    """Return active SLURM nodelist, preferring env vars over scontrol."""
    env_nodelist = os.environ.get("SLURM_NODELIST")
    if env_nodelist:
        return expand_slurm_nodelist(env_nodelist)

    jobid = get_slurm_active_jobid()
    if not jobid:
        return []
    return get_nodelist_from_slurm_jobid(jobid)


def make_slurm_hostfile(workdir: Path, *, tag: str = "active") -> Optional[Path]:
    hosts = get_slurm_active_nodelist()
    if not hosts:
        return None
    hostfile = workdir / f".xpu_slurm_{tag}.hostfile"
    hostfile.parent.mkdir(parents=True, exist_ok=True)
    hostfile.write_text("\n".join(hosts) + "\n", encoding="utf-8")
    return hostfile


def resolve_hostfile(
    preferred: Optional[str],
    scheduler: str,
    *,
    workdir: Optional[Path] = None,
) -> Optional[str]:
    """Resolve hostfile from explicit arg, env vars, or scheduler context."""
    candidates = [
        preferred,
        os.environ.get("HOSTFILE"),
        os.environ.get("PBS_NODEFILE"),
        os.environ.get("COBALT_NODEFILE"),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate

    if scheduler == "pbs":
        nodefile = get_pbs_nodefile_of_active_job()
        if nodefile and os.path.isfile(nodefile):
            return nodefile

    if scheduler == "slurm":
        made = make_slurm_hostfile(workdir or Path.cwd())
        if made is not None and made.is_file():
            return str(made)

    return None


def get_num_nodes_from_hostfile(hostfile: Optional[str]) -> Optional[int]:
    if hostfile is None:
        return _parse_positive_int(os.environ.get("SLURM_NNODES"))
    path = Path(hostfile)
    if not path.is_file():
        return _parse_positive_int(os.environ.get("SLURM_NNODES"))
    hosts = _read_unique_hosts(path)
    if hosts:
        return len(hosts)
    return _parse_positive_int(os.environ.get("SLURM_NNODES"))


def get_devices_per_node_hint() -> Optional[int]:
    """Infer tasks/devices per node from env hints without importing torch."""
    for key in (
        "NGPU_PER_HOST",
        "LOCAL_WORLD_SIZE",
        "PMI_LOCAL_SIZE",
        "SLURM_NTASKS_PER_NODE",
    ):
        parsed = _parse_positive_int(os.environ.get(key))
        if parsed is not None:
            return parsed
    return None


def infer_topology(
    *,
    requested_nproc: Optional[int],
    requested_nhosts: Optional[int],
    requested_nproc_per_node: Optional[int],
    hostfile: Optional[str],
) -> Topology:
    """Infer (nproc, nhosts, nproc_per_node) with scheduler-aware constraints."""
    max_nhosts = get_num_nodes_from_hostfile(hostfile)
    max_ppn = get_devices_per_node_hint()

    nhosts = requested_nhosts
    nproc = requested_nproc
    nproc_per_node = requested_nproc_per_node

    if nhosts is None:
        if nproc is not None and nproc_per_node is not None:
            if nproc_per_node <= 0 or nproc % nproc_per_node != 0:
                raise SystemExit(
                    "--nproc must be divisible by --nproc_per_node "
                    f"(nproc={nproc}, nproc_per_node={nproc_per_node})"
                )
            nhosts = nproc // nproc_per_node
        elif max_nhosts is not None:
            nhosts = max_nhosts
        else:
            nhosts = 1

    if nproc is None:
        if nproc_per_node is None:
            nproc_per_node = max_ppn or 1
        nproc = nhosts * nproc_per_node
    else:
        if nproc_per_node is None:
            if nhosts > 0 and nproc % nhosts == 0:
                nproc_per_node = nproc // nhosts
            else:
                raise SystemExit(
                    "--nproc must be divisible by --nhosts when --nproc_per_node is omitted "
                    f"(nproc={nproc}, nhosts={nhosts})"
                )
        else:
            expected = nhosts * nproc_per_node
            if expected != nproc:
                raise SystemExit(
                    "inconsistent topology: --nproc must equal --nhosts * --nproc_per_node "
                    f"(nproc={nproc}, nhosts={nhosts}, nproc_per_node={nproc_per_node})"
                )

    if nproc <= 0 or nhosts <= 0 or nproc_per_node <= 0:
        raise SystemExit(
            "invalid topology: expected positive nproc/nhosts/nproc_per_node "
            f"(nproc={nproc}, nhosts={nhosts}, nproc_per_node={nproc_per_node})"
        )

    if max_nhosts is not None and nhosts > max_nhosts:
        raise SystemExit(
            f"requested nhosts exceeds available hosts ({nhosts} > {max_nhosts})"
        )

    if max_ppn is not None and nproc_per_node > max_ppn:
        raise SystemExit(
            "requested nproc_per_node exceeds per-node hint "
            f"({nproc_per_node} > {max_ppn}). "
            "Set --nproc_per_node explicitly only when this is intentional."
        )

    return Topology(
        nproc=nproc,
        nhosts=nhosts,
        nproc_per_node=nproc_per_node,
        max_nhosts=max_nhosts,
        max_nproc_per_node=max_ppn,
    )


def choose_launcher_binary(scheduler: str) -> Optional[str]:
    """Select a launcher binary with scheduler-preferred priority."""
    candidates: list[str]
    if scheduler == "slurm":
        candidates = ["srun", "mpiexec", "mpirun"]
    elif scheduler == "pbs":
        candidates = ["mpiexec", "mpirun", "srun"]
    else:
        candidates = ["mpiexec", "mpirun", "srun"]

    for candidate in candidates:
        if shutil.which(candidate):
            return candidate
    return None
