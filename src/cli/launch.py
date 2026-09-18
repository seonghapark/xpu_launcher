"""Standalone launcher backend for `xpu launch`."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from cli import accelerators
from cli.compat import get_machine_name
from cli.failover_models import NodeAllocation
from cli.scheduler_topology import (
    Topology,
    choose_launcher_binary,
    detect_scheduler,
    infer_topology,
    resolve_hostfile,
)

_WATCHDOG_EXIT_CODE = 124
_WATCHDOG_KILL_GRACE_S = 10.0
_RETRY_BACKOFF_CAP_S = 60.0
_AUTO_RETRY_DEFAULT_TIMEOUT_S = 1800
_WALLTIME_RC = 143
_MAX_CONSECUTIVE_UNATTRIBUTED = 3
_FAILOVER_PROFILES = ("auto", "aurora", "slurm", "generic")
_SCHEDULERS = ("auto", "pbs", "slurm", "none")

_ANSI_RX = re.compile(r"\x1b\[[0-9;]*m")
_INNOCENT_RANK_CASCADE_RX = re.compile(
    r"rank \d+ died from signal (?:11|15)",
    flags=re.IGNORECASE,
)
_CRASH_PATTERNS_RX = re.compile(
    r"RuntimeError: \[.*gloo.*\] Connection closed by peer"
    r"|RuntimeError: \[.*gloo.*\] Timed out waiting"
    r"|OutOfMemoryError"
    r"|MemoryError: std::bad_alloc"
    r"|RuntimeError: could not create a memory"
    r"|UR_RESULT_ERROR_OUT_OF_RESOURCES"
    r"|died from signal"
    r"|rank \d+ exited with code [1-9][0-9]*"
    r"|EOFError: No data left in file"
    r"|unrecognized argument"
    r"|unrecognized option"
    r"|srun: error: \S+: tasks? [\d,-]+: Killed"
    r"|killed by signal"
    r"|connection reset by peer"
    r"|connection closed by peer",
    flags=re.IGNORECASE,
)
_WALLTIME_PATTERNS_RX = re.compile(
    r"walltime|time limit|due to time limit|job wall clock limit",
    flags=re.IGNORECASE,
)
_UNATTRIBUTED_IO_RX = re.compile(
    r"No space left on device"
    r"|\[Errno 28\]"
    r"|Disk quota exceeded"
    r"|\[Errno 122\]",
    flags=re.IGNORECASE,
)
_PROGRESS_MARKER_RX = re.compile(
    r"\b(?:iter|step|epoch|batch|idx)\s*[=:]\s*\d+",
    flags=re.MULTILINE,
)
_IPV4_RX = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

_CRASH_PATTERNS_BY_PROFILE: dict[str, re.Pattern[str]] = {
    "generic": _CRASH_PATTERNS_RX,
    "aurora": re.compile(
        _CRASH_PATTERNS_RX.pattern
        + r"|pals|shepherd|xccl|ze_affinity_mask|oneccl",
        flags=re.IGNORECASE,
    ),
    "slurm": re.compile(
        _CRASH_PATTERNS_RX.pattern
        + r"|slurmstepd|srun: error|task \d+: Exited",
        flags=re.IGNORECASE,
    ),
}

_WALLTIME_PATTERNS_BY_PROFILE: dict[str, re.Pattern[str]] = {
    "generic": _WALLTIME_PATTERNS_RX,
    "aurora": re.compile(
        _WALLTIME_PATTERNS_RX.pattern + r"|pbs job killed: walltime",
        flags=re.IGNORECASE,
    ),
    "slurm": re.compile(
        _WALLTIME_PATTERNS_RX.pattern + r"|DUE TO TIME LIMIT",
        flags=re.IGNORECASE,
    ),
}


class TerminationReason(Enum):
    SUCCESS = "success"
    WALLTIME = "walltime"
    BAD_NODE_KNOWN = "bad_node_known"
    BAD_NODE_BLIND = "bad_node_blind"
    RETRYABLE_UNATTRIBUTED = "retryable_unattributed"
    STUCK_PRE_TRAINING = "stuck_pre_training"
    EXHAUSTED = "exhausted"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class ClassificationResult:
    reason: TerminationReason
    has_progress: bool
    has_unattributed_io: bool = False


def _debug_log(enabled: bool, message: str) -> None:
    if enabled:
        print(f"xpu auto-retry debug: {message}", file=sys.stderr)


_active_accelerator: Optional[str] = None
_combined_crash_rx_cache: dict[tuple[str, str], re.Pattern[str]] = {}


def set_active_accelerator(name: Optional[str]) -> None:
    """Set the accelerator whose crash signatures augment the classifier."""
    global _active_accelerator
    _active_accelerator = None if name in (None, "none", "auto") else name


def get_active_accelerator() -> Optional[str]:
    return _active_accelerator


def _get_crash_rx(profile: str) -> re.Pattern[str]:
    base = _CRASH_PATTERNS_BY_PROFILE.get(
        profile, _CRASH_PATTERNS_BY_PROFILE["generic"]
    )
    accel = _active_accelerator
    if accel is None:
        return base
    backend = accelerators.get_accelerator(accel)
    if backend is None:
        return base
    key = (profile, accel)
    cached = _combined_crash_rx_cache.get(key)
    if cached is None:
        cached = re.compile(
            base.pattern + "|" + backend.crash_patterns(),
            flags=re.IGNORECASE,
        )
        _combined_crash_rx_cache[key] = cached
    return cached


def _get_walltime_rx(profile: str) -> re.Pattern[str]:
    return _WALLTIME_PATTERNS_BY_PROFILE.get(
        profile, _WALLTIME_PATTERNS_BY_PROFILE["generic"]
    )


def _matching_lines(
    log_text: str,
    rx: re.Pattern[str],
    *,
    skip_innocent: bool = False,
    limit: int = 3,
) -> list[str]:
    out: list[str] = []
    for raw_line in log_text.splitlines():
        line = _strip_ansi(raw_line)
        if skip_innocent and _INNOCENT_RANK_CASCADE_RX.search(line):
            continue
        if rx.search(line):
            out.append(line.strip())
            if len(out) >= limit:
                break
    return out


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {value!r}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def build_launch_parser(*, prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Launch a command on the current allocation with familiar launcher flags.\n\n"
            "Use '--' to separate launcher flags and the command when needed:\n"
            "  xpu launch -n 8 -ppn 4 -x FOO=bar -- python train.py\n\n"
            "Launcher prefix resolution order:\n"
            "1) --launcher\n"
            "2) $DIST_LAUNCH\n"
            "3) $XPU_LAUNCHER\n"
            "4) $LAUNCH_CMD\n"
            "5) $LAUNCH\n"
            "6) auto-detect (mpiexec/mpirun/srun) when size flags are provided\n"
            "If none apply, runs the command directly."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--print-source",
        action="store_true",
        help="Print the launcher source file path and exit.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        nargs="+",
        default=None,
        help="Deprecated compatibility flag. Ignored.",
    )
    parser.add_argument(
        "-n",
        "-np",
        "--n",
        "--np",
        "--nproc",
        "--world_size",
        "--nprocs",
        type=int,
        dest="nproc",
        default=-1,
        help="Number of processes.",
    )
    parser.add_argument(
        "-ppn",
        "--ppn",
        "--nproc_per_node",
        type=int,
        default=-1,
        dest="nproc_per_node",
        help="Processes per node.",
    )
    parser.add_argument(
        "-nh",
        "--nh",
        "--nhost",
        "--nnode",
        "--nnodes",
        "--nhosts",
        type=int,
        default=-1,
        dest="nhosts",
        help="Number of hosts.",
    )
    parser.add_argument(
        "--hostfile",
        type=str,
        default=None,
        dest="hostfile",
        help="Hostfile path passed through to the launcher when supported.",
    )
    parser.add_argument(
        "--host-ip-map",
        type=str,
        default=None,
        dest="host_ip_map",
        help=(
            "JSON file for deterministic IP->host attribution in auto-retry "
            "(avoids DNS dependence)."
        ),
    )
    parser.add_argument(
        "--cpu-bind",
        type=str,
        default=None,
        dest="cpu_bind",
        help="CPU bind setting passed through to the launcher when supported.",
    )
    parser.add_argument(
        "--timeout",
        type=_non_negative_int,
        default=None,
        help="Kill the process if no stdout/stderr appears for N seconds (0 disables watchdog).",
    )
    parser.add_argument(
        "--retries",
        type=_non_negative_int,
        default=0,
        help="Retry count for non-zero exits (exponential backoff: 5s, 10s, 20s, ...).",
    )
    parser.add_argument(
        "--auto-retry",
        action="store_true",
        dest="auto_retry",
        help=(
            "Retry on non-zero exits and perform host failover when bad nodes "
            "are detected from launcher logs."
        ),
    )
    parser.add_argument(
        "--spare-nodes",
        default=None,
        dest="spare_nodes",
        help="Spare host count for --auto-retry (integer or 'auto').",
    )
    parser.add_argument(
        "--max-failover-retries",
        type=_non_negative_int,
        default=None,
        dest="max_failover_retries",
        help="Cap retries used by --auto-retry compatibility mode.",
    )
    parser.add_argument(
        "--failover-profile",
        type=str,
        default="auto",
        choices=_FAILOVER_PROFILES,
        help="Pattern profile used for bad-node classification.",
    )
    parser.add_argument(
        "--failover-debug",
        action="store_true",
        default=False,
        help="Print auto-retry classifier matches and host attribution details.",
    )
    parser.add_argument(
        "--launcher",
        default=None,
        help="Explicit launcher prefix, e.g. 'mpiexec -n 8 --ppn 4'.",
    )
    parser.add_argument(
        "--scheduler",
        default="auto",
        choices=_SCHEDULERS,
        help=(
            "Scheduler hint for launch synthesis. 'auto' detects from env; "
            "'pbs' and 'slurm' force scheduler-specific launch command assembly."
        ),
    )
    parser.add_argument(
        "--accelerator",
        default="auto",
        choices=accelerators.ACCELERATOR_CHOICES,
        help=(
            "Accelerator backend (xpu=Intel, cuda=NVIDIA, rocm=AMD). 'auto' "
            "probes the machine; backend crash signatures augment --auto-retry "
            "classification and XPU_LAUNCH_ACCELERATOR/XPU_LAUNCH_DIST_BACKEND "
            "are exported to launched processes."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved command and exit without running.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to execute. Prefix with '--' when the command starts with flags.",
    )
    return parser


def _split_launch_and_command(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        idx = list(argv).index("--")
        return list(argv[:idx]), list(argv[idx + 1 :])
    return list(argv), []


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_launch_parser(prog="xpu launch")
    raw_argv = [] if argv is None else list(argv)
    launch_argv, command_from_sep = _split_launch_and_command(raw_argv)

    args, unknown = parser.parse_known_args(launch_argv)
    if command_from_sep:
        args.command = command_from_sep
        # Unknown flags before '--' are launcher passthrough flags.
        args.launcher_args = unknown
    else:
        # Without '--', argparse already captured the trailing command.
        args.launcher_args = []

    if args.auto_retry and args.retries:
        raise SystemExit(
            "--auto-retry is mutually exclusive with --retries. Pick one."
        )
    if args.nproc != -1 and args.nproc <= 0:
        raise SystemExit(f"--nproc must be > 0, got {args.nproc}")
    if args.nproc_per_node != -1 and args.nproc_per_node <= 0:
        raise SystemExit(
            f"--nproc_per_node must be > 0, got {args.nproc_per_node}"
        )
    if args.nhosts != -1 and args.nhosts <= 0:
        raise SystemExit(f"--nhosts must be > 0, got {args.nhosts}")

    return args


def _resolve_launcher_prefix(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit.strip() or None
    for env_name in ("DIST_LAUNCH", "XPU_LAUNCHER", "LAUNCH_CMD", "LAUNCH"):
        value = os.environ.get(env_name)
        if value and value.strip():
            return value.strip()
    return None


def _extract_launcher_name(tokens: Sequence[str]) -> str:
    if not tokens:
        return ""
    return Path(tokens[0]).name


def _detect_failover_profile(
    args: argparse.Namespace, launcher_tokens: Sequence[str]
) -> str:
    if args.failover_profile != "auto":
        return args.failover_profile

    launcher_name = _extract_launcher_name(launcher_tokens)
    if launcher_name == "srun":
        return "slurm"

    scheduler_hints = (
        os.environ.get("SLURM_JOB_ID"),
        os.environ.get("SLURM_CLUSTER_NAME"),
    )
    if any(scheduler_hints):
        return "slurm"

    aurora_hints = (
        os.environ.get("PBS_JOBID"),
        os.environ.get("COBALT_JOBID"),
        os.environ.get("ZE_AFFINITY_MASK"),
        os.environ.get("CCL_ATL_TRANSPORT"),
    )
    if any(aurora_hints):
        return "aurora"

    machine = get_machine_name()
    if machine in {"aurora", "sunspot", "polaris", "sophia"}:
        return "aurora"

    return "generic"


def _remove_option(tokens: list[str], option: str) -> list[str]:
    """Remove option variants like '--hostfile val' and '--hostfile=val'."""
    result: list[str] = []
    skip_next = False
    for tok in tokens:
        if skip_next:
            skip_next = False
            continue
        if tok == option:
            skip_next = True
            continue
        if tok.startswith(f"{option}="):
            continue
        result.append(tok)
    return result


def _resolved_scheduler(args: argparse.Namespace) -> str:
    explicit_scheduler = None if args.scheduler == "auto" else args.scheduler
    return detect_scheduler(explicit=explicit_scheduler)


def _resolve_requested_topology(
    args: argparse.Namespace,
    *,
    hostfile_override: Optional[str] = None,
    active_hosts_override: Optional[Sequence[str]] = None,
) -> Topology:
    requested_nproc = args.nproc if args.nproc > 0 else None
    requested_nproc_per_node = (
        args.nproc_per_node if args.nproc_per_node > 0 else None
    )

    if active_hosts_override is not None:
        requested_nhosts: Optional[int] = len(active_hosts_override)
    else:
        requested_nhosts = args.nhosts if args.nhosts > 0 else None

    return infer_topology(
        requested_nproc=requested_nproc,
        requested_nhosts=requested_nhosts,
        requested_nproc_per_node=requested_nproc_per_node,
        hostfile=hostfile_override,
    )


def _build_detected_launcher_tokens(
    args: argparse.Namespace,
    *,
    scheduler: str,
    hostfile_override: Optional[str] = None,
    active_hosts_override: Optional[Sequence[str]] = None,
    excluded_hosts: Optional[Sequence[str]] = None,
) -> list[str]:
    launcher = choose_launcher_binary(scheduler)
    if launcher is None:
        return []

    topology = _resolve_requested_topology(
        args,
        hostfile_override=hostfile_override,
        active_hosts_override=active_hosts_override,
    )
    tokens: list[str] = [launcher]

    if launcher in ("mpiexec", "mpirun"):
        tokens.extend(["-n", str(topology.nproc)])
        if topology.nproc_per_node > 0:
            ppn_flag = "--ppn" if launcher == "mpiexec" else "-N"
            tokens.extend([ppn_flag, str(topology.nproc_per_node)])
        hostfile_value = hostfile_override or args.hostfile
        if hostfile_value:
            tokens.extend(["--hostfile", hostfile_value])
        if args.cpu_bind:
            tokens.append(f"--cpu-bind={args.cpu_bind}")
        tokens.extend(args.launcher_args)
        return tokens

    # srun
    tokens.extend(["-u", "--verbose"])
    tokens.extend(["-N", str(topology.nhosts)])
    tokens.extend(["-n", str(topology.nproc)])
    if topology.nproc_per_node > 0:
        tokens.extend(["--ntasks-per-node", str(topology.nproc_per_node)])
    if active_hosts_override:
        tokens.extend(["--nodelist", ",".join(active_hosts_override)])
    else:
        hostfile_value = hostfile_override or args.hostfile
        if hostfile_value and os.path.isfile(hostfile_value):
            tokens.append(f"--nodelist={Path(hostfile_value).resolve()}")
    if excluded_hosts:
        uniq_excluded = sorted({h for h in excluded_hosts if h})
        if uniq_excluded:
            tokens.append(f"--exclude={','.join(uniq_excluded)}")
    if args.cpu_bind:
        tokens.append(f"--cpu-bind={args.cpu_bind}")
    tokens.extend(args.launcher_args)
    return tokens


def _build_launcher_tokens(
    args: argparse.Namespace,
    *,
    scheduler: str,
    hostfile_override: Optional[str] = None,
    active_hosts_override: Optional[Sequence[str]] = None,
    excluded_hosts: Optional[Sequence[str]] = None,
) -> list[str]:
    explicit = _resolve_launcher_prefix(args.launcher)
    if explicit:
        explicit_tokens = [*shlex.split(explicit), *args.launcher_args]
        launcher_name = _extract_launcher_name(explicit_tokens)
        if launcher_name in ("mpiexec", "mpirun") and hostfile_override:
            explicit_tokens = _remove_option(explicit_tokens, "--hostfile")
            explicit_tokens.extend(["--hostfile", hostfile_override])
        elif launcher_name == "srun":
            if active_hosts_override:
                explicit_tokens = _remove_option(explicit_tokens, "--nodelist")
                explicit_tokens.extend(["--nodelist", ",".join(active_hosts_override)])
            elif hostfile_override and os.path.isfile(hostfile_override):
                explicit_tokens = _remove_option(explicit_tokens, "--nodelist")
                explicit_tokens.append(f"--nodelist={Path(hostfile_override).resolve()}")
            if excluded_hosts:
                uniq_excluded = sorted({h for h in excluded_hosts if h})
                if uniq_excluded:
                    explicit_tokens = _remove_option(explicit_tokens, "--exclude")
                    explicit_tokens.append(f"--exclude={','.join(uniq_excluded)}")
        return explicit_tokens

    if any(
        (
            scheduler in ("pbs", "slurm"),
            args.nproc > 0,
            args.nproc_per_node > 0,
            args.nhosts > 0,
            args.hostfile,
            args.cpu_bind,
            args.launcher_args,
            hostfile_override,
            active_hosts_override,
        )
    ):
        return _build_detected_launcher_tokens(
            args,
            scheduler=scheduler,
            hostfile_override=hostfile_override,
            active_hosts_override=active_hosts_override,
            excluded_hosts=excluded_hosts,
        )

    return []


def _read_hostfile(path: str) -> list[str]:
    hosts: list[str] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            host = raw.strip()
            if not host or host in seen:
                continue
            seen.add(host)
            hosts.append(host)
    return hosts


def _write_hostfile(path: str, hosts: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for host in hosts:
            fh.write(f"{host}\n")


def _resolve_hostfile_path(args: argparse.Namespace) -> Optional[str]:
    scheduler = _resolved_scheduler(args)
    workdir = Path.cwd()
    return resolve_hostfile(args.hostfile, scheduler, workdir=workdir)


def _derive_active_hosts_requested(
    args: argparse.Namespace, total_hosts: int
) -> int:
    if args.nhosts > 0:
        return args.nhosts
    if args.nproc > 0 and args.nproc_per_node > 0:
        return (args.nproc + args.nproc_per_node - 1) // args.nproc_per_node
    return total_hosts


def _derive_spare_nodes(
    spare_nodes: Optional[str], total_hosts: int, active_hosts: int
) -> int:
    available = max(0, total_hosts - active_hosts)
    if spare_nodes is None or spare_nodes == "auto":
        return available
    try:
        parsed = int(spare_nodes)
    except ValueError as exc:
        raise SystemExit(
            f"--spare-nodes must be an integer or 'auto', got {spare_nodes!r}"
        ) from exc
    if parsed < 0:
        raise SystemExit(f"--spare-nodes must be >= 0, got {parsed}")
    return min(parsed, available)


def _build_host_alias_map(
    active_hosts: Sequence[str],
    *,
    debug: bool = False,
    manual_ip_map: Optional[dict[str, str]] = None,
    enable_dns: bool = True,
) -> dict[str, str]:
    alias_map: dict[str, str] = {}

    for host in active_hosts:
        short = host.split(".", 1)[0]
        alias_map[host] = host
        alias_map[short] = host

        if enable_dns:
            try:
                _name, _aliases, addrs = socket.gethostbyname_ex(host)
            except OSError as exc:
                _debug_log(debug, f"dns lookup failed for {host}: {exc}")
                continue

            for addr in addrs:
                if addr not in alias_map:
                    alias_map[addr] = host

    if manual_ip_map:
        for ip, host in manual_ip_map.items():
            if host in active_hosts and _IPV4_RX.fullmatch(ip):
                alias_map[ip] = host
        _debug_log(debug, f"manual ip map entries loaded: {len(manual_ip_map)}")

    if debug:
        ip_entries = {k: v for k, v in alias_map.items() if _IPV4_RX.fullmatch(k)}
        if ip_entries:
            _debug_log(True, f"ip alias map: {ip_entries}")

    return alias_map


def _extract_bad_hosts(
    log_text: str,
    active_hosts: Sequence[str],
    profile: str,
    *,
    debug: bool = False,
    manual_ip_map: Optional[dict[str, str]] = None,
) -> list[str]:
    bad_hosts: list[str] = []
    alias_map = _build_host_alias_map(
        active_hosts,
        debug=debug,
        manual_ip_map=manual_ip_map,
        enable_dns=manual_ip_map is None,
    )

    crash_rx = _get_crash_rx(profile)
    evidence: dict[str, str] = {}
    reverse_dns_cache: dict[str, Optional[str]] = {}

    def _resolve_ip_to_active_host(ip: str) -> Optional[str]:
        cached = reverse_dns_cache.get(ip)
        if cached is not None or ip in reverse_dns_cache:
            return cached
        try:
            hostname, _aliases, _addrs = socket.gethostbyaddr(ip)
            short = hostname.split(".", 1)[0]
            resolved = alias_map.get(hostname) or alias_map.get(short)
            reverse_dns_cache[ip] = resolved
            if debug:
                _debug_log(
                    True,
                    f"reverse dns {ip} -> {hostname}; mapped={resolved}",
                )
            return resolved
        except OSError as exc:
            reverse_dns_cache[ip] = None
            _debug_log(debug, f"reverse dns failed for {ip}: {exc}")
            return None

    for raw_line in log_text.splitlines():
        line = _ANSI_RX.sub("", raw_line)
        if _INNOCENT_RANK_CASCADE_RX.search(line):
            continue
        if crash_rx.search(line) is None:
            continue

        for alias, host in alias_map.items():
            if re.search(
                rf"(?<![A-Za-z0-9_.-]){re.escape(alias)}(?![A-Za-z0-9_.-])",
                line,
            ):
                if host not in bad_hosts:
                    bad_hosts.append(host)
                    evidence[host] = line

        if ":" in line:
            prefix = line.split(":", 1)[0].strip()
            mapped = alias_map.get(prefix)
            if mapped and mapped not in bad_hosts:
                bad_hosts.append(mapped)
                evidence[mapped] = line

        for ip in _IPV4_RX.findall(line):
            mapped = alias_map.get(ip) or _resolve_ip_to_active_host(ip)
            if mapped and mapped not in bad_hosts:
                bad_hosts.append(mapped)
                evidence[mapped] = line

    if debug and bad_hosts:
        for host in bad_hosts:
            src = evidence.get(host, "")
            _debug_log(True, f"bad host matched: {host} | evidence: {src}")

    return bad_hosts


def _load_host_ip_map(
    path: str,
    known_hosts: Sequence[str],
    *,
    debug: bool = False,
) -> dict[str, str]:
    host_alias_map: dict[str, str] = {}
    for host in known_hosts:
        host_alias_map[host] = host
        host_alias_map[host.split(".", 1)[0]] = host

    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except OSError as exc:
        raise SystemExit(f"failed to read --host-ip-map {path!r}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in --host-ip-map {path!r}: {exc}") from exc

    if not isinstance(payload, dict):
        raise SystemExit("--host-ip-map JSON must be an object")

    result: dict[str, str] = {}

    def _add_mapping(ip: str, host_alias: str) -> None:
        if not _IPV4_RX.fullmatch(ip):
            return
        canonical_host = host_alias_map.get(host_alias)
        if canonical_host is None:
            _debug_log(debug, f"host-ip-map ignored (unknown host alias): {host_alias}")
            return
        result[ip] = canonical_host

    for key, value in payload.items():
        if isinstance(value, str):
            if _IPV4_RX.fullmatch(key):
                _add_mapping(key, value)
            elif _IPV4_RX.fullmatch(value):
                _add_mapping(value, key)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and _IPV4_RX.fullmatch(item):
                    _add_mapping(item, key)
        elif isinstance(value, dict):
            host_alias = value.get("host")
            ips = value.get("ips")
            if isinstance(host_alias, str) and isinstance(ips, list):
                for item in ips:
                    if isinstance(item, str) and _IPV4_RX.fullmatch(item):
                        _add_mapping(item, host_alias)

    if not result:
        _debug_log(debug, "host-ip-map loaded but no usable mappings found")
    else:
        _debug_log(debug, f"host-ip-map usable mappings: {len(result)}")

    return result


def _has_crash_patterns(log_text: str, profile: str) -> bool:
    crash_rx = _get_crash_rx(profile)
    for raw_line in log_text.splitlines():
        line = _strip_ansi(raw_line)
        if _INNOCENT_RANK_CASCADE_RX.search(line):
            continue
        if crash_rx.search(line):
            return True
    return False


def _looks_like_walltime(log_text: str, profile: str) -> bool:
    walltime_rx = _get_walltime_rx(profile)
    return walltime_rx.search(_strip_ansi(log_text)) is not None


def _has_unattributed_io_failure(log_text: str) -> bool:
    return _UNATTRIBUTED_IO_RX.search(_strip_ansi(log_text)) is not None


def _strip_ansi(text: str) -> str:
    return _ANSI_RX.sub("", text)


def _has_progress_markers(log_text: str) -> bool:
    return _PROGRESS_MARKER_RX.search(_strip_ansi(log_text)) is not None


def _extract_inner_rc(log_text: str) -> Optional[int]:
    stripped = _strip_ansi(log_text)
    marker = "Execution finished with "
    idx = stripped.rfind(marker)
    if idx < 0:
        return None
    tail = stripped[idx + len(marker) :].split(None, 1)[0]
    tail = tail.rstrip(".,;:")
    try:
        return int(tail)
    except ValueError:
        return None


def _classify_attempt(
    shell_rc: int,
    log_text: str,
    scraped_bad_nodes: Sequence[str],
    *,
    profile: str,
    prior_attempt_had_progress: Optional[bool],
    has_spares: bool,
    consecutive_unattributed: int,
) -> ClassificationResult:
    inner_rc = _extract_inner_rc(log_text)
    crash = _has_crash_patterns(log_text, profile)
    has_progress = _has_progress_markers(log_text)
    unattributed_io = _has_unattributed_io_failure(log_text)

    effective_rc = shell_rc
    if shell_rc == 0 and inner_rc is not None and inner_rc != 0:
        effective_rc = inner_rc
    elif shell_rc == 0 and crash:
        effective_rc = 1

    def _result(reason: TerminationReason) -> ClassificationResult:
        return ClassificationResult(
            reason=reason,
            has_progress=has_progress,
            has_unattributed_io=unattributed_io,
        )

    if effective_rc == 0:
        return _result(TerminationReason.SUCCESS)

    if effective_rc == _WALLTIME_RC and not crash:
        return _result(TerminationReason.WALLTIME)

    if (
        unattributed_io
        and consecutive_unattributed < _MAX_CONSECUTIVE_UNATTRIBUTED
    ):
        return _result(TerminationReason.RETRYABLE_UNATTRIBUTED)

    if (
        prior_attempt_had_progress is False
        and not has_progress
        and not scraped_bad_nodes
    ):
        return _result(TerminationReason.STUCK_PRE_TRAINING)

    if effective_rc == _WATCHDOG_EXIT_CODE:
        if not has_spares:
            return _result(TerminationReason.EXHAUSTED)
        return _result(TerminationReason.BAD_NODE_BLIND)

    if scraped_bad_nodes:
        if not has_spares:
            return _result(TerminationReason.EXHAUSTED)
        return _result(TerminationReason.BAD_NODE_KNOWN)

    if not has_spares:
        return _result(TerminationReason.EXHAUSTED)
    return _result(TerminationReason.BAD_NODE_BLIND)


def _terminate_process(process: subprocess.Popen) -> None:
    """SIGTERM then SIGKILL so reader threads hit EOF before shutdown."""
    with contextlib.suppress(Exception):
        process.terminate()
        try:
            process.wait(timeout=_WATCHDOG_KILL_GRACE_S)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _run_with_watchdog_capture(
    cmd: Sequence[str], idle_timeout_s: int
) -> tuple[int, str]:
    if idle_timeout_s <= 0:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        output: list[str] = []
        assert process.stdout is not None
        try:
            for line in process.stdout:
                output.append(line)
                sys.stdout.write(line)
                sys.stdout.flush()
            rc = process.wait()
        except KeyboardInterrupt:
            _terminate_process(process)
            raise
        return rc, "".join(output)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
    output: list[str] = []
    last_activity = time.monotonic()
    activity_lock = threading.Lock()
    reader_done = threading.Event()

    def _drain_output() -> None:
        nonlocal last_activity
        assert process.stdout is not None
        try:
            for line in process.stdout:
                output.append(line)
                sys.stdout.write(line)
                sys.stdout.flush()
                with activity_lock:
                    last_activity = time.monotonic()
        finally:
            reader_done.set()

    threading.Thread(target=_drain_output, daemon=True).start()
    try:
        while True:
            rc = process.poll()
            if rc is not None:
                return rc, "".join(output)

            with activity_lock:
                idle_for = time.monotonic() - last_activity

            if idle_for >= idle_timeout_s:
                print(
                    (
                        f"xpu watchdog: no output for {idle_for:.1f}s "
                        f"(timeout={idle_timeout_s}s). Sending SIGTERM to PID {process.pid}."
                    ),
                    file=sys.stderr,
                )
                process.terminate()
                try:
                    process.wait(timeout=_WATCHDOG_KILL_GRACE_S)
                except subprocess.TimeoutExpired:
                    print(
                        (
                            f"xpu watchdog: PID {process.pid} still alive after "
                            f"{int(_WATCHDOG_KILL_GRACE_S)}s. Sending SIGKILL."
                        ),
                        file=sys.stderr,
                    )
                    process.kill()
                    process.wait()
                return _WATCHDOG_EXIT_CODE, "".join(output)

            sleep_for = min(1.0, max(0.1, idle_timeout_s - idle_for))
            time.sleep(sleep_for)
    except KeyboardInterrupt:
        # Kill the child so the reader thread reaches EOF; otherwise it holds
        # the stdout buffer lock during interpreter shutdown (fatal error).
        _terminate_process(process)
        raise
    finally:
        reader_done.wait(timeout=2.0)


def _run_with_auto_retry(
    args: argparse.Namespace,
    command: Sequence[str],
    timeout_s: int,
) -> int:
    hostfile_path = _resolve_hostfile_path(args)
    if hostfile_path is None:
        raise SystemExit(
            "--auto-retry requires a hostfile. Pass --hostfile or set HOSTFILE/PBS_NODEFILE."
        )

    all_hosts = _read_hostfile(hostfile_path)
    if not all_hosts:
        raise SystemExit(f"Hostfile has no hosts: {hostfile_path}")

    requested_active = _derive_active_hosts_requested(args, len(all_hosts))
    if requested_active <= 0:
        raise SystemExit("--auto-retry requires positive active host count")
    if requested_active > len(all_hosts):
        raise SystemExit(
            "requested active hosts exceed hostfile size: "
            f"active={requested_active}, total={len(all_hosts)}"
        )

    spare_count = _derive_spare_nodes(
        args.spare_nodes, len(all_hosts), requested_active
    )
    active_plus_spare_hosts = list(
        all_hosts[: requested_active + spare_count]
    )

    manual_ip_map: Optional[dict[str, str]] = None
    if args.host_ip_map:
        manual_ip_map = _load_host_ip_map(
            args.host_ip_map,
            all_hosts,
            debug=args.failover_debug,
        )

    active_hostfile = Path(f"{hostfile_path}.xpu_active")
    bad_nodes_path = Path(f"{hostfile_path}.xpu_bad_nodes.txt")
    log_dir = Path(f"{hostfile_path}.xpu_autoretry_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    allocation = NodeAllocation.from_full_nodelist(
        nodelist=active_plus_spare_hosts,
        nproc_active_hosts=requested_active,
        hostfile_path=active_hostfile,
        bad_nodes_path=bad_nodes_path,
    )

    attempt = 1
    last_rc = 1
    consecutive_unattributed = 0
    prior_attempt_had_progress: Optional[bool] = None
    excluded_hosts: set[str] = set()
    scheduler = _resolved_scheduler(args)
    try:
        while True:
            if (
                args.max_failover_retries is not None
                and attempt > (args.max_failover_retries + 1)
            ):
                print(
                    (
                        "[auto-retry] FAILOVER STOP: max_failover_retries="
                        f"{args.max_failover_retries} exhausted (rc={last_rc})"
                    ),
                    file=sys.stderr,
                )
                return last_rc

            launcher_tokens = _build_launcher_tokens(
                args,
                scheduler=scheduler,
                hostfile_override=str(active_hostfile),
                active_hosts_override=allocation.active,
                excluded_hosts=sorted(excluded_hosts),
            )
            failover_profile = _detect_failover_profile(args, launcher_tokens)
            # Rank 0 lands on the first active host; PBS_NODEFILE order can
            # differ after spare/swap rotation, so pin the rendezvous address.
            if allocation.active:
                os.environ["MASTER_ADDR"] = allocation.active[0]
            # PALS exposes rank/local-rank env but no world-size variable;
            # the launcher knows the exact rank count, so publish it.
            try:
                topo = _resolve_requested_topology(
                    args,
                    hostfile_override=str(active_hostfile),
                    active_hosts_override=allocation.active,
                )
                os.environ["WORLD_SIZE"] = str(topo.nproc)
            except Exception:
                pass
            _debug_log(
                args.failover_debug,
                (
                    f"profile={failover_profile}, launcher={' '.join(launcher_tokens) if launcher_tokens else '<none>'}"
                ),
            )
            full_cmd = [*launcher_tokens, *command]

            if args.dry_run:
                print(shlex.join(full_cmd))
                print(
                    (
                        "xpu auto-retry layout: "
                        f"profile={failover_profile}, "
                        f"active={allocation.active}, spares={list(allocation.spare)}"
                    ),
                    file=sys.stderr,
                )
                return 0

            if attempt > 1:
                backoff = min(_RETRY_BACKOFF_CAP_S, 5.0 * (2 ** (attempt - 2)))
                print(
                    (
                        f"xpu auto-retry attempt {attempt} "
                        f"(sleeping {int(backoff)}s)"
                    ),
                    file=sys.stderr,
                )
                time.sleep(backoff)

            print(
                (
                    f"xpu auto-retry: attempt {attempt}, profile={failover_profile}, "
                    f"active={allocation.active}, spares={len(allocation.spare)}"
                ),
                file=sys.stderr,
            )
            try:
                rc, output = _run_with_watchdog_capture(full_cmd, timeout_s)
            except KeyboardInterrupt:
                print(
                    "[auto-retry] FAILOVER STOP: interrupted (SIGINT)",
                    file=sys.stderr,
                )
                return 128 + 2
            last_rc = rc
            attempt_log = log_dir / f"attempt-{attempt}.log"
            attempt_log.write_text(output, encoding="utf-8", errors="replace")

            bad_hosts_internal = _extract_bad_hosts(
                output,
                allocation.active,
                failover_profile,
                debug=args.failover_debug,
                manual_ip_map=manual_ip_map,
            )
            bad_hosts = bad_hosts_internal
            result = _classify_attempt(
                rc,
                output,
                bad_hosts,
                profile=failover_profile,
                prior_attempt_had_progress=prior_attempt_had_progress,
                has_spares=allocation.has_spares,
                consecutive_unattributed=consecutive_unattributed,
            )
            reason = result.reason
            prior_attempt_had_progress = result.has_progress
            if result.has_unattributed_io:
                consecutive_unattributed += 1
            else:
                consecutive_unattributed = 0
            has_crash = _has_crash_patterns(output, failover_profile)

            if args.failover_debug:
                crash_lines = _matching_lines(
                    output,
                    _get_crash_rx(failover_profile),
                    skip_innocent=True,
                    limit=3,
                )
                wall_lines = _matching_lines(
                    output,
                    _get_walltime_rx(failover_profile),
                    limit=3,
                )
                io_lines = _matching_lines(output, _UNATTRIBUTED_IO_RX, limit=3)
                _debug_log(
                    True,
                    f"crash_match={has_crash}; crash_lines={crash_lines if crash_lines else '[]'}",
                )
                _debug_log(
                    True,
                    f"walltime_lines={wall_lines if wall_lines else '[]'}",
                )
                _debug_log(
                    True,
                    f"io_lines={io_lines if io_lines else '[]'}",
                )

            if reason is TerminationReason.SUCCESS:
                print(
                    f"[auto-retry] FAILOVER STOP: success (attempt {attempt})",
                    file=sys.stderr,
                )
                return 0

            if reason is TerminationReason.WALLTIME:
                print(
                    f"[auto-retry] FAILOVER STOP: walltime (rc={rc}, attempt {attempt})",
                    file=sys.stderr,
                )
                return rc

            if reason is TerminationReason.STUCK_PRE_TRAINING:
                print(
                    (
                        "[auto-retry] FAILOVER STOP: stuck_pre_training "
                        "(two consecutive attempts showed no progress markers, "
                        f"rc={rc})"
                    ),
                    file=sys.stderr,
                )
                return rc

            if reason is TerminationReason.EXHAUSTED:
                print(
                    f"[auto-retry] FAILOVER STOP: exhausted (no spare nodes left, rc={rc})",
                    file=sys.stderr,
                )
                return rc

            if reason is TerminationReason.RETRYABLE_UNATTRIBUTED:
                print(
                    (
                        "[auto-retry] retryable_unattributed: storage failure "
                        "without host attribution; retrying on same hosts "
                        f"({consecutive_unattributed}/{_MAX_CONSECUTIVE_UNATTRIBUTED}, rc={rc})."
                    ),
                    file=sys.stderr,
                )
                attempt += 1
                continue

            if reason is TerminationReason.BAD_NODE_KNOWN:
                try:
                    swaps = allocation.swap_in(bad_hosts, attempt=attempt)
                except RuntimeError:
                    print(
                        f"[auto-retry] FAILOVER STOP: exhausted (named hosts already swapped, rc={rc})",
                        file=sys.stderr,
                    )
                    return rc

                if not swaps:
                    if allocation.has_spares:
                        bad, replacement = allocation.swap_one_blind(
                            attempt=attempt
                        )
                        excluded_hosts.add(bad)
                        print(
                            f"[auto-retry] bad nodes named but not active; blind rotation: {bad} -> {replacement}",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"[auto-retry] FAILOVER STOP: exhausted (named hosts already swapped, rc={rc})",
                            file=sys.stderr,
                        )
                        return rc
                else:
                    for bad, replacement in swaps:
                        excluded_hosts.add(bad)
                        print(
                            f"[auto-retry] bad node swap: {bad} -> {replacement}",
                            file=sys.stderr,
                        )
            else:
                if allocation.has_spares and allocation.active:
                    bad, replacement = allocation.swap_one_blind(
                        attempt=attempt
                    )
                    excluded_hosts.add(bad)
                    print(
                        f"[auto-retry] blind rotation: {bad} -> {replacement}",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"[auto-retry] FAILOVER STOP: exhausted (no bad host named and no spares left, rc={rc})",
                        file=sys.stderr,
                    )
                    return rc

            attempt += 1
    finally:
        with contextlib.suppress(OSError):
            os.remove(active_hostfile)


def _run_with_watchdog(cmd: Sequence[str], idle_timeout_s: int) -> int:
    if idle_timeout_s <= 0:
        return subprocess.run(cmd, check=False).returncode

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )

    last_activity = time.monotonic()
    activity_lock = threading.Lock()
    reader_done = threading.Event()

    def _drain_output() -> None:
        nonlocal last_activity
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                with activity_lock:
                    last_activity = time.monotonic()
        finally:
            reader_done.set()

    threading.Thread(target=_drain_output, daemon=True).start()

    try:
        while True:
            rc = process.poll()
            if rc is not None:
                return rc

            with activity_lock:
                idle_for = time.monotonic() - last_activity

            if idle_for >= idle_timeout_s:
                print(
                    (
                        f"xpu watchdog: no output for {idle_for:.1f}s "
                        f"(timeout={idle_timeout_s}s). Sending SIGTERM to PID {process.pid}."
                    ),
                    file=sys.stderr,
                )
                process.terminate()
                try:
                    process.wait(timeout=_WATCHDOG_KILL_GRACE_S)
                except subprocess.TimeoutExpired:
                    print(
                        (
                            f"xpu watchdog: PID {process.pid} still alive after "
                            f"{int(_WATCHDOG_KILL_GRACE_S)}s. Sending SIGKILL."
                        ),
                        file=sys.stderr,
                    )
                    process.kill()
                    process.wait()
                return _WATCHDOG_EXIT_CODE

            sleep_for = min(1.0, max(0.1, idle_timeout_s - idle_for))
            time.sleep(sleep_for)
    except KeyboardInterrupt:
        # Kill the child so the reader thread reaches EOF; otherwise it holds
        # the stdout buffer lock during interpreter shutdown (fatal error).
        _terminate_process(process)
        raise
    finally:
        reader_done.wait(timeout=2.0)


def _run_with_retries(
    cmd: Sequence[str], timeout_s: int, retries: Optional[int]
) -> int:
    max_attempts = None if retries is None else max(1, retries + 1)
    last_rc = 0
    attempt = 1
    while True:
        if max_attempts is not None and attempt > max_attempts:
            return last_rc
        if attempt > 1:
            backoff = min(_RETRY_BACKOFF_CAP_S, 5.0 * (2 ** (attempt - 2)))
            print(
                (
                    f"xpu retry {attempt - 1}/{(max_attempts - 1) if max_attempts is not None else 'inf'} "
                    f"(prior exit={last_rc}); sleeping {int(backoff)}s"
                ),
                file=sys.stderr,
            )
            time.sleep(backoff)
        last_rc = _run_with_watchdog(cmd, timeout_s)
        if last_rc == 0:
            return 0
        attempt += 1


def _normalize_command(command: list[str]) -> list[str]:
    if command and command[0] == "--":
        return command[1:]
    return command


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.print_source:
        print(__file__)
        return 0

    if args.filter:
        print(
            "xpu launch: --filter is deprecated compatibility flag and is ignored.",
            file=sys.stderr,
        )

    if args.auto_retry:
        print("xpu launch: --auto-retry enabled.", file=sys.stderr)

    accelerator = accelerators.detect_accelerator(explicit=args.accelerator)
    set_active_accelerator(accelerator)
    accel_backend = accelerators.get_accelerator(accelerator)
    if accel_backend is not None:
        os.environ.setdefault("XPU_LAUNCH_ACCELERATOR", accelerator)
        os.environ.setdefault(
            "XPU_LAUNCH_DIST_BACKEND", accel_backend.distributed_backend()
        )
        print(
            (
                f"xpu launch: accelerator={accelerator} "
                f"(dist_backend={accel_backend.distributed_backend()}, "
                f"visible_devices_env={accel_backend.visible_devices_env()})"
            ),
            file=sys.stderr,
        )

    command = _normalize_command(list(args.command))
    if not command:
        raise SystemExit("missing command to launch")

    scheduler = _resolved_scheduler(args)
    if args.hostfile is None:
        resolved_hostfile = resolve_hostfile(args.hostfile, scheduler, workdir=Path.cwd())
        if resolved_hostfile is not None:
            args.hostfile = resolved_hostfile

    launcher_tokens = _build_launcher_tokens(args, scheduler=scheduler)
    # Pin the rendezvous address to the first hostfile entry (rank 0's node)
    if (
        "MASTER_ADDR" not in os.environ
        and args.hostfile
        and os.path.isfile(args.hostfile)
    ):
        hosts = _read_hostfile(args.hostfile)
        if hosts:
            os.environ["MASTER_ADDR"] = hosts[0]
    # PALS has no world-size env var; publish the launcher's rank count
    if "WORLD_SIZE" not in os.environ and launcher_tokens:
        try:
            topo = _resolve_requested_topology(args)
            os.environ["WORLD_SIZE"] = str(topo.nproc)
        except Exception:
            pass
    full_cmd = [*launcher_tokens, *command]

    timeout_s = args.timeout
    if timeout_s is None:
        timeout_s = _AUTO_RETRY_DEFAULT_TIMEOUT_S if args.auto_retry else 0

    if args.auto_retry:
        return _run_with_auto_retry(args, command, timeout_s)

    if args.dry_run:
        print(shlex.join(full_cmd))
        return 0

    retries: Optional[int] = args.retries

    return _run_with_retries(full_cmd, timeout_s, retries)
