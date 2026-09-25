"""Per-node CPU, memory, and accelerator resource monitoring.

Run this module as a command wrapper under a distributed launcher. Only local
rank zero samples host resources, so one JSONL stream is produced per node.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

_LOCAL_RANK_ENVS = (
    "LOCAL_RANK",
    "PALS_LOCAL_RANKID",
    "PMI_LOCAL_RANK",
    "MPI_LOCALRANKID",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "SLURM_LOCALID",
)

_CSV_FIELDS = (
    "timestamp",
    "hostname",
    "backend",
    "sampling_interval_seconds",
    "cpu_used_percent",
    "memory_total_bytes",
    "memory_available_bytes",
    "memory_used_bytes",
    "memory_used_percent",
    "process_pid",
    "process_rss_bytes",
    "device_id",
    "accelerator_utilization_percent",
    "accelerator_memory_used_mib",
    "accelerator_memory_used_percent",
    "accelerator_power_watts",
    "accelerator_energy_joules",
    "accelerator_temperature_celsius",
    "accelerator_error",
)


def _env_int(names: Sequence[str], default: int = 0) -> int:
    for name in names:
        value = os.environ.get(name)
        if value:
            try:
                return int(value)
            except ValueError:
                continue
    return default


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_cpu_times(path: str = "/proc/stat") -> tuple[int, int]:
    with open(path, encoding="utf-8") as file:
        fields = file.readline().split()
    if not fields or fields[0] != "cpu":
        raise RuntimeError(f"unexpected CPU statistics in {path}")
    values = [int(value) for value in fields[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


def _cpu_percent(previous: tuple[int, int], current: tuple[int, int]) -> float | None:
    total_delta = current[0] - previous[0]
    idle_delta = current[1] - previous[1]
    if total_delta <= 0:
        return None
    return round(100.0 * (total_delta - idle_delta) / total_delta, 3)


def _read_memory(path: str = "/proc/meminfo") -> dict[str, int]:
    values: dict[str, int] = {}
    with open(path, encoding="utf-8") as file:
        for line in file:
            key, raw = line.split(":", 1)
            parts = raw.split()
            if parts:
                values[key] = int(parts[0]) * 1024
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", values.get("MemFree", 0))
    return {
        "total_bytes": total,
        "available_bytes": available,
        "used_bytes": max(0, total - available),
        "used_percent": round(100.0 * (total - available) / total, 3)
        if total
        else 0.0,
    }


def _process_rss_bytes(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as file:
            for line in file:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
    return None


def _detect_backend() -> str:
    explicit = os.environ.get("XPU_LAUNCH_ACCELERATOR", "").lower()
    if explicit in {"xpu", "cuda", "rocm", "none"}:
        return explicit
    if os.environ.get("ZE_AFFINITY_MASK") or os.environ.get("ONEAPI_ROOT"):
        return "xpu"
    if os.environ.get("ROCR_VISIBLE_DEVICES") or os.environ.get("ROCM_PATH"):
        return "rocm"
    if os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("CUDA_HOME"):
        return "cuda"
    for backend, binary in (
        ("xpu", "xpu-smi"),
        ("cuda", "nvidia-smi"),
        ("rocm", "rocm-smi"),
    ):
        if shutil.which(binary):
            return backend
    return "none"


def _find_binary(name: str) -> str | None:
    if path := shutil.which(name):
        return path
    if name == "xpu-smi":
        release_roots: list[str] = []
        for value in (os.environ.get("LD_LIBRARY_PATH", ""), os.environ.get("PATH", "")):
            for entry in value.split(os.pathsep):
                if entry.startswith("/opt/aurora/"):
                    parts = Path(entry).parts
                    if len(parts) >= 4:
                        release_roots.append(str(Path(*parts[:4])))
        patterns = [
            f"{root}/spack/unified/*/install/*/xpu-smi-*/bin/xpu-smi"
            for root in dict.fromkeys(release_roots)
        ]
        patterns.append("/opt/aurora/*/spack/unified/*/install/*/xpu-smi-*/bin/xpu-smi")
        for pattern in patterns:
            matches = sorted(glob.glob(pattern), reverse=True)
            if matches:
                return matches[0]
    return None


def _number(value: str) -> float | int | None:
    value = value.strip().replace("%", "")
    if not value or value.upper() in {"N/A", "NA", "NONE", "[N/A]"}:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def _normalized_key(header: str) -> str:
    key = header.strip().lower()
    aliases = {
        "index": "device_id",
        "deviceid": "device_id",
        "gpu": "device_id",
        "gpu id": "device_id",
        "gpu utilization (%)": "utilization_percent",
        "gpu use (%)": "utilization_percent",
        "gpu use (%) ": "utilization_percent",
        "utilization.gpu [%]": "utilization_percent",
        "power.draw [w]": "power_watts",
        "gpu power (w)": "power_watts",
        "average graphics package power (w)": "power_watts",
        "gpu energy consumed (j)": "energy_joules",
        "memory.used [mib]": "memory_used_mib",
        "gpu memory used (mib)": "memory_used_mib",
        "gpu memory allocated (%)": "memory_used_percent",
        "temperature.gpu": "temperature_celsius",
        "gpu core temperature (celsius degree)": "temperature_celsius",
    }
    return aliases.get(key, key.replace(" ", "_"))


def _parse_csv_metrics(output: str) -> list[dict[str, Any]]:
    lines = [line for line in output.splitlines() if line.strip()]
    if len(lines) < 2:
        return []
    rows = csv.DictReader(lines, skipinitialspace=True)
    devices: list[dict[str, Any]] = []
    for row in rows:
        device: dict[str, Any] = {}
        for header, value in row.items():
            if header is None or value is None:
                continue
            key = _normalized_key(header)
            if key == "timestamp":
                continue
            parsed = _number(value)
            if parsed is not None:
                device[key] = parsed
        if device:
            devices.append(device)
    return devices


def _accelerator_command(backend: str) -> list[str] | None:
    if backend == "xpu" and (binary := _find_binary("xpu-smi")):
        return [binary, "dump", "-d", "-1", "-m", "0,1,8,18", "-n", "1"]
    if backend == "cuda" and (binary := _find_binary("nvidia-smi")):
        return [
            binary,
            "--query-gpu=index,utilization.gpu,power.draw,memory.used,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    if backend == "rocm" and (binary := _find_binary("rocm-smi")):
        return [binary, "--showuse", "--showpower", "--showmemuse", "--csv"]
    return None


def _sample_accelerators(backend: str) -> tuple[list[dict[str, Any]], str | None]:
    command = _accelerator_command(backend)
    if command is None:
        return [], f"telemetry command unavailable for backend {backend}"
    env = os.environ.copy()
    if backend == "xpu":
        for name in ("ZE_AFFINITY_MASK", "ONEAPI_DEVICE_SELECTOR", "SYCL_DEVICE_FILTER"):
            env.pop(name, None)
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
            env=env,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return [], str(error)
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        return [], detail
    if backend == "cuda":
        headers = "index,utilization.gpu [%],power.draw [W],memory.used [MiB],temperature.gpu"
        output = f"{headers}\n{result.stdout}"
    else:
        output = result.stdout
    devices = _parse_csv_metrics(output)
    if not devices:
        detail = result.stdout.strip() or "command returned no output"
        return [], f"no parseable accelerator metrics: {detail[:500]}"
    return devices, None


class _XpuStream:
    def __init__(self, interval: float) -> None:
        self.interval = max(1, int(round(interval)))
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.devices: list[dict[str, Any]] = []
        self.error: str | None = None

    def start(self) -> None:
        binary = _find_binary("xpu-smi")
        if binary is None:
            self.error = "xpu-smi is unavailable"
            self.ready.set()
            return
        env = os.environ.copy()
        for name in ("ZE_AFFINITY_MASK", "ONEAPI_DEVICE_SELECTOR", "SYCL_DEVICE_FILTER"):
            env.pop(name, None)
        command = [
            binary,
            "dump",
            "-d",
            "-1",
            "-m",
            "0,1,8,18",
            "-i",
            str(self.interval),
        ]
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        self.ready.wait(timeout=20)

    def _read(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        header: list[str] | None = None
        current_timestamp: str | None = None
        current_devices: list[dict[str, Any]] = []
        for line in self.process.stdout:
            fields = next(csv.reader([line], skipinitialspace=True))
            if header is None:
                if fields and fields[0].strip().lower() == "timestamp":
                    header = fields
                    self.ready.set()
                continue
            if len(fields) != len(header):
                continue
            row = dict(zip(header, fields))
            timestamp = row.get(header[0], "").strip()
            parsed = _parse_csv_metrics(",".join(header) + "\n" + line)
            if not parsed:
                continue
            if current_timestamp is not None and timestamp != current_timestamp:
                with self.lock:
                    self.devices = current_devices
                self.ready.set()
                current_devices = []
            current_timestamp = timestamp
            current_devices.extend(parsed)
        if current_devices:
            with self.lock:
                self.devices = current_devices
            self.ready.set()
        if self.process.poll() not in (None, 0):
            assert self.process.stderr is not None
            self.error = self.process.stderr.read().strip() or "xpu-smi stream failed"
            self.ready.set()

    def sample(self) -> tuple[list[dict[str, Any]], str | None]:
        with self.lock:
            return [device.copy() for device in self.devices], self.error

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.thread is not None:
            self.thread.join(timeout=2)


@dataclass
class _Summary:
    samples: int = 0
    sums: dict[str, float] = field(default_factory=dict)
    maxima: dict[str, float] = field(default_factory=dict)
    accelerator_energy_joules_estimate: float = 0.0

    def add(self, sample: dict[str, Any], elapsed_since_previous: float) -> None:
        self.samples += 1
        values: dict[str, float] = {
            "cpu_percent": sample.get("cpu", {}).get("used_percent"),
            "memory_used_bytes": sample.get("memory", {}).get("used_bytes"),
            "memory_used_percent": sample.get("memory", {}).get("used_percent"),
        }
        powers = [
            device.get("power_watts")
            for device in sample.get("accelerators", [])
            if device.get("power_watts") is not None
        ]
        utilizations = [
            device.get("utilization_percent")
            for device in sample.get("accelerators", [])
            if device.get("utilization_percent") is not None
        ]
        memory_values = [
            device.get("memory_used_mib")
            for device in sample.get("accelerators", [])
            if device.get("memory_used_mib") is not None
        ]
        if powers:
            values["accelerator_power_watts"] = sum(powers)
            self.accelerator_energy_joules_estimate += sum(powers) * elapsed_since_previous
        if utilizations:
            values["accelerator_utilization_percent"] = sum(utilizations) / len(utilizations)
        if memory_values:
            values["accelerator_memory_used_mib"] = sum(memory_values)
        for key, value in values.items():
            if value is None:
                continue
            numeric = float(value)
            self.sums[key] = self.sums.get(key, 0.0) + numeric
            self.maxima[key] = max(self.maxima.get(key, numeric), numeric)

    def as_dict(self, *, backend: str, started_at: str, elapsed_seconds: float) -> dict[str, Any]:
        return {
            "hostname": socket.gethostname(),
            "backend": backend,
            "started_at": started_at,
            "ended_at": _utc_now(),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "samples": self.samples,
            "averages": {
                key: round(value / self.samples, 3)
                for key, value in self.sums.items()
                if self.samples
            },
            "maxima": {key: round(value, 3) for key, value in self.maxima.items()},
            "accelerator_energy_joules_estimate": round(
                self.accelerator_energy_joules_estimate, 3
            ),
        }


def _monitor(
    output_dir: Path,
    interval: float,
    child_pid: int,
    stop: threading.Event,
    backend: str,
    xpu_stream: _XpuStream | None,
) -> None:
    hostname = socket.gethostname().split(".", 1)[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / f"{hostname}.jsonl"
    csv_path = output_dir / f"{hostname}.csv"
    summary_path = output_dir / f"{hostname}.summary.json"
    started_at = _utc_now()
    started = time.monotonic()
    previous_cpu = _read_cpu_times()
    previous_sample_time = started
    summary = _Summary()

    csv_exists = csv_path.exists() and csv_path.stat().st_size > 0
    with (
        samples_path.open("a", encoding="utf-8") as output,
        csv_path.open("a", encoding="utf-8", newline="") as csv_output,
    ):
        csv_writer = csv.DictWriter(csv_output, fieldnames=_CSV_FIELDS)
        if not csv_exists:
            csv_writer.writeheader()
        while not stop.is_set():
            sample_time = time.monotonic()
            current_cpu = _read_cpu_times()
            if xpu_stream is not None:
                devices, error = xpu_stream.sample()
            else:
                devices, error = _sample_accelerators(backend)
            sample: dict[str, Any] = {
                "timestamp": _utc_now(),
                "hostname": hostname,
                "backend": backend,
                "cpu": {"used_percent": _cpu_percent(previous_cpu, current_cpu)},
                "memory": _read_memory(),
                "process": {"pid": child_pid, "rss_bytes": _process_rss_bytes(child_pid)},
                "accelerators": devices,
            }
            if error:
                sample["accelerator_error"] = error
            previous_cpu = current_cpu
            summary.add(sample, sample_time - previous_sample_time)
            previous_sample_time = sample_time
            output.write(json.dumps(sample, sort_keys=True) + "\n")
            output.flush()
            csv_devices = devices or [{}]
            for device in csv_devices:
                csv_writer.writerow(
                    {
                        "timestamp": sample["timestamp"],
                        "hostname": hostname,
                        "backend": backend,
                        "sampling_interval_seconds": interval,
                        "cpu_used_percent": sample["cpu"]["used_percent"],
                        "memory_total_bytes": sample["memory"]["total_bytes"],
                        "memory_available_bytes": sample["memory"]["available_bytes"],
                        "memory_used_bytes": sample["memory"]["used_bytes"],
                        "memory_used_percent": sample["memory"]["used_percent"],
                        "process_pid": child_pid,
                        "process_rss_bytes": sample["process"]["rss_bytes"],
                        "device_id": device.get("device_id"),
                        "accelerator_utilization_percent": device.get("utilization_percent"),
                        "accelerator_memory_used_mib": device.get("memory_used_mib"),
                        "accelerator_memory_used_percent": device.get("memory_used_percent"),
                        "accelerator_power_watts": device.get("power_watts"),
                        "accelerator_energy_joules": device.get("energy_joules"),
                        "accelerator_temperature_celsius": device.get("temperature_celsius"),
                        "accelerator_error": error,
                    }
                )
            csv_output.flush()
            if stop.wait(interval):
                break

    summary_path.write_text(
        json.dumps(
            summary.as_dict(
                backend=backend,
                started_at=started_at,
                elapsed_seconds=time.monotonic() - started,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if args.interval <= 0:
        parser.error("--interval must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if _env_int(_LOCAL_RANK_ENVS) != 0:
        os.execvp(args.command[0], args.command)

    backend = _detect_backend()
    xpu_stream = _XpuStream(args.interval) if backend == "xpu" else None
    if xpu_stream is not None:
        xpu_stream.start()

    child = subprocess.Popen(args.command)
    stop = threading.Event()
    monitor = threading.Thread(
        target=_monitor,
        args=(args.output_dir, args.interval, child.pid, stop, backend, xpu_stream),
        daemon=True,
    )
    monitor.start()

    previous_handlers: dict[int, Any] = {}

    def forward_signal(signum: int, _frame: Any) -> None:
        if child.poll() is None:
            child.send_signal(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, forward_signal)
    try:
        return child.wait()
    finally:
        stop.set()
        monitor.join(timeout=max(21.0, args.interval + 1.0))
        if xpu_stream is not None:
            xpu_stream.stop()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
