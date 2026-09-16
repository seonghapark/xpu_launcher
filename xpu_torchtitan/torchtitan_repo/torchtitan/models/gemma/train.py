# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Instruction fine-tune ``google/gemma-7b`` with FSDP2 + HF Transformers.

Launch (single node, 8 devices):
    torchrun --nproc_per_node=8 -m torchtitan.models.gemma.train \
        --model_name_or_path ./assets/hf/gemma-7b \
        --dataset_name tatsu-lab/alpaca \
        --output_dir outputs/gemma-7b-sft

Or use ``run_sft.sh``.
"""

from __future__ import annotations

import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import tyro
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader, DistributedSampler

from .config import GemmaSFTConfig
from .model import GemmaModel
from .parallelize import parallelize_gemma
from .sftdataset import IGNORE_INDEX, SFTCollator, SFTDataset, load_sft_records
from .tokenizer import build_tokenizer


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


# Fallback lookup tables for env vars set by different launchers. torchrun
# populates ``RANK/WORLD_SIZE/LOCAL_RANK`` directly; MPI-style launchers
# (``mpiexec`` on Aurora / PALS / OpenMPI) use different names.
_RANK_ENV_KEYS = ("RANK", "PMI_RANK", "PALS_RANKID", "OMPI_COMM_WORLD_RANK")
_SIZE_ENV_KEYS = (
    "WORLD_SIZE",
    "PMI_SIZE",
    "PALS_NRANKS",
    "OMPI_COMM_WORLD_SIZE",
)
_LOCAL_RANK_ENV_KEYS = (
    "LOCAL_RANK",
    "PALS_LOCAL_RANKID",
    "PMI_LOCAL_RANK",
    "MPI_LOCALRANKID",
    "OMPI_COMM_WORLD_LOCAL_RANK",
)


def _first_env(keys: tuple[str, ...]) -> str | None:
    for k in keys:
        v = os.environ.get(k)
        if v is not None and v != "":
            return v
    return None


def _set_master_addr_via_mpi(comm=None) -> None:
    """Broadcast rank 0's hostname over MPI to populate ``MASTER_ADDR``.

    Required for multi-node ``mpiexec`` launches where every rank must agree
    on the same address. Falls back to the local hostname if ``mpi4py`` is
    unavailable (single-node case).
    """
    if comm is None:
        try:
            from mpi4py import MPI  # type: ignore
        except ImportError:
            import socket

            os.environ.setdefault("MASTER_ADDR", socket.gethostname())
            return
        comm = MPI.COMM_WORLD
    else:
        from mpi4py import MPI  # type: ignore # noqa: F401

    hostname = None
    if comm.Get_rank() == 0:
        # Prefer a routable hostname; fall back to `socket.gethostname()`.
        import socket

        hostname = socket.gethostname()
    master = comm.bcast(hostname, root=0)
    os.environ["MASTER_ADDR"] = str(master)


def _bootstrap_dist_env() -> None:
    """Ensure ``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK`` / ``MASTER_ADDR`` /
    ``MASTER_PORT`` are all set before ``init_process_group`` is called.

    Priority for rank / world_size:
      1. Existing torchrun-style env vars (already set)
      2. ``mpi4py.COMM_WORLD`` (authoritative under ``mpiexec`` / ezpz)
      3. Individual MPI-style env vars (``PMI_*``, ``PALS_*``, ``OMPI_*``)
      4. Single-process fallback (RANK=0, WORLD_SIZE=1)

    We reach for ``mpi4py`` when torchrun hasn't populated the env because
    Aurora / PALS does not always export a ``WORLD_SIZE``-equivalent variable
    (only per-rank ids), whereas ``mpi4py`` can always ask MPI directly.
    """
    torchrun_ok = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    rank_val: str | None = None
    size_val: str | None = None
    mpi_comm = None

    if not torchrun_ok:
        # Try mpi4py -- authoritative source under any MPI launcher.
        try:
            from mpi4py import MPI  # type: ignore

            mpi_comm = MPI.COMM_WORLD
            rank_val = str(mpi_comm.Get_rank())
            size_val = str(mpi_comm.Get_size())
        except ImportError:
            pass

        # Fall back to launcher-set env vars.
        if rank_val is None:
            rank_val = _first_env(_RANK_ENV_KEYS)
        if size_val is None:
            size_val = _first_env(_SIZE_ENV_KEYS)

        # Final fallback: single-process run.
        if rank_val is None or size_val is None:
            rank_val = "0"
            size_val = "1"

        os.environ["RANK"] = rank_val
        os.environ["WORLD_SIZE"] = size_val

    # LOCAL_RANK: honor any explicit setting, else consult launcher vars,
    # else default to 0 (correct for single-process).
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = _first_env(_LOCAL_RANK_ENV_KEYS) or "0"

    if "MASTER_ADDR" not in os.environ:
        _set_master_addr_via_mpi(mpi_comm)
    os.environ.setdefault("MASTER_PORT", "29500")

    _bootstrap_ccl_env(mpi_comm)


def _bootstrap_ccl_env(mpi_comm=None) -> None:
    """Populate CCL-facing env vars so oneCCL doesn't fall back to ATL probing.

    On Aurora, if ``CCL_LOCAL_RANK`` / ``CCL_LOCAL_SIZE`` are unset, oneCCL
    prints ``could not get local_idx/count from environment variables,
    trying to get them from ATL`` and can then hang in the ATL probe,
    especially when the FI provider is misconfigured. Setting them
    explicitly avoids the fallback entirely.
    """
    local_rank = os.environ["LOCAL_RANK"]
    os.environ.setdefault("CCL_LOCAL_RANK", local_rank)
    os.environ.setdefault("CCL_LOCAL_IDX", local_rank)

    # Node-local process count: prefer explicit launcher vars, then compute
    # via mpi4py's shared-memory split.
    local_size = (
        os.environ.get("PALS_LOCAL_SIZE")
        or os.environ.get("MPI_LOCALNRANKS")
        or os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE")
    )
    if local_size is None and mpi_comm is not None:
        try:
            from mpi4py import MPI  # type: ignore

            node_comm = mpi_comm.Split_type(MPI.COMM_TYPE_SHARED)
            local_size = str(node_comm.Get_size())
        except Exception:
            pass
    if local_size:
        os.environ.setdefault("CCL_LOCAL_SIZE", local_size)
        os.environ.setdefault("CCL_LOCAL_COUNT", local_size)


def _pick_xpu_backend() -> str:
    """Pick a distributed backend that can run on Intel XPU.

    Priority:
      1. ``TORCH_XPU_BACKEND`` env override (advanced users).
      2. ``xccl`` if PyTorch reports it as available (native support in
         recent nightly XPU builds).
      3. ``ccl`` if ``oneccl_bindings_for_pytorch`` can be imported (this
         registers the backend as a side effect).
    Raises with an actionable message if none work.
    """
    override = os.environ.get("TORCH_XPU_BACKEND")
    if override:
        return override

    # xccl (native) -- available in recent nightly XPU builds.
    try:
        from torch.distributed import is_backend_available  # type: ignore

        if is_backend_available("xccl"):
            return "xccl"
    except (ImportError, AttributeError):
        pass

    # ccl (via oneCCL bindings) -- importing the module registers the backend.
    try:
        import oneccl_bindings_for_pytorch  # type: ignore  # noqa: F401

        return "ccl"
    except ImportError:
        pass

    raise RuntimeError(
        "XPU device is available but no distributed backend is registered.\n"
        "Fix by ONE of:\n"
        "  * Use a PyTorch nightly XPU build with native `xccl` support, or\n"
        "  * pip install oneccl_bind_pt --extra-index-url "
        "https://pytorch-extension.intel.com/release-whl/stable/xpu/us/\n"
        "Or set TORCH_XPU_BACKEND=<backend_name> to force a choice."
    )


def _pick_backend_and_device() -> tuple[str, torch.device]:
    """Pick the right dist backend and per-rank device.

    Order: XPU (Aurora / Intel) first, then CUDA (NVIDIA), then CPU. XPU is
    checked first so that on Aurora we pick an XPU backend even in the rare
    case where a stub CUDA install reports as available.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.set_device(local_rank)
        backend = _pick_xpu_backend()
        return backend, torch.device(f"xpu:{local_rank}")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return "nccl", torch.device(f"cuda:{local_rank}")

    return "gloo", torch.device("cpu")


def _setup_distributed() -> tuple[torch.device, int, int]:
    _bootstrap_dist_env()
    backend, device = _pick_backend_and_device()
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    return device, rank, world_size


def _is_rank0() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def _log(msg: str) -> None:
    if _is_rank0():
        print(msg, flush=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def _lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup + linear decay to 0."""
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    remain = max(0, total_steps - step)
    return remain / float(max(1, total_steps - warmup_steps))


# ---------------------------------------------------------------------------
# Weights & Biases + resource monitoring (rank-0 only)
# ---------------------------------------------------------------------------


class _WandbSink:
    """Thin rank-0-only wrapper around wandb. Silently no-ops if wandb is not
    installed or ``enable_wandb`` is False, so callers never need to guard.
    """

    def __init__(self, cfg: GemmaSFTConfig, extra_config: dict | None = None) -> None:
        self.enabled = False
        self.run = None
        if not (_is_rank0() and cfg.enable_wandb):
            return
        try:
            import wandb  # type: ignore
        except ImportError:
            print(
                "[wandb] enable_wandb=True but `wandb` is not installed; "
                "skipping. `pip install wandb` to enable.",
                flush=True,
            )
            return

        # WANDB_MODE env takes precedence if already set (useful for offline nodes).
        os.environ.setdefault("WANDB_MODE", cfg.wandb_mode)
        merged_conf = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
        if extra_config:
            merged_conf.update(extra_config)
        self.run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_run_name,
            config=merged_conf,
            dir=cfg.output_dir,
        )
        self.wandb = wandb
        self.enabled = True
        # Tell the user exactly where to view this run: URL for online runs,
        # local directory for offline runs (sync later with `wandb sync`).
        url = getattr(self.run, "url", None) if self.run else None
        run_dir = getattr(self.run, "dir", None) if self.run else None
        mode = os.environ.get("WANDB_MODE", cfg.wandb_mode)
        print(
            f"[wandb] project={cfg.wandb_project} "
            f"run={self.run.name if self.run else '?'} mode={mode}",
            flush=True,
        )
        if url:
            print(f"[wandb] view run at: {url}", flush=True)
        if run_dir:
            print(f"[wandb] local run dir: {run_dir}", flush=True)
        if mode == "offline" and run_dir:
            # `run_dir` is `<WANDB_DIR>/wandb/<run>/files`; parent is the
            # directory `wandb sync` expects.
            sync_target = os.path.dirname(run_dir.rstrip(os.sep))
            print(
                f"[wandb] offline mode -- sync later with: "
                f"wandb sync {sync_target}",
                flush=True,
            )

    def log(self, metrics: dict, step: int | None = None) -> None:
        if not self.enabled:
            return
        try:
            self.wandb.log(metrics, step=step)
        except Exception as e:  # pragma: no cover - never fatal
            print(f"[wandb] log failed: {e}", flush=True)

    def finish(self) -> None:
        if not self.enabled:
            return
        try:
            self.wandb.finish()
        except Exception:
            pass


class _ResourceMonitor:
    """Per-node XPU/CUDA utilization + power sampler with wandb aggregation.

    Architecture (multi-node safe):
      * Every LOCAL_RANK 0 process samples ``xpu-smi`` / ``nvidia-smi`` for
        its own node's devices and appends JSON records to a per-host log
        file under ``<output_dir>/power_log/<hostname>.jsonl`` (shared FS).
      * Global rank 0 additionally reads those files and pushes the samples
        to wandb using keys that group all nodes onto a single graph:
            ``power_w_xpu/<host>_xpu<N>``   -- per-XPU per-node power (W)
            ``util_pct_xpu/<host>_xpu<N>``  -- per-XPU per-node util (%)
            ``mem_used_mib_xpu/<host>_xpu<N>``
            ``power_w_node/<host>``          -- per-node total (W)
            ``power_w_total``                -- cluster-wide total (W)
            ``util_pct_avg``                 -- cluster-wide avg util (%)

    The shared prefix means wandb groups all node lines into one section, and
    a single line panel with regex ``^power_w_xpu/`` plots every XPU on every
    node in one graph. ``power_w_total`` is a convenient single-line summary.
    """

    def __init__(
        self,
        sink: _WandbSink,
        device: torch.device,
        interval_sec: float,
        output_dir: str,
        is_rank0: bool,
        is_local_rank0: bool,
    ) -> None:
        self.sink = sink
        self.device = device
        self.interval_sec = interval_sec
        self.is_rank0 = is_rank0
        self.is_local_rank0 = is_local_rank0
        import socket

        self.hostname = socket.gethostname()
        self.power_dir = os.path.join(output_dir, "power_log")
        self._stop = None
        self._sampler_thread = None
        self._reader_thread = None
        self._probe = "torch-only"
        if interval_sec > 0 and is_local_rank0:
            self._probe = self._pick_probe()

    # ---- probe selection -------------------------------------------------
    def _pick_probe(self) -> str:
        import shutil

        if self.device.type == "xpu" and shutil.which("xpu-smi"):
            return "xpu-smi"
        if self.device.type == "cuda" and shutil.which("nvidia-smi"):
            return "nvidia-smi"
        return "torch-only"

    # ---- device sampling (LOCAL_RANK 0 only) -----------------------------
    def _sample_xpu_smi(self) -> dict[str, dict[str, float]]:
        """Return {"<xpu_id>": {"util_pct":..., "power_w":..., "energy_j":...,
        "mem_used_mib":...}}.

        Aurora xpu-smi metric IDs (verified via `xpu-smi dump --help`):
          0 = GPU Utilization (%)
          1 = GPU Power (W)
          8 = GPU Energy Consumed (J, monotonic counter)
         18 = GPU Memory Used (MiB)
        """
        import subprocess

        try:
            out = subprocess.check_output(
                ["xpu-smi", "dump", "-d", "-1", "-m", "0,1,8,18", "-n", "1"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode("utf-8", errors="ignore")
        except Exception:
            return {}
        by_dev: dict[str, dict[str, float]] = {}
        for row in out.strip().splitlines():
            if not row or row.startswith("Timestamp"):
                continue
            parts = [p.strip() for p in row.split(",")]
            # Columns: Timestamp, DeviceId, util, power, energy, mem_used
            if len(parts) < 6:
                continue

            def _to_float(x: str) -> float | None:
                if x in ("", "N/A", "NA", "null"):
                    return None
                try:
                    return float(x)
                except ValueError:
                    return None

            dev = parts[1]
            util = _to_float(parts[2])
            power = _to_float(parts[3])
            energy = _to_float(parts[4])
            mem = _to_float(parts[5])
            entry: dict[str, float] = {}
            if util is not None:
                entry["util_pct"] = util
            if power is not None:
                entry["power_w"] = power
            if energy is not None:
                entry["energy_j"] = energy
            if mem is not None:
                entry["mem_used_mib"] = mem
            if entry:
                by_dev[dev] = entry
        return by_dev

    def _sample_nvidia_smi(self) -> dict[str, dict[str, float]]:
        import subprocess

        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,power.draw,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode("utf-8", errors="ignore")
        except Exception:
            return {}
        by_dev: dict[str, dict[str, float]] = {}
        for row in out.strip().splitlines():
            parts = [p.strip() for p in row.split(",")]
            if len(parts) < 4:
                continue
            try:
                dev = parts[0]
                util = float(parts[1])
                power = float(parts[2])
                mem = float(parts[3])
            except ValueError:
                continue
            by_dev[dev] = {
                "util_pct": util,
                "power_w": power,
                "mem_used_mib": mem,
            }
        return by_dev

    def _sample_torch(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        try:
            if self.device.type == "xpu" and hasattr(torch, "xpu"):
                metrics["torch_xpu_mem_alloc_gib"] = (
                    torch.xpu.memory_allocated() / (1024**3)
                )
                metrics["torch_xpu_mem_reserved_gib"] = (
                    torch.xpu.memory_reserved() / (1024**3)
                )
            elif self.device.type == "cuda":
                metrics["torch_cuda_mem_alloc_gib"] = (
                    torch.cuda.memory_allocated() / (1024**3)
                )
                metrics["torch_cuda_mem_reserved_gib"] = (
                    torch.cuda.memory_reserved() / (1024**3)
                )
        except Exception:
            pass
        return metrics

    def _sampler_loop(self) -> None:
        """Runs on LOCAL_RANK 0 of every node. Appends per-host JSONL."""
        import json

        assert self._stop is not None
        os.makedirs(self.power_dir, exist_ok=True)
        path = os.path.join(self.power_dir, f"{self.hostname}.jsonl")
        while not self._stop.is_set():
            xpus: dict[str, dict[str, float]] = {}
            if self._probe == "xpu-smi":
                xpus = self._sample_xpu_smi()
            elif self._probe == "nvidia-smi":
                xpus = self._sample_nvidia_smi()
            record = {
                "ts": time.time(),
                "host": self.hostname,
                "torch": self._sample_torch(),
                "xpus": xpus,
            }
            try:
                # Line-buffered append: each JSON line is atomic on POSIX for
                # writes smaller than PIPE_BUF (~4KB), which our records are.
                with open(path, "a") as f:
                    f.write(json.dumps(record) + "\n")
                    f.flush()
            except Exception:
                pass
            self._stop.wait(self.interval_sec)

    # ---- reader / wandb pusher (global rank 0 only) ----------------------
    def _reader_loop(self) -> None:
        """Runs on global rank 0. Tails every host's JSONL, logs to wandb."""
        import glob
        import json

        assert self._stop is not None
        offsets: dict[str, int] = {}
        # Give samplers a moment to create their files on first startup.
        self._stop.wait(min(2.0, self.interval_sec))
        while True:
            stopping = self._stop.is_set()
            try:
                files = sorted(glob.glob(os.path.join(self.power_dir, "*.jsonl")))
            except Exception:
                files = []

            new_records: list[dict] = []
            for f in files:
                start = offsets.get(f, 0)
                try:
                    with open(f, "r") as fh:
                        fh.seek(start)
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                new_records.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                        offsets[f] = fh.tell()
                except FileNotFoundError:
                    continue
                except Exception:
                    continue

            # Bucket records by timestamp (rounded to interval) so that samples
            # taken across nodes at approximately the same time land on the
            # same wandb x-axis point and can be summed into totals.
            if new_records:
                buckets: dict[int, list[dict]] = {}
                for rec in new_records:
                    key = int(rec["ts"] // max(1.0, self.interval_sec))
                    buckets.setdefault(key, []).append(rec)

                for ts_key in sorted(buckets):
                    metrics: dict[str, float] = {}
                    total_power = 0.0
                    util_sum = 0.0
                    util_count = 0
                    for rec in buckets[ts_key]:
                        host = rec.get("host", "unknown")
                        node_power = 0.0
                        for xpu_id, m in rec.get("xpus", {}).items():
                            tag = f"{host}_xpu{xpu_id}"
                            if "power_w" in m:
                                metrics[f"power_w_xpu/{tag}"] = m["power_w"]
                                node_power += m["power_w"]
                            if "util_pct" in m:
                                metrics[f"util_pct_xpu/{tag}"] = m["util_pct"]
                                util_sum += m["util_pct"]
                                util_count += 1
                            if "energy_j" in m:
                                metrics[f"energy_j_xpu/{tag}"] = m["energy_j"]
                            if "mem_used_mib" in m:
                                metrics[f"mem_used_mib_xpu/{tag}"] = m["mem_used_mib"]
                        if node_power > 0:
                            metrics[f"power_w_node/{host}"] = node_power
                            total_power += node_power
                        for k, v in rec.get("torch", {}).items():
                            metrics[f"resource_torch/{host}_{k}"] = v
                    if total_power > 0:
                        metrics["power_w_total"] = total_power
                    if util_count > 0:
                        metrics["util_pct_avg"] = util_sum / util_count
                    if metrics:
                        # Don't override the training step axis; let wandb use
                        # its internal step counter for these async samples.
                        self.sink.log(metrics)

            if stopping:
                break
            self._stop.wait(self.interval_sec)

    # ---- lifecycle -------------------------------------------------------
    def start(self) -> None:
        if self.interval_sec <= 0:
            return
        import threading

        self._stop = threading.Event()

        if self.is_local_rank0 and self._probe != "torch-only":
            self._sampler_thread = threading.Thread(
                target=self._sampler_loop,
                name=f"resource-sampler-{self.hostname}",
                daemon=True,
            )
            self._sampler_thread.start()
            print(
                f"[resource-monitor] host={self.hostname} probe={self._probe} "
                f"interval={self.interval_sec}s -> {self.power_dir}",
                flush=True,
            )
        elif self.is_local_rank0:
            # LOCAL_RANK 0 but no xpu-smi/nvidia-smi found: at least record
            # torch memory stats so users still see per-node activity.
            self._sampler_thread = threading.Thread(
                target=self._sampler_loop,
                name=f"resource-sampler-{self.hostname}",
                daemon=True,
            )
            self._sampler_thread.start()
            print(
                f"[resource-monitor] host={self.hostname} probe=torch-only "
                "(xpu-smi/nvidia-smi not on PATH; power will not be recorded)",
                flush=True,
            )

        if self.is_rank0 and self.sink.enabled:
            self._reader_thread = threading.Thread(
                target=self._reader_loop,
                name="resource-reader",
                daemon=True,
            )
            self._reader_thread.start()
            print(
                "[resource-monitor] rank0 aggregating per-node samples to wandb. "
                "In the wandb UI, add a line panel with metric-regex "
                "`^power_w_xpu/` to see all XPUs on all nodes in one graph, "
                "or use `power_w_total` for the cluster sum.",
                flush=True,
            )

    def stop(self) -> None:
        if self._stop is None:
            return
        self._stop.set()
        for t in (self._sampler_thread, self._reader_thread):
            if t is not None:
                t.join(timeout=self.interval_sec + 5.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@torch.no_grad()
def _evaluate(
    model: GemmaModel,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int = 0,
) -> dict[str, float]:
    """Compute validation cross-entropy loss and next-token top-1 accuracy.

    Reductions are averaged across ranks via ``dist.all_reduce`` so every
    caller gets the global metric. Only non-``IGNORE_INDEX`` label positions
    contribute.
    """
    was_training = model.training
    model.eval()

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    correct = torch.zeros((), device=device, dtype=torch.float64)
    tokens = torch.zeros((), device=device, dtype=torch.float64)

    for i, batch in enumerate(val_loader):
        if max_batches and i >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        # Ask the model for logits by NOT passing labels (so we can compute
        # per-token accuracy ourselves in addition to the loss).
        outputs = model.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs.logits  # [B, T, V]
        labels = batch["labels"]  # [B, T]

        # Standard causal-LM shift: predict token t+1 from position t.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        mask = shift_labels != IGNORE_INDEX

        # Loss (sum, so it can be averaged by the true non-ignored token count).
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        preds = shift_logits.argmax(dim=-1)
        correct_here = ((preds == shift_labels) & mask).sum()
        tokens_here = mask.sum()

        loss_sum += loss.double()
        correct += correct_here.double()
        tokens += tokens_here.double()

    if dist.is_initialized():
        for t in (loss_sum, correct, tokens):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    total_tokens = tokens.item()
    if total_tokens <= 0:
        metrics = {"val/loss": float("nan"), "val/accuracy": float("nan"),
                   "val/perplexity": float("nan"), "val/tokens": 0.0}
    else:
        avg_loss = loss_sum.item() / total_tokens
        metrics = {
            "val/loss": avg_loss,
            "val/accuracy": correct.item() / total_tokens,
            "val/perplexity": math.exp(min(avg_loss, 20.0)),
            "val/tokens": total_tokens,
        }

    if was_training:
        model.train()
    return metrics


# ---------------------------------------------------------------------------
# Checkpointing (HF format, rank-0 only)
# ---------------------------------------------------------------------------


def _save_hf_checkpoint(
    model: GemmaModel,
    tokenizer,
    save_dir: str,
) -> None:
    """Save an HF-format checkpoint by gathering full state on rank 0.

    Uses FSDP2's ``full_state_dict`` semantics: each parameter's ``full_tensor``
    is materialized on rank 0 while other ranks contribute their shards.
    """
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        StateDictOptions,
    )

    os.makedirs(save_dir, exist_ok=True) if _is_rank0() else None

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    full_sd = get_model_state_dict(model, options=options)

    if _is_rank0():
        # `full_sd` keys are prefixed with `model.` (the HF module inside
        # GemmaModel). Strip that prefix so HF's `from_pretrained` can consume
        # the checkpoint directly.
        prefix = "model."
        hf_sd = {}
        for k, v in full_sd.items():
            new_k = k[len(prefix):] if k.startswith(prefix) else k
            hf_sd[new_k] = v.detach().to("cpu")

        # Load into a lightweight (meta) copy for save_pretrained sharding.
        model.model.save_pretrained(save_dir, state_dict=hf_sd)
        tokenizer.save_pretrained(save_dir)
        print(f"[rank0] saved HF checkpoint to {save_dir}", flush=True)

    if dist.is_initialized():
        dist.barrier()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: GemmaSFTConfig) -> None:
    device, rank, world_size = _setup_distributed()
    _set_seed(cfg.seed + rank)

    _log(f"[config]\n{cfg}")
    _log(f"[dist] world_size={world_size} device={device}")

    # ------------------------ Tokenizer + data ------------------------
    tokenizer = build_tokenizer(cfg.model_name_or_path)

    records = load_sft_records(
        dataset_name=cfg.dataset_name,
        dataset_config_name=cfg.dataset_config_name,
        dataset_split=cfg.dataset_split,
        dataset_local_path=cfg.dataset_local_path,
    )
    _log(f"[data] loaded {len(records)} records")

    # Deterministic train/val split (same on every rank).
    val_records: list = []
    if 0.0 < cfg.validation_split < 1.0 and len(records) > 1:
        rng = random.Random(cfg.seed)
        idxs = list(range(len(records)))
        rng.shuffle(idxs)
        n_val = max(1, int(len(records) * cfg.validation_split))
        val_idx = set(idxs[:n_val])
        val_records = [records[i] for i in sorted(val_idx)]
        records = [r for i, r in enumerate(records) if i not in val_idx]
        _log(f"[data] train={len(records)} val={len(val_records)}")

    dataset = SFTDataset(
        records,
        tokenizer,
        max_seq_len=cfg.max_seq_len,
        instruction_key=cfg.instruction_key,
        input_key=cfg.input_key,
        output_key=cfg.output_key,
        mask_instruction=cfg.mask_instruction,
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=cfg.seed,
        drop_last=True,
    )
    collator = SFTCollator(pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(
        dataset,
        batch_size=cfg.per_device_batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=cfg.num_workers,
        pin_memory=(device.type in {"cuda", "xpu"}),
        drop_last=True,
    )

    val_loader: DataLoader | None = None
    if val_records:
        val_dataset = SFTDataset(
            val_records,
            tokenizer,
            max_seq_len=cfg.max_seq_len,
            instruction_key=cfg.instruction_key,
            input_key=cfg.input_key,
            output_key=cfg.output_key,
            mask_instruction=cfg.mask_instruction,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=cfg.seed,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.per_device_batch_size,
            sampler=val_sampler,
            collate_fn=collator,
            num_workers=cfg.num_workers,
            pin_memory=(device.type in {"cuda", "xpu"}),
            drop_last=False,
        )

    # ------------------------ Model + parallelism ---------------------
    _log(f"[model] loading {cfg.model_name_or_path}")
    model = GemmaModel(cfg)
    # ``from_pretrained`` loads on CPU. Move to the per-rank device BEFORE
    # FSDP2 shards so each rank ends up holding only its shard on-device.
    model.to(device)

    mesh = init_device_mesh(
        device_type=device.type, mesh_shape=(world_size,), mesh_dim_names=("fsdp",)
    )
    model = parallelize_gemma(
        model,
        mesh["fsdp"],
        param_dtype=cfg.param_dtype,
        reduce_dtype=cfg.reduce_dtype,
    )
    model.train()

    # ------------------------ Optimizer + schedule --------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
        foreach=True,
    )

    steps_per_epoch = math.ceil(len(loader) / cfg.gradient_accumulation_steps)
    total_steps = steps_per_epoch * cfg.num_epochs
    warmup_steps = max(1, int(cfg.warmup_ratio * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: _lr_lambda(s, warmup_steps, total_steps)
    )
    _log(
        f"[schedule] total_steps={total_steps} warmup_steps={warmup_steps} "
        f"steps_per_epoch={steps_per_epoch}"
    )

    # ------------------------ Observability --------------------------
    wandb_sink = _WandbSink(
        cfg,
        extra_config={
            "world_size": world_size,
            "device": str(device),
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
        },
    )
    monitor = _ResourceMonitor(
        wandb_sink,
        device,
        cfg.resource_log_interval_sec,
        output_dir=cfg.output_dir,
        is_rank0=_is_rank0(),
        is_local_rank0=(int(os.environ.get("LOCAL_RANK", "0")) == 0),
    )
    monitor.start()

    # ------------------------ Training loop ---------------------------
    global_step = 0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    running_loss = 0.0

    for epoch in range(cfg.num_epochs):
        sampler.set_epoch(epoch)
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            loss = model(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                attention_mask=batch["attention_mask"],
            )
            (loss / cfg.gradient_accumulation_steps).backward()
            running_loss += loss.detach().float().item()
            micro_step += 1

            if micro_step % cfg.gradient_accumulation_steps != 0:
                continue

            # Optimizer step boundary.
            if cfg.max_grad_norm and cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=cfg.max_grad_norm
                )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % cfg.log_interval == 0:
                avg_loss = running_loss / (
                    cfg.log_interval * cfg.gradient_accumulation_steps
                )
                elapsed = time.perf_counter() - t0
                lr = scheduler.get_last_lr()[0]
                _log(
                    f"[step {global_step}/{total_steps}] "
                    f"epoch={epoch} loss={avg_loss:.4f} lr={lr:.3e} "
                    f"elapsed={elapsed:.1f}s"
                )
                wandb_sink.log(
                    {
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "train/epoch": epoch + (global_step / max(1, total_steps)),
                        "train/elapsed_sec": elapsed,
                    },
                    step=global_step,
                )
                running_loss = 0.0

            if (
                val_loader is not None
                and cfg.eval_interval
                and global_step % cfg.eval_interval == 0
            ):
                val_metrics = _evaluate(
                    model, val_loader, device, max_batches=cfg.eval_max_batches
                )
                _log(
                    f"[eval step {global_step}] "
                    f"loss={val_metrics['val/loss']:.4f} "
                    f"acc={val_metrics['val/accuracy']:.4f} "
                    f"ppl={val_metrics['val/perplexity']:.3f}"
                )
                wandb_sink.log(val_metrics, step=global_step)

            if (
                cfg.save_interval
                and global_step % cfg.save_interval == 0
                and global_step > 0
            ):
                _save_hf_checkpoint(
                    model,
                    tokenizer,
                    os.path.join(cfg.output_dir, f"step-{global_step}"),
                )

        # End-of-epoch validation (runs regardless of `eval_interval`).
        if val_loader is not None:
            val_metrics = _evaluate(
                model, val_loader, device, max_batches=cfg.eval_max_batches
            )
            _log(
                f"[eval epoch {epoch}] "
                f"loss={val_metrics['val/loss']:.4f} "
                f"acc={val_metrics['val/accuracy']:.4f} "
                f"ppl={val_metrics['val/perplexity']:.3f}"
            )
            wandb_sink.log(
                {**val_metrics, "val/epoch": epoch + 1}, step=global_step
            )

    # Final save.
    _save_hf_checkpoint(
        model, tokenizer, os.path.join(cfg.output_dir, "final")
    )

    monitor.stop()
    wandb_sink.finish()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    # `use_underscores=True` keeps CLI flags matching the dataclass field names
    # (e.g. `--num_epochs` instead of tyro's default `--num-epochs`).
    cfg = tyro.cli(GemmaSFTConfig, use_underscores=True)
    try:
        main(cfg)
    except Exception:
        # Ensure non-zero exit propagates through torchrun cleanly.
        import traceback

        traceback.print_exc()
        sys.exit(1)
