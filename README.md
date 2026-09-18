# xpu-launch

Standalone launcher CLI + Python library for single/multi-node jobs on
PBS/SLURM systems (Aurora-first). No `ezpz` dependency.

Subcommands: `xpu launch` · `xpu doctor` · `xpu submit` · `xpu dist` · `xpu integrations`

## Contents

- [Quick start](#quick-start)
- [`xpu launch` reference](#xpu-launch-reference)
- [Auto-retry and failover](#auto-retry-and-failover)
- [Accelerator backends (XPU / CUDA / ROCm)](#accelerator-backends-xpu--cuda--rocm)
- [Python library](#python-library)
- [Package layout](#package-layout)
- [Tests](#tests)
- [CLI examples](#cli-examples)
- [Training templates](#training-templates)
- [Subsystem notes](#subsystem-notes)

## Quick start

```bash
cd /lus/flare/projects/datascience/seonghapark/xpu_launcher
pip install -e .

# run a command under the detected scheduler/launcher
xpu launch -- python train.py --epochs 10

# explicit topology + dry run
xpu launch -n 8 -ppn 4 --dry-run python train.py

# multi-node with host failover
xpu launch --auto-retry --hostfile hosts.txt --nhosts 4 --spare-nodes auto -- python train.py
```

For TorchTitan / generic training wrappers see
[Training templates](#training-templates).

## `xpu launch` reference

### Flags

| Flag | Meaning |
|---|---|
| `-n` / `--nproc` | total ranks |
| `-ppn` / `--nproc_per_node` | ranks per node |
| `-nh` / `--nnodes` | node count |
| `--hostfile PATH` | host list (else `HOSTFILE` / `PBS_NODEFILE`) |
| `--scheduler {auto,pbs,slurm,none}` | scheduler hint |
| `--accelerator {auto,xpu,cuda,rocm,none}` | accelerator backend (default `auto`) |
| `--cpu-bind ...` | passed through to the launcher |
| `--timeout N` | idle-watchdog seconds |
| `--retries N` | simple retry with backoff |
| `--auto-retry` | retry + bad-node failover ([details](#auto-retry-and-failover)) |
| `--spare-nodes N\|auto` | spare pool size for failover |
| `--failover-profile {auto,aurora,slurm,generic}` | crash-pattern profile |
| `--failover-debug` | print classifier matches per retry |
| `--host-ip-map FILE` | manual IP→host mapping (skips DNS) |
| `--dry-run` | print resolved command and exit |

### Launcher prefix resolution

`xpu launch` prepends a launcher prefix using this order; if none applies it
runs the command directly:

1. `--launcher "..."`
2. `DIST_LAUNCH`
3. `XPU_LAUNCHER`
4. `LAUNCH_CMD`
5. `LAUNCH`
6. auto-detect (`mpiexec`/`mpirun`/`srun`) when size flags are provided

### Scheduler & topology

- Scheduler detection (`auto`) uses `XPU_SCHEDULER`, then PBS/SLURM env,
	then scheduler binaries (`qsub`/`sbatch`).
- Topology inference enforces consistency for `nproc`, `nhosts`, `nproc_per_node`
	(no silent mismatch).
- Under SLURM, hostfile can be synthesized from `SLURM_NODELIST` (or active job info)
	for deterministic launch construction.
- Under scheduler jobs, `xpu launch` now assembles scheduler-aware launcher commands
	by default even without explicitly passing `--launcher`.
- If the scheduler-native launcher binary is unavailable (for example `srun` not on PATH),
	xpu falls back to the next available launcher (`mpiexec`/`mpirun`) while preserving
	inferred topology.
- PBS active job fallback: use `PBS_JOBID` fast path, then
	`qstat`-based user-job scan and `/var/spool/pbs/aux` nodefile lookup.
- SLURM active job fallback: use `SLURM_JOB_ID` fast path,
	then running-job discovery (`sacct`/`squeue`) and `scontrol show job` nodelist lookup.

## Auto-retry and failover

`--auto-retry` performs host failover with active/spare rotation. Provide a
hostfile with `--hostfile` (or via `HOSTFILE`/`PBS_NODEFILE`; inside a PBS job
it is derived automatically).

Classification behavior:

- Walltime-like exits (for example rc 143 with time-limit logs) stop without host swap.
- Unattributed storage failures (`Errno 28`, `Errno 122`) retry in place first.
- Crash lines that name hosts trigger targeted host swap from spare pool.
- If no host is named, a blind rotation is used while spares remain.
- On `srun` launches, detected bad hosts are cumulatively re-injected via
	`--exclude=<host1,host2,...>` across retries.
- The auto-retry classifier follows an explicit termination matrix with
	explicit reasons: success, walltime, bad_node_known, bad_node_blind,
	retryable_unattributed, stuck_pre_training, exhausted.
- Pattern handling is synchronized more closely: innocent rank-cascade lines
	are filtered (`rank N died from signal 11/15`), walltime vs crash is
	disambiguated, and `Execution finished with N` inner exit codes are respected.

Tuned signatures now include real incident forms such as:

- `rank N exited with code K` (host-attributed crash)
- `RuntimeError: could not create a memory`
- `MemoryError: std::bad_alloc`
- `srun: error: <node>: tasks ...: Killed`

`--failover-profile` defaults to `auto`:

- Detects `slurm` for `srun` or `SLURM_*` environments.
- Detects `aurora` for `PBS_*`/`COBALT_*`/XPU hints.
- Falls back to `generic` otherwise.

Use `--failover-debug` to print which crash/walltime/storage patterns matched
and which host was attributed as bad on each retry attempt.
When crash lines only contain peer IPs, xpu also attempts host attribution via
hostfile DNS lookups plus reverse-DNS fallback.
To avoid DNS completely, pass `--host-ip-map` with a manual IP mapping file.

Accepted `--host-ip-map` JSON shapes:

- `{ "10.113.12.17": "x4007c6s4b0n0.hsn.cm.aurora.alcf.anl.gov" }`
- `{ "x4007c6s4b0n0.hsn.cm.aurora.alcf.anl.gov": ["10.113.12.17", "10.113.12.42"] }`
- `{ "entry1": { "host": "x4007c6s4b0n0", "ips": ["10.113.12.17"] } }`

## Accelerator backends (XPU / CUDA / ROCm)

Backend-specific functionality is isolated in one module per backend with an
identical public API, so callers use `cuda.func(...)`, `xpu.func(...)`,
`rocm.func(...)` interchangeably:

```python
from xpu_launch.accelerators import cuda, xpu, rocm, detect_accelerator

cuda.visible_devices_env()    # "CUDA_VISIBLE_DEVICES"
xpu.visible_devices_env()     # "ZE_AFFINITY_MASK"
rocm.visible_devices_env()    # "ROCR_VISIBLE_DEVICES" (+ HIP_VISIBLE_DEVICES)

cuda.distributed_backend()    # "nccl"
xpu.distributed_backend()     # "xccl"
rocm.distributed_backend()    # "nccl" (RCCL registers as nccl)

cuda.smi_binary()             # "nvidia-smi"
xpu.smi_binary()              # "xpu-smi"
rocm.smi_binary()             # "rocm-smi"

detect_accelerator()          # "xpu" | "cuda" | "rocm" | "none"
```

Common API per module: `name`, `is_available`, `device_count`,
`visible_devices_env`, `extra_visible_devices_envs`, `visible_devices`,
`set_visible_devices`, `distributed_backend`, `collective_library`,
`smi_binary`, `smi_query_command`, `env_hints`, `crash_patterns`,
`doctor_payload`.

CLI integration:

- `xpu launch --accelerator {auto,xpu,cuda,rocm,none}` (default `auto`,
  probing order XPU → CUDA → ROCm; override with `XPU_LAUNCH_ACCELERATOR`).
- The resolved backend's crash signatures (NCCL/CUDA, Level Zero/oneCCL,
  HIP/RCCL) augment `--auto-retry` bad-node classification.
- `XPU_LAUNCH_ACCELERATOR` and `XPU_LAUNCH_DIST_BACKEND` are exported to
  launched processes.
- `xpu doctor` reports per-backend availability, device count, SMI tool, and
  an API-parity check under `xpu.accelerator`.

## Python library

Usable as a library after `pip install -e .`:

```python
from xpu_launch import __version__
from xpu_launch.launch import build_launch_parser, parse_args, run

print(__version__)
args = parse_args(["-n", "4", "-ppn", "4", "--", "python", "train.py"])
run(args)
```

CLI entry points also remain available:

```bash
xpu --help
python -m xpu_launch --help
```

## Package layout

The implementation modules are under `src/cli`, and the installable Python
library namespace is `src/xpu_launch`.

- src/cli/__init__.py: top-level Click group and xpu command entry
- src/cli/launch_cmd.py: xpu launch subcommand wrapper
- src/cli/launch.py: launcher runtime and auto-retry/failover logic
- src/cli/scheduler_topology.py: scheduler detection + topology inference + hostfile resolution
- src/cli/failover_models.py: NodeAllocation/BadNodeRecord/provenance postmortem model
- src/cli/compat.py: standalone machine/scheduler/distributed summary helpers
- src/cli/accelerators/: cuda/xpu/rocm backend modules with identical public API
- src/cli/doctor_cmd.py, dist_cmd.py, submit_cmd.py, integrations_cmd.py: subsystem entrypoints
- src/cli/__about__.py: version
- src/xpu_launch/__init__.py, cli.py, launch.py, __main__.py: public library/API facade

## Tests

Parity-oriented tests under `tests/`:

- `test_launch_snapshots.py`: dry-run snapshots for scheduler/topology launch assembly,
  and auto-retry `srun --exclude` reinjection behavior.
- `test_scheduler_topology.py`: PBS/SLURM job-query fallback and hostfile resolution paths.
- `test_classifier_sync.py`: enum/state-machine and pattern classification parity checks.
- `test_failover_models.py`: NodeAllocation/BadNodeRecord/provenance persistence checks.
- `test_subsystems_cli.py`: top-level subsystem command registration and doctor output checks.

## CLI examples

```bash
xpu launch -- python train.py --epochs 10
xpu launch --help
xpu launch -n 8 -ppn 4 --dry-run python train.py
xpu launch --auto-retry --hostfile /path/to/hosts --nhosts 4 --spare-nodes auto -- python train.py
xpu doctor --hostfile /path/to/hosts --nproc 8 --nproc-per-node 4
xpu dist validate --hostfile /path/to/hosts --nproc 8 --nproc-per-node 4
xpu submit --hostfile /path/to/hosts --nproc 8 --nproc-per-node 4 --command "python train.py" --script-path ./job.sh --no-run
```

## Training templates

Two executable helper scripts are included in this repo:

- `xpu_torchtitan/run_train_torchtitan.sh`: `xpu launch` wrapper for real TorchTitan training.
- `run_train_non_torchtitan.sh`: `xpu launch` wrapper for generic Python training scripts.

Both support single-node and multi-node modes.

### TorchTitan template

```bash
cd /lus/flare/projects/datascience/seonghapark/xpu_launcher

# single node
MODEL=/path/to/hf/model ./xpu_torchtitan/run_train_torchtitan.sh single --dry-run

# multi node (hostfile optional inside a PBS job: PBS_NODEFILE is used)
MODEL=/path/to/hf/model ./xpu_torchtitan/run_train_torchtitan.sh multi --dry-run
MODEL=/path/to/hf/model ./xpu_torchtitan/run_train_torchtitan.sh multi /path/to/hosts --dry-run

# real run-style example
MODEL=/path/to/hf/Llama-3.1-8B \
MODULE=llama3 \
CONFIG=llama3_8b \
DATASET_NAME=c4 \
DATASET_PATH=allenai/c4 \
LOG_DIR=/path/to/logs/tt_run1 \
CKPT_FOLDER=checkpoint \
./xpu_torchtitan/run_train_torchtitan.sh multi -- --training.steps 5000
```

Key env vars for `xpu_torchtitan/run_train_torchtitan.sh`:

- `MODEL` **(required)**: path to the model/tokenizer assets directory
  (forwarded to torchtitan as `--hf_assets_path`)
- `MODULE` (default: `llama3`; torchtitan config-registry module)
- `CONFIG` (default: `llama3_debugmodel`; callable in that module's `config_registry.py`)
- `HF_ASSETS_PATH` (optional override; default: `MODEL`)
- `DATASET_NAME` (default: `pg19_multinews` — PG19+MultiNews interleaved, HF streaming; `c4_test` = offline bundled sample)
- `DATASET_PATH` (optional)
- `LOG_DIR` (default: `xpu_torchtitan/torchtitan_repo/outputs/xpu_torchtitan_<timestamp>`)
- `CKPT_FOLDER` (default: `checkpoint`)
- `TRAINING_STEPS` (default: `100`)
- `SEQ_LEN` (default: `16384`, passed as `--training.seq_len`)
- `TORCHTITAN_ROOT` (default: `xpu_torchtitan/torchtitan_repo`)

#### `config_registry.py` is mandatory for every MODULE

TorchTitan resolves `MODULE`/`CONFIG` through a `config_registry.py` file — a
new module **must** ship one or config parsing fails with
`Cannot import config_registry for module '<name>'`.

Resolution order for `MODULE=<name>`
(`torchtitan/config/manager.py`):

1. `torchtitan.models.<name>.config_registry`
2. `torchtitan.experiments.<name>.config_registry`
3. `torchtitan.experiments.rl.examples.<name>.config_registry`
4. Fully qualified paths also work: `MODULE=my.pkg` tries
   `my.pkg.config_registry`, then `my.pkg` itself.

`CONFIG=<name>` must be a **callable defined in that file** returning a
`Trainer.Config`. Minimal example
(`torchtitan_repo/torchtitan/models/mymodel/config_registry.py`):

```python
from torchtitan.trainer import Trainer

def mymodel_debug() -> Trainer.Config:
    cfg = ...  # build model_spec / dataloader / optimizer config
    return cfg
```

Then run with `MODULE=mymodel CONFIG=mymodel_debug`. Existing example:
`torchtitan/models/llama3/config_registry.py` (used by the defaults
`MODULE=llama3 CONFIG=llama3_debugmodel`).

### Non-TorchTitan template

```bash
cd /lus/flare/projects/datascience/seonghapark/xpu_launcher

# single node
./run_train_non_torchtitan.sh single --dry-run -- --epochs 1

# multi node
./run_train_non_torchtitan.sh multi /path/to/hosts --dry-run -- --epochs 1

# real run-style example
MODEL_PATH=/path/to/model \
DATASET_PATH=/path/to/dataset \
LOG_DIR=/path/to/logs/generic1 \
./run_train_non_torchtitan.sh multi /path/to/hosts -- --epochs 3 --lr 1e-4
```

Key env vars for `run_train_non_torchtitan.sh`:

- `TRAIN_ENTRY` (default: `train.py`)
- `PYTHON_BIN` (default: `python`)
- `MODEL_PATH` (default: `/path/to/model`)
- `DATASET_PATH` (default: `/path/to/dataset`)
- `LOG_DIR` (default: `./outputs/xpu_generic_<timestamp>`)

Both templates forward launch-related environment controls to `run_train.sh`:

- `XPU_CMD`, `SCHEDULER`, `NPROC_PER_NODE`, `NNODES`, `NPROC`
- `AUTO_RETRY`, `SPARE_NODES`, `FAILOVER_PROFILE`, `HOST_IP_MAP`

## Subsystem notes

- `xpu doctor` now emits structured checks for scheduler env, launcher availability,
	hostfile resolution, and topology inference.
- `xpu doctor` now also checks MPI/PMI consistency: launcher detection,
	linked MPI/PMI libraries (`ldd`), PMI/PMIx environment visibility,
	and the resolved MPI/PMI soname lists.
- `xpu doctor` also reports launcher family (`pals`/`mpich`/`openmpi`/`slurm`) and
	applies PALS-aware PMI compatibility logic (for Cray PALS environments where
	`libpmi` may not appear directly in launcher `ldd` output).
- `xpu dist validate` performs mismatch diagnostics between inferred topology and
	distributed/runtime environment hints (`WORLD_SIZE`, `LOCAL_WORLD_SIZE`, etc.).
- `xpu submit` now generates scheduler scripts (`PBS`/`SLURM`) and can optionally
	submit via `qsub` or `sbatch` using `--run`.
- `xpu submit` supports machine-aware PBS defaults:
	Aurora/Sunspot/Sophia -> `filesystems=home:flare`, `place=scatter`
	Polaris -> `filesystems=home:eagle`, `place=scatter`.
	You can override with `--queue`, `--account`, `--filesystems`.

Examples:

```bash
export DIST_LAUNCH='mpiexec -n 8 --ppn 4'
xpu launch -- python train.py --epochs 10

xpu launch --launcher 'srun -N2 -n16 -u' -- python train.py
```
