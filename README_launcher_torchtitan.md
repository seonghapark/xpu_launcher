# xpu_launcher and TorchTitan

`xpu_launcher` is framework-independent. The package under `src/` handles
scheduler detection, topology, accelerator detection, process launch, retry,
and bad-node failover. TorchTitan integration is isolated in
`xpu_torchtitan/`.

## Current launch path

All supported TorchTitan jobs use one wrapper:

```text
xpu_torchtitan/run_train_torchtitan.sh
  -> run_train.sh
  -> xpu launch
  -> mpiexec / mpirun / srun
  -> xpu_torchtitan/titan_train.py
  -> torchtitan.train.main()
  -> torchtitan.trainer.Trainer
```

`titan_train.py` normalizes MPI/PBS rank variables, installs the core XCCL
split-group workaround when required, and delegates to the official
TorchTitan entry point. The active path does not import `ezpz` or use
`FaultTolerantTrainer`.

The old model-specific wrappers
`run_agpt_torchtitan_xpu.sh` and `run_gemma_torchtitan_xpu.sh` have been
removed. Use only `run_train_torchtitan.sh` for TorchTitan training.

## Model selection

The wrapper accepts `MODEL_PATH` as an alias for `MODEL`. If both are set,
`MODEL` takes precedence. The selected directory is normalized to an absolute
path and passed to TorchTitan as `--hf_assets_path`.

`MODEL_PATH` supplies assets such as `config.json` and tokenizer files. It
does not select the model class. `MODULE` and `CONFIG` select a callable from
TorchTitan's config registry:

```text
MODULE=agpt CONFIG=agpt_2b
  -> torchtitan.models.agpt.config_registry.agpt_2b()

MODULE=llama3 CONFIG=llama3_8b
  -> torchtitan.models.llama3.config_registry.llama3_8b()
```

Every module used by the generic Trainer must provide a `config_registry.py`.
The current standalone Gemma implementation does not provide that registry,
so Gemma is not yet supported by this generic path.

## Usage

```bash
cd /lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan

# Single-node command inspection
MODEL_PATH=/path/to/model/assets \
MODULE=llama3 \
CONFIG=llama3_debugmodel \
./run_train_torchtitan.sh single --dry-run

# Multi-node run inside a PBS allocation; PBS_NODEFILE is used automatically
MODEL_PATH=/path/to/model/assets \
MODULE=llama3 \
CONFIG=llama3_8b \
./run_train_torchtitan.sh multi -- --training.steps 2000

# A hostfile can also be supplied explicitly
MODEL_PATH=/path/to/model/assets \
./run_train_torchtitan.sh multi /path/to/hosts --dry-run
```

Arguments following `--` are passed directly to TorchTitan.

Important environment variables:

| Variable | Meaning |
|---|---|
| `MODEL` / `MODEL_PATH` | Required model and tokenizer assets directory; `MODEL` takes precedence |
| `MODULE` | Config-registry module, default `llama3` |
| `CONFIG` | Callable in that module's `config_registry.py`, default `llama3_debugmodel` |
| `HF_ASSETS_PATH` | Optional override for the path passed as `--hf_assets_path` |
| `DATASET_NAME` | Dataset registry name, default `pg19_multinews` |
| `DATASET_PATH` | Optional dataset path |
| `SEQ_LEN` | Sequence length, default `16384` |
| `TRAINING_STEPS` | Training steps, default `100` |
| `LOG_DIR` | TorchTitan dump directory |
| `CKPT_FOLDER` | Checkpoint output folder under `LOG_DIR`, default `checkpoint` |
| `TORCHTITAN_ROOT` | Vendored TorchTitan root |

Launch topology and recovery are controlled by `NPROC_PER_NODE`, `NNODES`,
`NPROC`, `SCHEDULER`, `AUTO_RETRY`, `SPARE_NODES`, `FAILOVER_PROFILE`, and
`HOST_IP_MAP`. In a PBS job, multi-node mode derives hosts from
`PBS_NODEFILE`. The wrapper defaults to four processes per node; accelerator
detection does not choose this value automatically.

## Loading the AGPT 2B checkpoint

The model assets directory and a Distributed Checkpoint (DCP) are separate:

- `MODEL_PATH` provides configuration and tokenizer assets.
- `--checkpoint.initial_load_path` provides DCP model weights.
- `MODULE=agpt CONFIG=agpt_2b` ensures those weights are loaded into the
  matching architecture.

```bash
CKPT=/lus/flare/projects/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144/step-92859

MODEL_PATH=/lus/flare/projects/datascience/seonghapark/agpt-2b-v2-256n-step-92859-safetensors \
MODULE=agpt \
CONFIG=agpt_2b \
./run_train_torchtitan.sh multi -- \
  --checkpoint.initial_load_path "$CKPT"
```

This command performs model-only initialization. It is not a full resume of
the old run because the DCP contains SophiaG optimizer state while the current
core `agpt_2b` configuration uses AdamW. Do not use
`--checkpoint.initial_load_in_hf` for this DCP; that option is for Hugging Face
checkpoint loading.

New checkpoints are written beneath `LOG_DIR/CKPT_FOLDER`; that output setting
is independent from `--checkpoint.initial_load_path`.