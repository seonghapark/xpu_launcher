# Standalone AGPT os._exit experiments

Copied launchers, TorchTitan source, shutdown/cache shims, tokenizer/config assets,
and both PBS experiments live here. Nothing in this package imports or executes
the original launcher checkout. Original source files and running jobs are unchanged.
The vendored TorchTitan license is in `xpu_torchtitan/torchtitan_repo/LICENSE`.

## Input and environment

- Initial native DCP:
  `/lus/flare/projects/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144/step-49900`
- Only the 111 model tensors are loaded. SophiaG optimizer, LR schedule, data
  position, and training step are **not** restored. Each trial starts at step 0
  and performs 3 steps using the original experiment's AGPT 2B YaRN settings.
- Model keys and shapes must match before loading; invalid inputs raise errors.
  The external DCP is read-only and is never used as an output directory.
- Tokenizer/config files are local in `assets/`; no HF model weights are used.
- External prerequisites remain: Aurora `frameworks/2026.1.0`,
  `/lus/flare/projects/datascience/seonghapark/venv`, local PG19 parquet data at
  `/lus/flare/projects/datascience/seonghapark/assets/hf/datasets/pg19/data`,
  PBS/MPI/SSH and the external input DCP.
- Local `cli`, `xpu_launch`, and `torchtitan` sources take precedence over any
  editable installation in the shared venv. There is no install/download step.

## Submit

```bash
cd /lus/flare/projects/datascience/seonghapark/agpt_osexit_standalone
qsub -v AGPT_PACKAGE_ROOT="$PWD" test_job_osexit_experiment.sh
# Submit the other experiment separately, after the first job finishes:
qsub -v AGPT_PACKAGE_ROOT="$PWD" test_job_osexit_ccl_conditions.sh
```

Both default to `debug-scaling`, 4 nodes, 1 hour. Do not submit both simultaneously
against the queue's per-user limits. No experiment job was submitted while creating
this package.

PBS spools scripts outside this directory, so pass `AGPT_PACKAGE_ROOT` as above
if you move the package. Override the model input with
`-v "AGPT_PACKAGE_ROOT=$PWD,AGPT_INITIAL_DCP=/absolute/path/to/step-N"`.

## Outputs and behavior

- Each job writes under this package's `outputs/`.
- Each CCL condition has its own training/checkpoint folder; the simple experiment
  also has a job-local training folder. Reload probes inspect only that trial.
- Inductor isolation and the guarded shutdown shim retain their original behavior.
- The CCL experiment retains the original A/B/C/S conditions and pmix acceptance
  probe. An unsupported launcher is a configuration failure, not a clean result.
- These remain hardware experiments; copied source and CPU validation do not prove
  that the teardown defect reproduces on a new node allocation.

## Validate without submitting jobs

```bash
module load frameworks/2026.1.0
source /lus/flare/projects/datascience/seonghapark/venv/bin/activate
source ./package_env.sh
python -m unittest discover -s tests -v
```

Tests cover model-only loading, empty output directories, local-checkpoint
precedence, invalid shapes/missing input, all 111 real model key/shape matches,
and an actual tensor read from the requested step-49900 DCP.
