# Running with Newer PyTorch (>= 2.10)

> [!NOTE]
> We will use the alias `uvi`:
>
> ```bash
> alias uvi='uv pip install --no-cache --link-mode=copy'
> ```

> [!IMPORTANT]
> To access the internet, you need to set the following environment variables:
>
> ```bash
> export http_proxy="http://proxy.alcf.anl.gov:3128"
> export https_proxy="http://proxy.alcf.anl.gov:3128"
> export no_proxy="localhost,127.0.0.1,*.alcf.anl.gov,*.anl.gov"
> ```

1. Clone torchtitan:

   ```bash
   gh repo clone saforem2/torchtitan -- --branch ezpz
   cd torchtitan
   ```

1. Load modules and export environment variables[^ezpz-setup]:

   ```bash
   source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job && ezpz_load_modules
   ```

1. Create venv:

   ```bash
   # to use the python from `/opt/aurora/.../python-3.12.12-xxx/bin/python3`
   module load python

   # create venv
   uv venv \
       --system-site-packages \
       --relocatable \
       --no-cache \
       --link-mode=copy \
       --python=$(which python3)

   # activate venv
   source .venv/bin/activate
   ```

1. Install PyTorch:

   ```bash
   uvi torch torchvision torchaudio torchdata \
       --pre \
       --index-url https://download.pytorch.org/whl/nightly/xpu \
       --upgrade
   ```

   - **NOTE** (2026-06-09): The _nightly_ PyTorch 2.13 has missing symbols and
     is **currently** broken.  
     The latest (confirmed) functional PyTorch 2.13 wheel is `torch==2.13.0.dev20260519+xpu`

1. Install dependencies:

   ```bash
   uvi spmd_types torchcomms tyro tensorboard deepspeed mpi4py
   uvi "git+https://github.com/zhenghh04/blendcorpus"
   uvi "git+https://github.com/saforem2/ezpz"
   ```

1. Remove Intel's MPI runtime (`impi-rt`):

   ```bash
   uv pip uninstall impi-rt
   ```

1. Download tokenizers:

   ```bash
   python3 scripts/download_hf_assets.py --repo_id google/gemma-7b --assets tokenizer
   ```

1. Run training:

   ```bash
   MODULE=ezpz.agpt
   CONFIG=agpt_2b
   ezpz launch python3 -m torchtitan.experiments.ezpz.train \
       --module="${MODULE}" \
       --config="${CONFIG}" \
       --training.steps=10 \
       --checkpoint.no-enable \
       --training.local-batch-size=2
   ```

   - <details closed><summary>AuroraGPT-20B:</summary>

     ```bash
     MODULE=ezpz.agpt
     CONFIG=agpt_20b
     ezpz launch python3 -m torchtitan.experiments.ezpz.train \
         --module="${MODULE}" \
         --config="${CONFIG}" \
         --training.steps=10 \
         --checkpoint.no-enable \
         --training.local-batch-size=2
     ```

     </details>

   - <details closed><summary>AuroraGPT-80B:</summary>

     ```bash
     MODULE=ezpz.agpt
     CONFIG=agpt_80b
     ezpz launch python3 -m torchtitan.experiments.ezpz.train \
         --module="${MODULE}" \
         --config="${CONFIG}" \
         --training.steps=10 \
         --checkpoint.no-enable \
         --training.local-batch-size=1 \
         --optimizer=adamw \
         --optimizer.lr=1e-6 \
         --parallelism.tensor-parallel-degree=2 \
         --compile.no-enable
     ```

     See [`guides/training/agpt_80b.md`](training/agpt_80b.md) for the
     full 80B walkthrough (prerequisites, expected step-by-step
     numbers, scale-out, known issues).

     </details>

[^ezpz-setup]: Explicitly, the `ezpz_load_modules` sets:

     ```bash
     module load oneapi/release/2025.3.1 hdf5 pti-gpu
     export ZE_FLAT_DEVICE_HIERARCHY=FLAT
     export CCL_PROCESS_LAUNCHER=pmix
     export CCL_OP_SYNC=1
     export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
     export TORCH_CPP_LOG_LEVEL=ERROR
     ```


## Running at Large Scale (> 512 nodes)

At a few nodes, importing Python from a shared filesystem is a small tax —
~milliseconds per import, paid once.

At 6k workers all hammering Lustre with the same import waterfall, that small
tax becomes minutes of dead time before the first training step lands, plus
tail-latency stragglers that dominate collective wait time for the rest of the
run.

[`ezpz yeet`](https://ezpz.cool/cli/yeet/) sidesteps this by copying the active
venv (or a pre-built `.venv.tar.gz`) to node-local `/tmp/` storage on every
worker, so subsequent imports, checkpoint loads, and config reads hit local SSD
instead of Lustre.

The local copy is patched once on the source node (activate scripts, shebangs,
symlinks) and then distributed via a greedy `rsync` fan-out — each completed
node immediately becomes a source for others, so the broadcast tree grows in
roughly `O(log N)` time instead of saturating one NIC.

### Measured scaling on Aurora (8 → 4096 nodes)

| Nodes | yeet (s) | Per-node (ms) |
| ----: | -------: | ------------: |
|     8 |     69.7 |         8,712 |
|    16 |     89.7 |         5,606 |
|    32 |     89.2 |         2,788 |
|    64 |     91.2 |         1,425 |
|   128 |    110.4 |           862 |
|   256 |    132.9 |           519 |
|   512 |    174.5 |           341 |
|  1024 |    255.4 |           249 |
|  2048 |    421.4 |           206 |
|  4096 |    750.6 |           183 |

Two regimes:

- **< 128 nodes** the cost is dominated by the one-time local extract (~70-91 s
  flat)
- **≥ 128 nodes** the broadcast tree depth and per-leaf contention dominate,
  with each 2× in nodes adding ~1.5-1.8× wall-clock.

Even at full-Aurora 4096-node scale the pre-launch overhead is under 13 minutes,
versus the 1--2 hours the per-file `rsync` mode was projected to take.

For more detail (full sweep, plots, methodology) see the
[yeet CLI docs](https://ezpz.cool/cli/yeet/) and the
[benchmark harness](https://github.com/saforem2/torchtitan/tree/ezpz/torchtitan/experiments/ezpz/docs/scaling/yeet_env).

### Workflow

> [!TIP]
> If you already built a tarball with `ezpz tar-env`, pass it explicitly —
> tarball broadcast is ~10× faster than per-file `rsync` at scale because the
> Lustre side becomes one sequential read instead of millions of `stat()`s.
>
> Plain `ezpz yeet` will print a hint when it sees a same-named `.tar.gz`
> sitting nearby.

1. Be sure to load the appropriate modules and activate the `.venv` we just
   created:

   ```bash
   source <(curl -fsSL https://bit.ly/ezpz-utils)
   ezpz_setup_job
   ezpz_setup_xpu
   source .venv/bin/activate
   ```

1. Create a compressed tarball (`.tar.gz`) from the `.venv`:

   ```bash
   ezpz tar-env
   # or, alternatively:
   # tar czvf .venv.tar.gz --directory .venv .
   ```

   Note:
   This will take a ~few minutes but only needs to be done _once_ (and can be
   done on CPU).
   After that, we can simply reuse the `.venv.tar.gz` for subsequent
   runs[^tarball].

1. Distribute the tarball to every node's `/tmp/`:

   ```bash
   ezpz yeet .venv.tar.gz   # faster at scale

   # alternatively, yeet the uncompressed `.venv/`:
   # ezpz yeet              # slower at scale
   ```

   <details closed><summary>Output</summary>

   ```bash
   #[05/04/26,09:08:42][x4302c2s3b0n0][~/a/f/p/s/torchtitan-ezpz][ezpz][?]
   ; ezpz yeet .venv.tar.gz
   [2026-05-04 09:08:53][I][utils/yeet_env:331:_maybe_apply_hsn_suffix] HSN interface available on 2/2 nodes (-hsn0 suffix)
   [2026-05-04 09:08:53][I][utils/yeet_env:1013:run] Source: /lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz/.venv.tar.gz (2.7G)
   [2026-05-04 09:08:53][I][utils/yeet_env:1014:run] Target: /tmp/.venv/ on 2 node(s)
   [2026-05-04 09:08:53][I][utils/yeet_env:1016:run]   local:  x4302c2s3b0n0 (rsync to /tmp/.venv/)
   [2026-05-04 09:08:53][I][utils/yeet_env:1019:run]   remote: x4302c2s4b0n0-hsn0
   [2026-05-04 09:08:53][I][utils/yeet_env:1054:run] Syncing (2 nodes)...

       ✓ x4302c2s3b0n0 (local, tar.gz (pre-built)) — 48.5s
       ✓ x4302c2s4b0n0-hsn0 — 20.1s
   [2026-05-04 09:10:02][I][utils/yeet_env:1334:run] Done in 68.9s

   To use this environment:
     deactivate 2>/dev/null
     source /tmp/.venv/bin/activate

   Then launch your training (from a shared filesystem path):
     cd /path/to/your/project
     ezpz launch python3 -m your_app.train

   Note: /tmp is node-local. Make sure your working directory
   is on a shared filesystem (e.g. Lustre) before launching,
   so all ranks can access data and outputs.
   [2026-05-04-091002] Command: ezpz yeet .venv.tar.gz
   took: 1 min. 14 s.
   ```

   </details>

1. Deactivate the _current_ `.venv` and activate the one we just created at
   `/tmp/.venv/`:

   ```bash
   deactivate && source /tmp/.venv/bin/activate
   ```

   ```bash
   $ which python3
   /tmp/.venv/bin/python3
   ```

1. Launch training as usual; `ezpz launch` respects `$VIRTUAL_ENV`, so it picks
   up the `/tmp/` venv automatically:

   ```bash
   ezpz launch python3 -m torchtitan.experiments.ezpz.train \
       --module=ezpz.agpt \
       --config=agpt_2b \
       --training.steps=10 \
       --checkpoint.no-enable \
       --training.local-batch-size=2
   ```

> [!IMPORTANT]
> `/tmp/` is node-local.
> Keep your project directory (data, checkpoints) on a shared filesystem so
> all ranks can read inputs and write outputs.

[^tarball]: If we install additional packages or make changes to the `.venv/`, we will need
to repeat this step and create a new `.venv.tar.gz` to capture these.
