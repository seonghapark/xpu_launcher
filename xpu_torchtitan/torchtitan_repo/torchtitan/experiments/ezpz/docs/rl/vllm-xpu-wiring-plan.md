# Wiring vLLM-XPU into ezpz/rl — architecture + sequencing plan

**Date:** 2026-06-13
**Author:** notes drafted from a feasibility survey
**Sibling docs:**
[`vllm-xpu-investigation.md`](vllm-xpu-investigation.md) (vLLM-XPU stack
validation, 2026-06-10).

## TL;DR

Two viable architectures. Recommend **TRL `vllm_mode="server"`** for
ezpz/rl, with Monarch+TorchStore as a "fast path" upgrade later.
Smoke install + 5-min test sequence at the bottom of this doc.

| Option | Trainer side | Generator side | Weight sync | Effort | Risk |
|---|---|---|---|---|---|
| **A. Monarch+TorchStore actors** | `RLTrainer` actor (upstream torchtitan, **not** TRL) | `VLLMGenerator` actor | `torchstore` (RDMA on CUDA → Gloo on XPU) | **1-2 weeks** | High — XPU untested for Monarch, requires rewriting ezpz/rl off TRL |
| **B. TRL `vllm_mode="server"`** | Existing `train_grpo.py` (TRL-based) | Standalone vLLM-XPU HTTP server | TRL's `WeightSyncWorkerExtension` (HTTP weight push, with optional XCCL collective fast-path) | **~2-3 days** | Low — TRL already has XPU code paths |

Reason for the rec: ezpz/rl is built on **TRL `GRPOTrainer`**, not on
upstream `torchtitan/experiments/rl/`'s custom `RLTrainer`. Adopting
Monarch would mean rewriting our GRPO loss, reward computation,
dataset handling, and per-prompt advantage logic — all currently
inherited from TRL. The Monarch actor model is excellent, but the
delta to integrate it is much larger than "wire the existing
generator to vLLM".

The two paths aren't exclusive — TRL server-mode can be a stepping
stone. Once it's running, we'll have a concrete production GRPO
trajectory using vLLM generation. If/when the weight-sync HTTP cost
becomes the bottleneck, we revisit the Monarch path with empirical
numbers in hand.

## What's already in place

1. **vLLM-XPU stack (validated)** —
   [`vllm-xpu-investigation.md`](vllm-xpu-investigation.md) documents
   the Option B sibling-venv setup:
   `venvs/vllm-test/` pinned to `torch==2.12.0+xpu` +
   `vllm==0.22.1` + `vllm-xpu-kernels==0.1.9.1`. End-to-end verified
   2026-06-10: loaded SFT'd `checkpoint-729-hf` in 5.6s, 48 GiB
   KV cache, correct answers to test prompts.
2. **TRL has the server-mode plumbing** — `GRPOConfig` exposes
   `use_vllm`, `vllm_mode`, `vllm_server_base_url`,
   `vllm_server_host`, `vllm_server_port`,
   `vllm_gpu_memory_utilization`, `vllm_tensor_parallel_size`, and
   sampling-correction knobs out of the box (TRL v1.5.1 installed in
   `.venv`).
3. **TRL ships `trl vllm-serve`** — already XPU-aware:
   `trl/scripts/vllm_serve.py` imports
   `from transformers import is_torch_xpu_available` and branches
   on `torch.xpu` for device properties (lines 70, 84, 97, 130, 139).
4. **`train_grpo.py` config stub** — `EzpzGRPOArgs.use_vllm: bool = False`
   is declared but not wired to anything yet.

## What's still missing

For **Option B (recommended)**:

1. Pass-through wiring of TRL's `vllm_*` config fields from
   `EzpzGRPOArgs` to `GRPOConfig`. Today only `use_vllm` is
   declared on the ezpz side; the rest of the knobs aren't exposed.
   ~30 LOC.
2. PBS submit script that launches a vLLM server on a dedicated tile
   (or set of tiles) BEFORE the trainer mpiexec, using `venvs/vllm-test/`.
   Pattern: `qsub` allocates N nodes, script `mpirun --np 1 --host
   <head_node> venvs/vllm-test/bin/python -m trl vllm_serve ...` &
   then `mpirun --np $((NRANKS-1)) --host <other_nodes>
   .venv/bin/python -m torchtitan.experiments.ezpz.train ...`.
3. Trainer config points at the server URL. Existing `train_grpo.py`
   flow continues, just with TRL hitting the server instead of
   in-process generation.
4. Smoke validation: 10-step run, confirm step-1 loss matches the
   in-process baseline within FP-summation noise, confirm wall-clock
   speedup is real (~3-5× expected at the EP=12 shape).

For **Option A (deferred)**:

5. Smoke install + actor test of torchmonarch + torchstore on XPU
   (this doc's section below).
6. Rewrite `train_grpo.py` against `RLTrainer` instead of
   `GRPOTrainer`. Reimplement GRPO loss + reward functions on top of
   torchtitan's training loop.
7. Adopt `VLLMGenerator` / `PolicyTrainer` actor pattern, wire
   TorchStore weight sync.

## Why upstream's actor pattern doesn't drop in

`torchtitan/experiments/rl/` is its own RL stack:

- **Trainer:** `RLTrainer` (custom torchtitan-native), not HF Trainer
  or TRL `GRPOTrainer`.
- **Generator:** `VLLMGenerator(Actor)` directly wrapping
  `vllm.LLMEngine`, not TRL's `vllm_serve` HTTP server.
- **Weight sync:** `torchstore` (`ts.get_state_dict`) for direct
  in-memory pulls, with RDMA fast path.
- **Orchestration:** `monarch.actor` with `HostMesh` / `ProcMesh` /
  `spawn_procs` — each actor lives in its own process with its own
  GPU mesh.

The endpoint surface is real:

| Actor | Endpoints |
|---|---|
| `VLLMGenerator` | `sync_log_step`, `generate`, `pull_model_state_dict`, `close` |
| `PolicyTrainer` | `close`, `sync_log_step`, `forward_backward`, `optim_step`, `save_checkpoint`, `push_model_state_dict` |

Pretty clean if you're starting from scratch. But ezpz/rl already has
**~700 LOC of train_grpo.py** built around TRL — switching to
upstream's stack means deprecating most of that, including patterns
that survived the 4 PBS + 8 autoretry production GRPO run.

## Monarch on XPU — feasibility (yes, but untested)

**Findings from the survey:**

- `torchmonarch` ships pre-built pip wheels (`pip install
  torchmonarch`) that don't hard-dep torch.
  [PyPI](https://pypi.org/project/torchmonarch/) v0.5.0
  released 2026-05-20; nightlies continue.
- `MONARCH_GPU_PLATFORM=none` (build-time flag) drops the CUDA/NCCL/RDMA
  tensor-engine extras while keeping the actor framework intact.
  Actor processes can still see and use GPUs (XPU) through their own
  torch stack — Monarch doesn't gate that.
- `USE_TENSOR_ENGINE=0` strips even further (actors-only, no torch
  dep). Probably too aggressive — we want tensor-engine for state-dict
  transport, just over Gloo instead of NCCL.
- `torchstore` has 5 transport backends; **Gloo** and **Monarch RPC**
  work without CUDA/RDMA. Override via
  `LocalRankStrategy(default_transport_type=TransportType.Gloo)`.
- **Nobody's tested any of this on XPU.** README doesn't mention XPU.
  Risk areas: (a) NCCL-only code paths inside the tensor engine that
  don't gate on `MONARCH_GPU_PLATFORM`, (b) `is_rdma_available()`
  assumptions, (c) actor process startup picking up the wrong venv
  if XPU init is import-order-sensitive.

**Bottom line:** feasible enough to put effort into if Option B's
HTTP weight-sync turns out to be a real bottleneck, but not the
right first investment.

## Implementation sequence for Option B

### Phase 1: Smoke the vLLM-XPU stack at scale (~30 min)

Validate that `venvs/vllm-test/` still works on current Sunspot env
and a fresh allocation (the 2026-06-10 verification was on a
specific node; might have drifted with the same system shifts that
broke other XPU paths).

```bash
# On a compute node with at least 1 XPU tile visible:
source /lus/flare/projects/datascience/foremans/projects/saforem2/torchtitan/venvs/vllm-test/bin/activate
python -c "from vllm import LLM, SamplingParams; \
    llm = LLM(model='outputs/sft/aurora2b-sophiag-tulu-mix-32n-gbs6144/checkpoint-729-hf', \
              tensor_parallel_size=1, gpu_memory_utilization=0.5, enforce_eager=True); \
    print(llm.generate(['What is 3 + 7 + 2?'], SamplingParams(max_tokens=32))[0].outputs[0].text)"
```

If that prints something reasonable, the stack's still alive. If it
crashes, re-run the [`vllm-xpu-investigation.md`](vllm-xpu-investigation.md)
install steps first.

### Phase 2: TRL `vllm-serve` standalone test (~30 min)

Bring up `trl vllm-serve` from the `.venv` (NOT vllm-test — TRL is in
.venv). The server itself can run anywhere; the issue is the
torch version pinning. Since `trl vllm-serve` imports vllm at module
level, it needs vllm available. Options:

- **Option B.1 (cleanest):** Run `trl vllm-serve` from `venvs/vllm-test/`
  with `PYTHONPATH` pointing at `.venv`'s trl package. Avoids
  installing trl twice. Test:
  ```bash
  source venvs/vllm-test/bin/activate
  PYTHONPATH=/lus/.../.venv/lib/python3.14/site-packages \
    python -m trl.scripts.vllm_serve \
      --model outputs/sft/aurora2b-sophiag-tulu-mix-32n-gbs6144/checkpoint-729-hf \
      --tensor_parallel_size 1 --host 0.0.0.0 --port 8000 \
      --gpu_memory_utilization 0.5 --enforce_eager True
  ```
- **Option B.2:** Install trl into vllm-test via `uv pip install --no-deps trl`.
  Simpler but adds a second copy.

Verify with `curl localhost:8000/health` from the same node.

### Phase 3: Wire TRL config pass-through in train_grpo.py (no code needed!)

**Update from initial survey:** `EzpzGRPOConfig` in `train_grpo.py`
already subclasses `trl.GRPOConfig`. **All of TRL's `vllm_*` fields are
inherited automatically**, including `vllm_mode`, `vllm_server_base_url`,
`vllm_server_host`, `vllm_server_port`, `vllm_gpu_memory_utilization`,
`vllm_tensor_parallel_size`, `vllm_server_timeout`, etc.

So no code changes needed in `train_grpo.py` for the config pass-through.
The user can pass `--use_vllm --vllm_mode=server
--vllm_server_base_url=http://<server-host>:8000` on the CLI today,
and TRL handles the rest internally.

(The `EzpzGRPOArgs.use_vllm` stub at line 201 is leftover dead code
from an earlier design — it can be removed in a cleanup commit.)

### Phase 4: PBS submit script (~half day)

`scripts/grpo/aurora2b_sft_arithmetic_vllm.sh` analog of the existing
non-vLLM submit script, but:

- Splits allocation: 1 tile (or 1 node) for vLLM server, rest for
  trainer ranks.
- Launches server first, waits for `/health` to pass.
- Launches trainer with `--use_vllm --vllm_mode=server --vllm_server_base_url=http://...`.
- Trap to kill server on trainer exit.

### Phase 5: Smoke + production (~half day)

- 4N smoke at GBS=24 for 10 steps, compare loss to in-process baseline.
- If within FP noise, queue an 8N production run mirroring the
  `aurora2b_sft_arithmetic_8n` config.

## Option B install + smoke commands (ready to run)

User-runnable commands. **Don't auto-run** — install touches the venv
and the CLAUDE.md rule #5 ("never `pip install` torch or its deps")
requires explicit user authorization for any package not already
declared.

```bash
# Verify baseline (.venv torch + XPU)
.venv/bin/python3 -c "import torch; print('torch:', torch.__version__); \
    print('xpu:', torch.xpu.is_available())"

# Verify vllm-test venv still has its stack
venvs/vllm-test/bin/python -c "import vllm, torch; \
    print('vllm:', vllm.__version__); print('torch:', torch.__version__)"

# Bring up server (from a compute node with XPU access)
source venvs/vllm-test/bin/activate
PYTHONPATH=/lus/flare/projects/datascience/foremans/projects/saforem2/torchtitan/.venv/lib/python3.14/site-packages \
  python -m trl.scripts.vllm_serve \
    --model outputs/sft/aurora2b-sophiag-tulu-mix-32n-gbs6144/checkpoint-729-hf \
    --tensor_parallel_size 1 --port 8000 \
    --gpu_memory_utilization 0.5 --enforce_eager True &

# Wait for health
until curl -sf http://localhost:8000/health > /dev/null; do sleep 2; done

# Sanity request
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is 3 + 7 + 2?", "max_tokens": 32, "model": "outputs/sft/aurora2b-sophiag-tulu-mix-32n-gbs6144/checkpoint-729-hf"}'
```

If those work, Phase 3 onwards is just code (no more environment
detective work needed).

## Option A install commands (for the future Monarch path)

**Per CLAUDE.md rule #5: only run with explicit user authorization.**

```bash
# Pre-built wheels — no CUDA toolkit needed at install time
uv pip install --python .venv/bin/python3 --no-deps --no-cache --link-mode=copy torchmonarch
uv pip install --python .venv/bin/python3 --no-deps --no-cache --link-mode=copy \
    "git+https://github.com/meta-pytorch/torchstore.git@main"
uv pip install --python .venv/bin/python3 --no-deps --no-cache --link-mode=copy \
    pygtrie portpicker

# Verify torch + XPU still work after each install
.venv/bin/python3 -c "import torch; assert '+xpu' in torch.__version__; print('OK', torch.__version__)"

# Verify monarch imports
.venv/bin/python3 -c "from monarch.actor import Actor, current_rank, endpoint; print('monarch OK')"

# Verify torchstore imports
.venv/bin/python3 -c "import torchstore as ts; print('torchstore OK')"
```

If any of those fail (especially the post-install torch check —
silent CUDA build swap is the failure mode CLAUDE.md rule #5
warns about), `backup .venv` and rebuild from the
`.venv.tar.gz-20260607-185903` snapshot or similar.
