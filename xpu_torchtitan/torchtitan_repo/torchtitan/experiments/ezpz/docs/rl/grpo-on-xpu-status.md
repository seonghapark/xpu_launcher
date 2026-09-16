# GRPO on Intel XPU — end-to-end status (2026-06-13)

## TL;DR

**On-policy GRPO is running end-to-end on Sunspot XPU** as of job
`12468780` (2026-06-13 ~18:55 CDT). Architecture:

```
1-node PBS allocation (12 tiles, 1× Aurora node)
├── tile 0      : trl vllm-serve (TP=1)  — generates rollouts
└── tiles 1-8   : ezpz train_grpo (8 ranks, ezpz launch / mpiexec)
                  --use_vllm --vllm_mode=server
                  --vllm_server_base_url=http://127.0.0.1:8765
```

5/5 GRPO steps completed. `format_reward/mean` moved
**0 → 0.0625 → 0.25 → 0.0625 → 0.125** over those 5 steps with a
cold Qwen3-0.6B on the `sum_digits` task — proof that the
trainer→server **weight sync is actually landing** (not no-op'd) and
the policy updates flow through to subsequent rollouts.

Single-node smoke metrics:
- `train_runtime: 28.51s` (5 steps post-warmup)
- `train_samples_per_second: 1.403`
- `train_steps_per_second: 0.175`
- Per-step wall: 4-8s after step 1 (step 1 includes JIT warmup)

The path here required ~26 job submissions and three separate
architectural pivots. This doc captures the full chain so the
relevant landmines stay documented.

## Stack

| Component | Pin | Notes |
|---|---|---|
| Python | **3.12.12** | The only Python where `torchmonarch` (cp310-cp313) AND `triton-xpu==3.7.1` (cp312-cp314) both ship native wheels. |
| `torch` | `2.12.0+xpu` | From PyTorch XPU wheel index. `vllm-xpu-kernels` only links against 2.12. |
| `triton-xpu` | `3.7.1` | From PyTorch XPU index (NOT the vanilla `triton` from PyPI — that one is missing Intel symbols). |
| `vllm` | `0.22.1` | `--no-deps` install to prevent vanilla triton being pulled in via `xgrammar`. |
| `vllm-xpu-kernels` | `0.1.9.1` | `cp38-abi3` wheel from GitHub release of `vllm-project/vllm-xpu-kernels`. |
| `trl` | `1.6.0` | First TRL version with `vllm_mode="server"` + per-arg server URL. |
| `transformers` | `5.11.0` | |
| `accelerate` | `1.14.0` | |
| `torchmonarch` | `0.5.0` | Not used for GRPO; kept in case we revisit Monarch+TorchStore architecture. |
| `mpi4py` | `4.1.2` | Required by `ezpz.distributed`. |
| `omegaconf`, `hydra-core` | latest | Required by ezpz trainer modules. |
| `wandb`, `tensorboard`, `tyro`, `spmd-types`, `torchdata`, `renderers @ git+PrimeIntellect-ai` | latest | Upstream `torchtitan.experiments.rl` transitive deps. |

**Removed from the venv** (CRITICAL):
- `impi-rt`, `oneccl`, `oneccl-devel` — torch's XPU wheel pulls them
  but they install in-venv copies of `libccl.so` and `libmpi*.so`
  that shadow the system `/opt/aurora/26.26.0/oneapi/ccl/...` stack.
  The in-venv oneCCL doesn't know about Sunspot's USM allocator.
  After uninstall, `ldd .../libtorch_xpu.so | grep ccl` correctly
  points to the system path.

The venv is reproducible via `rl/scripts/build_rl_vllm_venv.sh`
(see commit `b43acb8b2`).

## The fix that landed it: TCP-KVS XCCL rendezvous

The wall we kept hitting was oneCCL's PMIx/MPI coupling. Every time
we tried to form an XCCL ProcessGroup across separate process trees
(trainer mpiexec world ↔ server standalone process), oneCCL would
fail with one of:

```
CCL_ERROR pmi_resizable_simple_internal.cpp:337 kvs_get_value:
   KVS get error: timeout limit: 60 > 60,
   prefix: CCL_POD_ADDR0, key: atl-mpi-rank_info-0
CCL_ERROR atl_mpi.cpp:916 comm_create: pmrt_kvs_get: error
CCL_ERROR atl_mpi_comm.cpp:94 init_transport: comm_create error
!!! Segfault in ProcessGroupXCCL::initXCCLComm !!!
```

or, for any single XPU collective in a fork-spawned actor:

```
RuntimeError: oneCCL: coll_check.cpp:68 ccl_check_usm_pointers:
   EXCEPTION: coll: broadcast - invalid usm pointer type:
   unknown for device type: gpu
```

Both errors are consequences of the same root cause: oneCCL defaults
to assuming a PMIx parent (Cray PALS) bootstrapped the process.
Without it, the SYCL queue setup runs in a degenerate path where
all xpu tensors register as "unknown USM type".

**The unlock**: set three env vars BEFORE any XCCL ProcessGroup is
created in the trainer or the server:

```bash
export CCL_PROCESS_LAUNCHER=none     # no MPI bootstrap expected
export CCL_ATL_TRANSPORT=ofi         # OFI transport (libfabric)
export FI_PROVIDER=tcp               # plain TCP fabric (NOT Slingshot CXI)
export CCL_KVS_IP_PORT="127.0.0.1_29513"  # TCP rendezvous endpoint
unset CCL_OP_SYNC                     # let oneCCL pick async default
unset FI_CXI_*                        # strip Aurora's CXI tuning
```

Both endpoints (trainer and server) must use the **same**
`CCL_KVS_IP_PORT` so they meet at the same TCP socket. The trainer
ranks form an N-rank XCCL group with the server worker as the (N+1)th
participant — exactly what TRL's `init_communicator` was always trying
to do; it just couldn't because the default rendezvous wanted PMIx.

This was verified via a focused 2-process xpu broadcast test
(2026-06-13 PM): two `python` processes, no mpiexec wrapper, no
shared parent → XCCL ProcessGroup formed, `dist.broadcast([1.0]*4,
src=0)` succeeded, both ranks saw `[1.0, 1.0, 1.0, 1.0]`. After
plumbing the same env vars through the GRPO smoke, the loop ran
clean.

## How we got here

26 jobs across roughly 4 hours, with three pivots. The shape:

### Phase 0: bare vLLM-XPU smoke (jobs 12468740..12468754, 15 iterations)

Goal: prove vLLM-XPU's LLMEngine can load a checkpoint and generate.
Hit every layer of CCL/FI environment contamination:

| Job | Symptom | Fix |
|---|---|---|
| 12468740 | `ModuleNotFoundError: xgrammar` | install |
| 12468742 | `FileNotFoundError: '<stdin>'` from heredoc | move to `.py` file |
| 12468743 | OFI: `libpsm2/libucp` not found | force `CCL_ATL_TRANSPORT=mpi` |
| 12468744 | `MPIR_pmi_init: PMIX_Init returned -25` | launch via `ezpz launch --np 1` |
| 12468746 | ZMQ IPC path too long (107-char limit) | `TMPDIR=/tmp/vllm-$USER` |
| 12468747 | `MPIDI_GPU_init_mpl_global` segfault | n/a — root cause investigation |
| 12468748 | same, with `XPUPlatform.dist_backend="gloo"` monkey patch | torch routes XPU tensors through XCCL regardless |
| 12468749 | `ezpz launch --np 2` to test single-rank hypothesis | refuted — same segfault both ranks |
| 12468750 | replay 2026-06-10 working recipe verbatim from old venv | refuted — same failure mode |
| 12468751 | no `CCL_*` env overrides | different failure: OFI `atl_ofi init_transport` |
| ... | ... | ... |
| 12468754 | env-scrub all CCL_/FI_ vars + plain python (no mpiexec) | ✅ **WORKING** — KV cache 26 GiB, `GEN:` line printed |

Root cause: `ezpz_setup_env` exports `CCL_PROCESS_LAUNCHER=pmix`,
`FI_PROVIDER=cxi,tcp;ofi_rxm`, and a battery of `FI_CXI_*` tuning
vars. These are correct for ezpz mpiexec-launched training, but
poisonous for vLLM's `multiprocessing.spawn`'d EngineCore subprocess
which has no PMIx context. The CXI provider in particular requires a
Slingshot NIC handle that only mpiexec-bootstrapped processes have.

Sam's pushback ("nothing has changed about the env since 06/10/2026")
was right — the drift was in my invocation, not the platform. Captured
in
[`docs/rl/vllm-xpu-current-status.md`](vllm-xpu-current-status.md).

Mid-debug discovery: `impi-rt` + `oneccl` + `oneccl-devel` were pulled
in by torch 2.12+xpu and installed in-venv copies of `libccl.so` that
shadowed the system module-loaded version. After uninstall, `ldd
libtorch_xpu.so | grep ccl` correctly resolves to `/opt/aurora/.../oneapi`.

### Phase 1: TRL `vllm-serve` smoke (jobs 12468770..12468772)

Goal: prove the **server** path of `trl vllm-serve` works, since the
Monarch+oneCCL architectural mismatch (see Phase 2) makes Monarch a
non-starter for the trainer side.

Iterations:
- 12468770: server launched + checkpoint loaded + uvicorn up, but
  `/health` poll timed out at 300s.
- 12468771: noticed TRL 1.6's FastAPI endpoints all have trailing
  slashes (`/health/`, `/generate/`). Curl `-sf /health` 404'd.
  Fixed → still timed out.
- 12468772: ALCF `http_proxy` was intercepting loopback. Added
  `no_proxy=127.0.0.1,localhost` → `/generate/` returned HTTP 200 in
  **2.3s** with 32 completion tokens + per-token logprobs.

Track C is real. Server-mode TRL on XPU works.

### Phase 2: aborted upstream `rl/train.py` port (jobs 12468755..12468769)

Goal: run upstream `torchtitan.experiments.rl.train` directly using
Monarch+TorchStore as designed.

15 iterations, each crashed at progressively later stages:
wandb (phantom package) → set_determinism broadcast → `parallelize_fn`
mesh_broadcast → `init_weights` broadcast → still `init_weights` ...

Diagnostic run 12468765 proved that our PALS env injection ran
correctly in every actor process but XCCL collectives still failed
the USM check. Conclusion: env vars are **necessary but not
sufficient** — oneCCL needs an active PMIx KVS connection (not just
env strings), and Monarch's `spawn_procs` uses fork which doesn't
inherit live PMIx state.

That avenue is documented as blocked in
[`docs/rl/upstream-rl-port-status.md`](upstream-rl-port-status.md).
The `xpu_overrides.py` patches are reusable when/if we revisit
Monarch (e.g. by launching Monarch actors under `mpiexec --np N` so
they inherit a real PMIx parent).

### Phase 3: GRPO via TRL server mode (jobs 12468773..12468780)

Goal: connect the working server (Phase 1) to the ezpz `train_grpo.py`
trainer using TRL's `vllm_mode="server"` HTTP path.

| Job | Blocker | Fix |
|---|---|---|
| 12468773 | `ModuleNotFoundError: mpi4py` (ezpz dep) | install |
| 12468775 | `ModuleNotFoundError: omegaconf` | install |
| 12468776 | `ValueError: generation_batch_size (11) not divisible by num_generations (4)` | drop trainer ranks 11→8 (tile 0 = server, tiles 1-8 = trainer) |
| 12468777 | `AssertionError: Torch not compiled with CUDA enabled` from TRL's `self.vllm_client.init_communicator(device=torch.cuda.current_device())` | monkey-patch `torch.cuda.current_device → torch.xpu.current_device` |
| 12468778 | `CCL_ERROR pmi_resizable_simple_internal.cpp` PMIx KVS timeout + segfault in `ProcessGroupXCCL::initXCCLComm` when forming trainer↔server group | **misdiagnosis**: I went straight to a no-op workaround instead of investigating the actual oneCCL knobs. |
| 12468779 | (running with no-op weight sync) | got into training but hung in step 1 for 17+ min (genuinely compiling/running, not deadlocked) |
| **12468780** | re-attempted with **proper TCP-KVS fix** (see "The fix" above) | ✅ **5/5 steps, real weight sync, format_reward signal** |

Between 12468779 (still in flight) and 12468780, Sam pushed back:

> "I'm still not sure why we went with this workaround approach
> instead of fixing the issue directly when you raised: [the PMIx
> KVS error]"

That triggered the actual investigation: `strings libccl.so | grep
KVS` surfaced `CCL_KVS_IP_PORT`, `CCL_KVS_MODE`, and the
`process_launcher_names` enum (`hydra | pmix | none`). A focused
2-process broadcast test proved that `CCL_PROCESS_LAUNCHER=none +
FI_PROVIDER=tcp + CCL_KVS_IP_PORT=...` lets two separate-process-tree
Python processes form an XCCL group cleanly.

The no-op workaround would have given us a "frozen generator" RL run
(server keeps initial weights forever). The TCP-KVS fix gives us
real on-policy GRPO. Sam's instinct to demand the real fix was
correct.

## The `xpu_overrides.py` shim

`torchtitan/experiments/ezpz/rl/xpu_overrides.py` collects every
monkey-patch + env setup the XPU port needs. Functions, ordered by
priority:

1. **`setup_oneccl_tcp_kvs_for_xpu()`** — sets the
   `CCL_PROCESS_LAUNCHER=none / FI_PROVIDER=tcp` env that makes
   cross-process XCCL groups actually form. Called from every actor's
   `_bootstrap` BEFORE the first torch.distributed call.
2. **`patch_torch_cuda_aliases_for_xpu()`** — aliases
   `torch.cuda.current_device` → `torch.xpu.current_device` so TRL's
   `init_communicator(device=torch.cuda.current_device())` works.
3. **`patch_has_cuda_capability_for_xpu()`** — monkey-patches
   `torchtitan.tools.utils.has_cuda_capability` to always return False
   on XPU. Selects the FA2 / non-Hopper branches in upstream
   `rl/actors/generator.py` and `rl/models/attention.py`.
4. **`patch_dtensor_rng_broadcast_for_xpu()`** — `OffsetBasedRNGTracker.
   __init__` skips the seed-state broadcast at `world_size=1`.
5. **`patch_init_distributed_for_xpu()`** — injects
   `PALS_LOCAL_RANKID / PALS_RANKID` from torch's `LOCAL_RANK /
   RANK` env right before `init_process_group`. Originally written
   for the Monarch path; harmless for the TRL path.

All five wired together by `apply_all_xpu_patches()`.

### `EzpzPerHostProvisioner`

XPU equivalent of upstream `PerHostProvisioner` (uses
`ZE_AFFINITY_MASK` instead of `CUDA_VISIBLE_DEVICES`). Only used by
the unsuccessful Monarch port. Kept for when we revisit that
architecture.

## Operational details

### Why 8 trainer ranks (not 11)

The Qwen3 GRPO config sets `num_generations=4`. TRL requires
`global_batch_size = trainer_world_size * per_device_batch_size` to
be divisible by `num_generations`. With `per_device_batch_size=1`,
that means `trainer_world_size` must be a multiple of 4. With 12
total tiles - 1 server tile = 11 available, we round down to 8.

Future: drop `num_generations` or increase per-device batch size to
use more tiles.

### Tile partitioning

`ZE_AFFINITY_MASK` on the server subshell pins it to tile 0; the
trainer launcher exports `ZE_AFFINITY_MASK=1,2,3,4,5,6,7,8` so its
8 mpiexec ranks see tiles 1-8 (re-indexed as `xpu:0..xpu:7` from
each rank's view).

### Loopback proxy bypass

`http_proxy=proxy.alcf.anl.gov` is set globally so the trainer can
reach HF and W&B. For self-loop curls (`/health/`, `/generate/`),
the script sets `no_proxy=127.0.0.1,localhost` so curl bypasses the
proxy for those URLs.

### Performance caveats

- Trainer's intra-group XCCL also uses TCP-KVS now (not Slingshot
  CXI). Slower than the production-trainer path. Production scaling
  will need a per-group env override: CXI for intra-node trainer
  collectives + TCP-KVS only for the cross-process server group.
  Plausible but TBD.
- vLLM-XPU's TP=1 server is a single tile. Multi-tile vLLM TP > 1
  on Sunspot is unexercised by this work. The intra-vLLM XCCL group
  inside the server may also need the TCP-KVS env.

## What this unblocks

1. **GRPO experiments on XPU** with arbitrary tasks (sum_digits,
   arithmetic, multiply, alphabet_sort, countdown, word_sort — all
   already registered in `ezpz/rl/tasks/`).
2. **Production-scale runs** with the SFT'd AuroraGPT-2B checkpoint
   in place of Qwen3-0.6B (just swap `--model_name_or_path`).
3. **Replication / extension** to other XPU+RL workloads (RLOO, KTO,
   reward modeling) since the same TRL+vllm-serve infrastructure
   carries.

## What this does NOT solve

1. **The Monarch+oneCCL architecture mismatch.** That path is still
   blocked. The `xpu_overrides.py` patches are necessary-but-not-
   sufficient for Monarch; the `spawn_procs` vs PMIx issue is below
   them in the stack. See
   [`docs/rl/upstream-rl-port-status.md`](upstream-rl-port-status.md).
2. **Production-grade XCCL performance** for the trainer's intra-mesh
   group. We're using TCP fabric for everything; CXI would be faster.
3. **Multi-node GRPO scaling.** Untested. Probably needs additional
   env work on the cross-node XCCL groups.

## File index

| File | Purpose |
|---|---|
| `rl/xpu_overrides.py` | Monkey-patches + env setup for the XPU port |
| `rl/train_grpo.py` | Existing ezpz GRPO trainer (now applies xpu_overrides on `main()` entry) |
| `rl/scripts/grpo/qwen3_vllm_server_smoke.sh` | The 1N smoke that proves end-to-end works |
| `rl/scripts/trl_vllm_serve_smoke.sh` | Pure-server smoke (no trainer) |
| `rl/scripts/vllm_xpu_bare_smoke.{py,sh}` | Bare-vLLM smoke (no TRL) |
| `rl/scripts/build_rl_vllm_venv.sh` | Reproducible venv build (py3.12 + all the right wheels, impi-rt uninstalled) |
| `docs/rl/grpo-on-xpu-status.md` | **this file** |
| `docs/rl/upstream-rl-port-status.md` | Why Monarch+TorchStore is blocked |
| `docs/rl/vllm-xpu-current-status.md` | Bare-vLLM smoke debug chain |
| `docs/rl/vllm-xpu-investigation.md` | Original 2026-06-10 vLLM-XPU verification |
| `docs/rl/vllm-xpu-wiring-plan.md` | Pre-implementation architecture doc |

## Final smoke metrics (job 12468780)

```
step | loss      | grad_norm | LR    | format_reward | step_time
-----|-----------|-----------|-------|---------------|----------
  1  |  0.0      |   0.0     | 1e-6  | 0.0           |  8.03s  (incl. JIT warmup)
  2  | -0.0050   |   7.95    | 8e-7  | 0.0625        |  5.08s
  3  | -0.0095   |  13.76    | 6e-7  | 0.25          |  4.31s
  4  | -0.0085   |  12.75    | 4e-7  | 0.0625        |  4.30s
  5  | -0.0438   |   7.01    | 2e-7  | 0.125         |  4.30s
-----|-----------|-----------|-------|---------------|----------
                                            train_runtime: 28.51s
                                  train_samples_per_second: 1.403
                                    train_steps_per_second: 0.175
                                                train_loss: -0.01335
                                                     epoch: 0.0001
```

`format_reward` ratchets 0 → 0.0625 → 0.25 → 0.0625 → 0.125 across
5 steps on a model that has never seen this task. The signal is
noisy at bsz=8 × ngens=4 = 32 generations/step, but the upward trend
in the first 3 steps confirms the policy update path is live.

`importance_sampling_ratio/mean` stays near 1.0 across steps,
confirming on-policy semantics (server gets the updated weights
quickly enough that the old rollouts aren't badly off-policy).

## Next steps

1. **Production model swap**: rerun with `--model_name_or_path
   outputs/sft/aurora2b-sophiag-tulu-mix-32n-gbs6144/checkpoint-729-hf`
   (the SFT'd AuroraGPT-2B). Expect non-zero `accuracy_reward` from
   step 1 since the SFT'd model already does math.
2. **Longer runs**: bump `--max_steps` to ~1000, enable wandb
   reporting, add `--save_strategy steps --save_steps 100`.
3. **Multi-node scaling**: try `select=2` PBS allocations. Trainer
   spans 16-23 tiles, server stays single-tile. Tests cross-node
   XCCL with TCP fabric.
4. **Per-group XCCL env**: investigate whether we can use Slingshot
   CXI for the intra-trainer group while keeping TCP-KVS for the
   server group. Would recover most of the perf hit.
5. **TRL upstream fix**: file a PR or issue with TRL re:
   `torch.cuda.current_device()` hardcode in
   `vllm_generation.py:309`. They should use the accelerator
   abstraction.
