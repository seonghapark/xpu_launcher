# Upstream `torchtitan.experiments.rl.train` port status (2026-06-13)

## TL;DR

The XPU compatibility layer needed to run upstream's GRPO loop is
*almost* trivial — but a deeper architectural mismatch between
Monarch's `spawn_procs` (fork-based) and oneCCL's PMIx-coupled
USM allocator blocks the trainer side. The generator side works.

**What works:**
- Bare vLLM-XPU end-to-end (`vllm_xpu_bare_smoke.sh`, job 12468754)
- Monarch actor framework on XPU (`monarch_smoke.sh`, job 12468739)
- Upstream `rl/`'s full import chain on XPU (with patches in
  `xpu_overrides.py`)
- All 4 generator actors initializing vLLM-XPU `LLMEngine` with TP=4

**What doesn't:**
- Trainer actors hitting any `torch.distributed` collective
  (broadcast / allreduce). Every attempt fails oneCCL XCCL's
  `ccl_check_usm_pointers` validation:
  > `RuntimeError: oneCCL: coll_check.cpp:68
  > ccl_check_usm_pointers: EXCEPTION: coll: broadcast / allreduce
  > - invalid usm pointer type: unknown for device type: gpu`

## XPU porting layer (working)

`torchtitan/experiments/ezpz/rl/xpu_overrides.py` provides:

1. **`has_xpu_kernels()`** — monkey-patches
   `torchtitan.tools.utils.has_cuda_capability` to always return
   `False` on XPU. Selects the FA2 (non-Hopper) branches in
   `rl/actors/generator.py:465` and `rl/models/attention.py:70`.

2. **`EzpzPerHostProvisioner`** — XPU equivalent of upstream
   `PerHostProvisioner`. Partitions Sunspot/Aurora tiles via
   `ZE_AFFINITY_MASK` instead of `CUDA_VISIBLE_DEVICES`.

3. **`patch_dtensor_rng_broadcast_for_xpu()`** —
   `OffsetBasedRNGTracker.__init__` skips the seed-state broadcast at
   `world_size=1`.

4. **`patch_init_distributed_for_xpu()`** —
   - Sets `PALS_LOCAL_RANKID` / `PALS_RANKID` from torch's
     `LOCAL_RANK` / `RANK` right before `init_process_group`
   - Sets `ZE_AFFINITY_MASK` and other static PALS env in the
     actor `_bootstrap` callable

5. **`_bootstrap`** — eagerly imports `torch.distributed.checkpoint`
   to dodge a torchstore circular-import race during actor setup.

6. **`venvs/rl-vllm/`** built via `build_rl_vllm_venv.sh`:
   - Excludes `impi-rt` / `oneccl` / `oneccl-devel` pip packages
     (their in-venv copies shadow `/opt/aurora/.../oneapi/ccl`'s
     libccl with broken USM-allocator versions)
   - Uses py3.12 (the intersection where both `torchmonarch` cp312
     and `triton-xpu==3.7.1` cp312 wheels exist)

## The core blocker: Monarch fork-spawn ↔ PMIx

Diagnostic confirmed (job 12468765) that our env injection runs
correctly in every actor process:
- `PALS_LOCAL_RANKID` matches the actor's mesh-local rank
- `PALS_RANKID` matches the actor's global rank
- `PALS_LOCAL_SIZE` matches the actor mesh size
- `ZE_AFFINITY_MASK` correctly partitions tiles

**Yet every XCCL collective still fails the USM pointer check.**

We confirmed via baseline that XCCL works fine on plain xpu tensors
when launched directly with `mpiexec --np 2`:
```
$ mpiexec --np 2 python -c '
    import torch, torch.distributed as dist
    dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    torch.xpu.set_device(int(os.environ["LOCAL_RANK"]))
    t = torch.zeros(4, device=f"xpu:{rank}").fill_(rank+1.0)
    dist.all_reduce(t)
    print(rank, t.tolist())'
[0] AFTER: [3.0, 3.0, 3.0, 3.0]
[1] AFTER: [3.0, 3.0, 3.0, 3.0]
```

But **the same allreduce fails when the process was spawned by
Monarch's `spawn_procs`** — even when launched under an outer
`mpiexec --np 1` wrapper that does provide PMIx env to the
controller.

### Why env-only injection isn't enough

The PMIx environment variables (`PALS_*`, `PMIX_*`) are sentinels
that point to **shared mmap segments** owned by the PMIx-launching
process (e.g. `mpirun`/`palsd`). When oneCCL sees those vars, it
attempts to call into PMIx's KVS via those segments to coordinate
USM allocator setup across ranks.

When Monarch forks an actor process, the child inherits the env
vars but the segments they reference were registered for the
*parent's* PID. The child can't open them; oneCCL falls back to a
degenerate state where the USM allocator isn't properly initialized,
and any tensor it sees gets classified as "unknown" USM type.

This is structural — wrapping the controller in mpiexec doesn't
change the fork-child relationship inside Monarch.

## Things we tried (all failed)

| # | Approach | Outcome |
|---|---|---|
| 1 | `--debug.seed 42` (skip seed broadcast) | passes; next broadcast at parallelize_fn |
| 2 | `tensor-parallel-degree=1` (skip mesh broadcast) | passes; next broadcast at init_weights |
| 3 | Patch `OffsetBasedRNGTracker` to skip at ws=1 | passes; init_weights still broadcasts at ws>1 |
| 4 | Inject `PALS_LOCAL_RANKID` / `PALS_RANKID` / `PALS_LOCAL_SIZE` | env injection runs correctly, USM check still fails |
| 5 | Wrap launcher in `mpiexec --np 1` for PMIx parent | USM check still fails; PMIx state doesn't fork-inherit |
| 6 | Force `xpu→gloo` backend in default map | passes USM; Gloo can't broadcast xpu tensors |

## Paths forward

### A. Replace Monarch `spawn_procs` with mpiexec-launched ranks (correct)

Architecture change: launch the trainer with `mpiexec --np 2` and
the generator with `mpiexec --np 4` as separate MPI worlds. Have
Monarch act as a controller actor that talks to *already-running*
mpiexec-launched processes via TCP/RPC instead of forking them.

This is how production HPC + actor frameworks normally compose, and
it's how `torchtitan/experiments/rl` is presumably intended to run
on CUDA clusters with NCCL (which has similar PMIx requirements on
some networks).

Requires:
- New entrypoint that splits the launch into 3 mpiexec calls
  (controller, trainer, generator)
- TCP/RPC bootstrap between Monarch controller and the worker MPI
  groups
- vLLM-side: it already uses an "external launcher" pattern which
  is PMIx-aware. Should work once the worker procs are mpiexec-spawned.

### B. Use Gloo with a CPU-staging copy layer (slow but works today)

Wrap DTensor's collective dispatch so xpu tensors are `.to("cpu")`
before broadcast/allreduce, then `.to("xpu")` back. This means every
collective hits a 2x CPU↔XPU copy. Workable for the trainer's
infrequent state-sync collectives, painful for any hot loop.

### C. Stay with TRL `vllm_mode="server"` instead

The bare vLLM smoke + the env-scrub fix means we can stand up a TRL
GRPO loop that uses an external vLLM server (no Monarch on the
trainer side). The trainer runs as our normal ezpz trainer process
(mpiexec-launched, with working XCCL); the generator is the bare
vLLM-XPU server we already validated. TRL handles weight sync over
HTTP. Lower throughput than Monarch+TorchStore RDMA, but doesn't
depend on solving the Monarch+PMIx mismatch.

## Recommendation

(C) is the fastest path to a working RL loop on XPU and matches
what was already in the wiring plan as Track 2. (A) is the
architecturally correct next step if/when we want the Monarch
+TorchStore performance.

The XPU porting layer (`xpu_overrides.py`) is reusable for either —
the `has_cuda_capability` patch and `EzpzPerHostProvisioner` apply
regardless of how the worker processes are launched.

## Update 2026-06-13 deeper eve: torch 2.13 nightly probe — untested

Sam suggested trying torch 2.13 nightly to see if newer xpu allocator
fixes the DTensor USM problem. Built `venvs/torch213-test/` (py3.12
+ torch 2.13.0.dev20260611+xpu) but ran into stack-version mismatch:

- `torch.xpu` import fails because the system
  `/opt/aurora/26.26.0/oneapi/compiler/latest/lib/libur_loader.so.0`
  (loaded via `ezpz_setup_job`'s `module load oneapi/release/2025.3.1`)
  is missing symbol `urDeviceWaitExp` that torch 2.13 needs.
- Skipping the module load fails differently: `pyzes` hardcodes
  `/usr/lib/x86_64-linux-gnu/libze_loader.so.1` (Debian path) but
  Sunspot has it at `/usr/lib64/libze_loader.so.1`.

This is fixable (symlink + careful LD_LIBRARY_PATH) but each round
takes minutes and the next layer of breakage (vllm-xpu-kernels
0.1.9.1 is built against torch 2.12 → likely C++ ABI mismatch like
the original 2026-06-10 `c10::impl::cow::materialize_cow_storage`
problem) probably blocks the actual test we want to run.

**Practical answer**: deferring the torch 2.13 path. If we revisit,
the recipe is:
1. Build `venvs/rl-vllm-torch213/` (py3.12 + torch
   2.13.0.dev20260611+xpu) with `module unload` first
2. Symlink `/usr/lib/x86_64-linux-gnu/libze_loader.so.1` →
   `/usr/lib64/libze_loader.so.1` (or set
   `LD_LIBRARY_PATH=/usr/lib64`)
3. Pin `triton-xpu==3.7.2+git5fcc14d9` (matches torch 2.13)
4. Build vllm-xpu-kernels from source against torch 2.13 (HEAD of
   vllm-project/vllm-xpu-kernels)
5. Re-run the Monarch GRPO smoke; if DTensor's USM issues are
   resolved, we get the full Monarch+TorchStore path on XPU.

Track C (TRL+vllm-serve, py3.12 + torch 2.12) is the live production
path until then.

## Update 2026-06-13 deep eve: extensive Monarch-port investigation (jobs 12468783..12468794)

After Track C (TRL+vllm-serve) landed working GRPO end-to-end, took
a serious run at unblocking the Monarch+TorchStore path. 12 iterations
each pushed the failure to a new layer:

| Job | Discovery | Fix added |
|---|---|---|
| 12468783-784 | MASTER_ADDR/RANK/LOCAL_RANK env all correct at init_distributed | (diagnostic only) |
| 12468785 | per-actor ZE_AFFINITY_MASK needed (was 0,1 for both trainer ranks) | parse HYPERACTOR_PROCESS_NAME |
| 12468786-788 | Monarch's HYPERACTOR_PROCESS_NAME='anon-N' gives per-actor rank | extract N as rank_in_mesh |
| 12468789 | trainer.__init__ crashes: xpu:1 out of range | force LOCAL_RANK=0 + PALS_*_RANKID=rank_in_mesh in _bootstrap |
| 12468790 | Monarch RESETS LOCAL_RANK between _bootstrap and trainer.__init__ | (diagnostic confirmed) |
| 12468791 | set_device patched to pin device 0 | torch.xpu.set_device→0 |
| 12468792 | torch.device patch broke type unions (`int \| device \| None`) | reverted |
| 12468793 | os.environ wrapper for LOCAL_RANK reads | patched __getitem__/get to always return "0" |
| 12468794 | RNG broadcast → fixed; now buffer init broadcast also fails | rewrote _get_device_state to use torch.empty(device=) for USM |

**Net result**: every DTensor broadcast on XPU under Monarch's spawn
pattern still fails the USM check, regardless of how much we sanitize
the env. The RNG-state broadcast was just the first; `init_states →
distribute_tensor → _make_replicate_tensor → mesh_broadcast` is the
next, and there are more.

The 2-process bare baseline (`mpiexec --np 2 python ... torch.zeros
(device='xpu:0'); dist.broadcast(t)`) works perfectly. The same
broadcast inside DTensor's Replicate-placement materialization under
Monarch fails. The differentiator is something about how DTensor's
internal tensor allocation interacts with Monarch's process tree —
likely each `_make_replicate_tensor` allocation goes through a path
that doesn't produce SYCL-USM-device memory.

**Fixing this would require either**:
- An upstream torch.xpu / oneCCL change so allocator paths reliably
  produce USM-classified tensors regardless of how they're constructed.
- Reimplementing `Replicate._make_replicate_tensor` (and the other
  DTensor collective wrappers) to allocate via `torch.empty(device=...)`
  + copy_ rather than `.to(device)`. Too intrusive to do as a shim.

Track C remains the working production path. The `xpu_overrides.py`
patches added during this investigation are independently useful (the
RNG-state USM allocator fix, the LOCAL_RANK hook, the set_device pin)
and are kept in place for future Monarch attempts.

## Update 2026-06-13 late eve: TCP-KVS XCCL fix tried but Monarch path STILL blocked

After the TCP-KVS XCCL discovery unblocked Track C (TRL+vllm-serve,
see `grpo-on-xpu-status.md`), I applied the same fix to the Monarch
path (jobs 12468781, 12468782). Same env vars set, same
`apply_all_xpu_patches()` running in each actor's `_bootstrap`:

```
[xpu_patch] override PALS_LOCAL_RANKID=1 PALS_RANKID=1
   CCL_PROCESS_LAUNCHER=none CCL_ATL_TRANSPORT=ofi FI_PROVIDER=tcp
   CCL_KVS_IP_PORT=127.0.0.1_29515 PALS_LOCAL_SIZE=2
```

Diagnostic confirms every env var oneCCL needs is present at
`init_distributed` time. Yet the same `mesh_broadcast` →
`ccl_check_usm_pointers: invalid usm pointer type` error fires.

So the TCP-KVS env knobs are **necessary but still not sufficient**
for Monarch's particular spawn pattern. The remaining hypotheses:

1. **Process-tree relationship**: TRL's working case has each rank
   independently launched from bash (mpiexec child or background
   subshell). Monarch's actors all descend from a single Python
   parent via `BootstrapCommand(program=...)`. Even though execve
   gives them fresh interpreters, something about that shared
   ancestry might leak into oneCCL/SYCL state.
2. **TCPStore vs env://**: TRL's `init_communicator` constructs an
   explicit `torch.distributed.TCPStore` and passes it to
   `ProcessGroupXCCL`. The torchtitan trainer calls
   `init_process_group(backend="xccl")` without a store, falling
   back to env-based rendezvous. The env-based path may invoke
   different oneCCL init code that still trips PMI.
3. **Master-bind race**: TCPStore needs one rank to be `is_master=True`
   and bind the port. With Monarch spawning 6 actors near-simultaneously,
   the race to bind could mess up rendezvous.

Track C unblocks production GRPO on XPU, so the urgency to fix the
Monarch path is lower. The XPU porting shim (`xpu_overrides.py`)
remains the right place to land any further fixes when revisited.

## Update 2026-06-13 PM: Track C confirmed working (job 12468772)

`trl_vllm_serve_smoke.sh` runs `trl vllm-serve` from `venvs/rl-vllm/`
against Qwen3-0.6B and works end-to-end:

- Phase 1: stack imports clean (torch 2.12+xpu / vllm 0.22.1 /
  trl 1.6.0 / transformers 5.11.0 / xpu_count=12)
- Phase 2: `trl vllm-serve` launches, EngineCore loads the
  checkpoint, uvicorn comes up on :8765
- Phase 3: POST `/generate/` returns HTTP 200 in 2.3s with 32
  completion tokens + per-token logprobs

Two non-obvious issues caught along the way (now baked into the
smoke script):
- TRL 1.6's FastAPI endpoints all have trailing slashes (`/health/`,
  `/generate/`, `/chat/`). `/health` without the slash 404s.
- The ALCF `http_proxy` intercepts loopback. Need
  `no_proxy=127.0.0.1,localhost` for our own server polls.

This unblocks Track C entirely. The TRL `vllm_mode="server"` GRPO
loop can use this server pattern with our existing ezpz trainer
(mpiexec-launched, XCCL-working) on one set of tiles and the vLLM
server on another. No Monarch in the picture means no PMIx mismatch.

## Files added during this investigation

- `torchtitan/experiments/ezpz/rl/xpu_overrides.py`
- `torchtitan/experiments/ezpz/rl/train_upstream.py`
- `torchtitan/experiments/ezpz/rl/scripts/grpo_qwen3_smoke.sh`
- `torchtitan/experiments/ezpz/rl/scripts/build_rl_vllm_venv.sh` (updated)
- `torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B/` (HF download)

## Job log

| Job | Stage reached | Blocker |
|---|---|---|
| 12468755 | trainer ctor | wandb phantom install (fixed) |
| 12468756 | set_determinism | seed broadcast USM (--debug.seed 42) |
| 12468757 | parallelize_fn TP=2 | impi-rt poisoning libccl (uninstalled) |
| 12468758-9 | init_weights TP=1 | RNG broadcast USM |
| 12468760 | init_weights TP=2 | DTensor mesh_broadcast USM |
| 12468761 | init_weights | PALS env injection (insufficient) |
| 12468762 | torchstore init | circular import (fixed) |
| 12468763 | init_weights | regression (import order) |
| 12468764 | init_weights | reorder didn't help |
| 12468765 | init_weights | DIAGNOSTIC: env injection confirmed working but USM still fails |
| 12468766 | init_weights | Gloo override: "No backend type for xpu" |
| 12468767 | init_weights | permanent Gloo map: same error |
| 12468768 | init_weights | "xpu:gloo,cpu:gloo": Gloo can't broadcast xpu tensors |
| 12468769 | init_weights | mpiexec --np 1 wrapper: PMIx doesn't fork-inherit |
