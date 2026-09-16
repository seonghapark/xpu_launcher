# Monarch + torch 2.13 deep-dive (2026-06-14)

End-of-session writeup. Burned ~16 PBS jobs (12468799 → 12468815)
adding layer after layer of XPU compatibility patches. Got past
every torchtitan + DCP + XCCL issue we hit. Final blocker is a
vLLM-XPU `profile_run` oneDNN allocator failure that needs Intel
involvement — out of scope for ezpz workarounds.

## TL;DR

Built a complete torch-2.13 + py3.13 + monarch + vllm-xpu venv
(`venvs/rl-monarch-torch213/`, includes custom-built
`vllm-xpu-kernels` against torch 2.13). Pushed the upstream
`torchtitan.experiments.rl.train` Monarch + GRPO pipeline through
~10 progressively-deeper failure modes, all fixed in
`xpu_overrides.py`. Final crash is `RuntimeError: could not create
a memory` from oneDNN during vLLM's `profile_run` → first decoder
layer `F.linear`. Same crash with custom kernels disabled
(`VLLM_DISABLED_KERNELS=xpu_kernels`), so it's torch's own oneDNN
matmul that's blowing up.

## Patches that worked (now in `xpu_overrides.py`)

In order of when they fired:

1. **`EzpzPerHostProvisioner.make_bootstrap_command_for_gpu_ids`** —
   uses Monarch's `bootstrap_command=` (vs `bootstrap=`) to inject
   `ZE_AFFINITY_MASK`, `LOCAL_RANK`, `RANK`, `WORLD_SIZE`,
   `PALS_*_RANKID` into the actor's env BEFORE execve. The previous
   `bootstrap=callable` runs AFTER monarch's `bootstrap_main.py`
   already did `import torch`, so torch.xpu's primary SYCL context
   was built before the mask narrowed.

2. **`patch_dtensor_make_replicate_for_xpu` (skip broadcast variant)** —
   `Replicate._make_replicate_tensor` calls `mesh_broadcast(t, mesh)`
   on whatever tensor it's given. Buffers reach this path during
   `init_states` → `distribute_tensor(precomputed_xpu_buf, mesh,
   [Replicate()])`. On torch 2.13 + Monarch + XPU, the broadcast
   fails `ccl_check_usm_pointers` even on fresh `torch.empty(device=
   xpu)` allocations. WORKAROUND: skip the broadcast entirely and
   return the local tensor — every rank already constructed an
   identical buffer via the same deterministic init code, so the
   broadcast is a no-op anyway.

3. **`patch_init_distributed_for_xpu` (force `enable_cpu_backend=True`)** —
   `torchtitan.distributed.utils.init_distributed` builds the default
   PG with backend `xccl` only. DCP's planner uses
   `dist.all_gather_object` for the central plan, which routes object
   pickles through device tensors → oneCCL's USM check fails. With
   `enable_cpu_backend=True`, the backend becomes `xpu:xccl,cpu:gloo`
   and `all_gather_object` uses gloo on CPU.

4. **`patch_vllm_xpu_skip_oneccl_warmup`** —
   `vllm/v1/worker/xpu_worker.py:103` does
   `torch.distributed.all_reduce(torch.zeros(1).xpu())` after
   `init_worker_distributed_environment` "for overall oneccl warm
   up". On Monarch actors this allreduce trips the USM check and
   kills the engine. Patch suppresses it (only the specific warmup
   site — real user allreduces still go through).

5. **`patch_vllm_xpu_attention_backend`** — vLLM-XPU's platform
   selector at `vllm/platforms/xpu.py:89` raises `ValueError: Invalid
   attention backend for xpu` when it sees `AttentionBackendEnum.
   CUSTOM`. `rl/actors/generator.py` configures vLLM with CUSTOM
   when the model spec uses `VarlenAttention.Config`. Patch returns
   CUSTOM's registered path instead.

6. **`patch_vllm_xpu_no_alias_current_stream`** —
   `vllm/v1/worker/xpu_model_runner.py:_torch_cuda_wrapper` does
   `torch.cuda.current_stream = torch.xpu.current_stream`. Dynamo's
   `(cuda, xpu, accelerator).current_stream` handler-table build then
   trips `AssertionError: Handler already registered` because the
   same function appears twice. Patch keeps every other alias
   (`Stream`, `default_stream`, `stream`, `mem_get_info`, `Event`,
   `set_stream`, graph stuff) but skips `current_stream`.

## What still blocks

vLLM's `EngineCore` runs `profile_run → _dummy_run(max_num_tokens=
2048, skip_attn=True, is_profile=True)`, which calls into the model:

```
torchtitan/models/qwen3/model.py:60 in forward
torchtitan/models/common/attention.py:709 self.qkv_linear(x)
torchtitan/models/common/attention.py:599 self.wq(x), self.wk(x), self.wv(x)
torch/nn/modules/linear.py:134 F.linear(input, self.weight, self.bias)
RuntimeError: could not create a memory
```

Tried (no change):
- `--generator.gpu-memory-limit 0.5` (down from 0.9)
- `--generator.sampling.max-tokens 256` (down from 700)
- `--generator.cudagraph.no-enable` (skip CUDA graph capture)
- `--compile.no-enable` (skip torch.compile)
- `--batcher.batch.seq-len 512` (down from 2048)
- Narrow `ZE_AFFINITY_MASK` per actor (single tile visible)
- `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` (drop opencl backend)
- `VLLM_DISABLED_KERNELS=xpu_kernels` (fall back to torch oneDNN)

Model load succeeds (`Model loading took 1.22 GiB and 1.789503
seconds`), so 46 GiB tile budget is intact — this is NOT actual
memory exhaustion. It's oneDNN's `dnnl::memory` constructor failing,
likely due to a SYCL context mismatch between torch.xpu's allocator
and oneDNN's primitive scratchpad.

Suspect causes (not confirmed):
- Our custom-built `vllm-xpu-kernels` against torch 2.13 silently
  introduces an allocator-pool conflict. Disabling its custom op
  impls via `VLLM_DISABLED_KERNELS=xpu_kernels` didn't help though.
- `torch.xpu`'s primary SYCL context is constructed differently
  under Monarch's execve'd actors vs mpiexec'd processes. mpiexec
  + PMIx gives oneCCL/oneDNN a coordinated device handle table;
  Monarch doesn't.
- oneDNN's scratchpad allocator queries `sycl::get_pointer_type`
  with a different context than torch's allocator returns from. Same
  root cause as the oneCCL USM check we worked around — but for
  oneDNN there's no "skip the check" fallback.

## Job log (this session)

| Job | Patch under test | Outcome |
|---|---|---|
| 12468799 | `_make_replicate_tensor` rebox via `empty_like().copy_()` | broadcast still fails USM check |
| 12468800 | Broad ZE_AFFINITY_MASK + LOCAL_RANK pick tile | same USM failure |
| 12468801 | + diagnostic `CCL_LOG_LEVEL=info` + tensor ptr logging | found: stream context = correct device, but `get_pointer_type` returns unknown |
| 12468802 | `bootstrap_command=` env overlay (pre-execve ZE_AFFINITY_MASK) | same USM failure |
| 12468803 | **SKIP broadcast in `_make_replicate_tensor`** | **passed init_weights**, hit DCP allgather USM crash |
| 12468804 | + force `enable_cpu_backend=True` (gloo for objects) | **passed DCP HF load**, hit vLLM TP=4 allreduce USM crash |
| 12468805 | + TP=1 everywhere (sidestep intra-mesh XCCL) | hit vLLM `xpu_worker.py:103` warmup allreduce |
| 12468806 | + skip vllm-xpu oneCCL warmup allreduce | hit `Invalid attention backend for xpu` (CUSTOM) |
| 12468807 | + `VLLM_ATTENTION_BACKEND=FLASH_ATTN` | same — env not enough, need platform selector patch |
| 12468808 | + `XPUPlatform.get_attn_backend_cls` accepts CUSTOM | hit dynamo `Handler already registered for current_stream` |
| 12468809 | + patched `_torch_cuda_wrapper` to NOT alias `current_stream` (or `Stream`) | hit `torch.cuda.Stream requires CUDA` (overcorrected) |
| 12468810 | + narrower patch — only skip `current_stream` alias | hit `could not create a memory` |
| 12468811 | + `gpu_memory_limit=0.4` + `max_tokens=256` | same OOM |
| 12468812 | + `cudagraph.no-enable` + `compile.no-enable` + `seq_len=512` | same OOM |
| 12468813 | + narrow ZE_AFFINITY_MASK per actor | same OOM |
| 12468814 | + `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` (drop opencl) | same OOM |
| 12468815 | + `VLLM_DISABLED_KERNELS=xpu_kernels` | same OOM (not the custom kernels) |

## Next steps (when resumed)

Don't keep flailing at the oneDNN allocator in actor processes. The
honest path forward is one of:

1. **Get an Intel torch.xpu engineer to look at oneDNN's `dnnl::
   memory` failure under Monarch's execve'd actors.** The error
   `could not create a memory` is bare and not actionable without
   oneDNN debug output (`DNNL_VERBOSE=2` or similar).

2. **Try this same stack under mpiexec instead of Monarch.** If the
   same code path inside an mpiexec-launched single process
   completes profile_run successfully, that's strong evidence the
   bug is Monarch-spawn-specific. The TRL+vllm-serve case
   (`venvs/rl-vllm/` + torch 2.12) already proved mpiexec + XCCL
   works.

3. **Try torch 2.12 instead of 2.13.** We rebuilt vllm-xpu-kernels
   against 2.13 because of the missing `materialize_cow_storage`
   symbol on the prebuilt 0.1.9.1 wheel. Maybe try torch 2.12 +
   the prebuilt vllm-xpu-kernels — known-good combo on Sunspot,
   though will lose torch 2.13's DTensor USM fixes (which we ended
   up skipping the broadcast for anyway).

4. **Skip vLLM entirely for the generator side.** Replace
   `VLLMGenerator` with a torch-only "naive generator" that does
   greedy / sampled decode with `model.forward` calls. Loses
   throughput but bypasses every vLLM-XPU init step.

## Files changed

- `torchtitan/experiments/ezpz/rl/xpu_overrides.py` — added
  `make_bootstrap_command_for_gpu_ids` to the provisioner, added
  6 new monkey-patch functions, registered them all in
  `apply_all_xpu_patches`.
- `torchtitan/experiments/ezpz/rl/train_upstream.py` — wires
  `bootstrap_command=` in the single-host spawn branch.
- `torchtitan/experiments/ezpz/rl/scripts/grpo_monarch_torch213_smoke.sh` —
  new PBS smoke script, encodes the final working env + CLI overrides.
