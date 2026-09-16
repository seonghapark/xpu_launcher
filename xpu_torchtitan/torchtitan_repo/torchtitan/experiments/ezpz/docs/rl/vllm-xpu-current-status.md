# vLLM-XPU + Monarch RL infra status (as of 2026-06-13 PM)

## Summary

| Component | Status | Notes |
|---|---|---|
| `venvs/rl-actors/` venv build | ✅ done | py3.13 + torch 2.12+xpu + vllm 0.22 + monarch + torchstore + TRL 1.6, all imports clean |
| Monarch actors on XPU | ✅ working | Job `12468739`: 2-rank `spawn_procs` confirmed, both ranks see `xpu_count=12` from inside actor |
| TorchStore on XPU (Gloo transport) | ✅ importable | Full round-trip not yet smoked; transport class loads cleanly |
| **vLLM-XPU bare engine init (single-tile)** | ✅ **WORKING** | Verified interactive 2026-06-13 PM on x1921c3s0b0n0: KV cache 48.43 GiB, max concurrency 1033x — matches 2026-06-10 baseline. Fix: scrub CCL_*/FI_* env vars set by `ezpz_setup_env` before invoking vLLM. |
| TRL `vllm_mode="server"` | 🟡 needs re-test | Should work now that the underlying engine init is fixed. Worth re-running the failing 12468737 smoke with the env-scrub fix in place. |
| ezpz `EzpzVLLMGenerator` skeleton | scaffolded | `rl/actors/ezpz_generator.py` documents the 5 upstream override points; ready to wire end-to-end now. |

## TL;DR

The bug was self-inflicted. `ezpz_setup_env` exports oneCCL/libfabric
env vars (`CCL_PROCESS_LAUNCHER=pmix`, `FI_PROVIDER=cxi,tcp;ofi_rxm`,
several `FI_CXI_*`) that are correct for ezpz/mpiexec training but
fatal for vLLM's standalone EngineCore subprocess. The `cxi`
Slingshot provider needs a NIC handle that only mpiexec-bootstrapped
processes have; without it `fi_getinfo` returns 0 providers and
ATL init fails. Stripping these env vars lets oneCCL fall back to
working defaults.

**Fix** (in `vllm_xpu_bare_smoke.sh`):
```bash
unset CCL_OP_SYNC CCL_PROCESS_LAUNCHER CCL_ATL_TRANSPORT CCL_OFI_PROVIDER
unset FI_PROVIDER FI_LOG_LEVEL FI_LOG_PROV FI_LOG_LOCATION
unset FI_CXI_DEFAULT_CQ_SIZE FI_CXI_DEFAULT_TX_SIZE FI_CXI_OFLOW_BUF_COUNT
unset FI_CXI_OFLOW_BUF_SIZE FI_CXI_RDZV_EAGER_SIZE FI_CXI_RDZV_THRESHOLD
unset FI_CXI_REQ_BUF_MAX_CACHED FI_CXI_REQ_BUF_MIN_POSTED FI_CXI_REQ_BUF_SIZE
unset FI_CXI_RX_MATCH_MODE FI_MR_CACHE_MAX_COUNT FI_MR_CACHE_MAX_SIZE
```

And invoke vLLM via plain `python` (no `ezpz launch` / `mpiexec`
wrapper — TP=1 doesn't need it, and the wrapper actively re-pollutes
the env).

## What works

The **Monarch framework itself** is fully functional on Sunspot XPU.
Job `12468739` ran a 2-actor smoke (`this_host().spawn_procs({"gpus": 2})`)
that:
1. Echoed messages between controller and two spawned actors.
2. Each actor independently called `torch.xpu.device_count()` and
   reported 12 — confirming the actor processes correctly inherit
   XPU visibility.
3. Imported `TorchStore.LocalRankStrategy(default_transport_type=TransportType.Gloo)`
   for the non-CUDA path.

This unblocks the question "can the upstream actor pattern work on
XPU at all?" — yes.

## Root cause (resolved 2026-06-13 PM)

The bug was **self-inflicted by our smoke scripts**, not a stack
regression. `ezpz_setup_env` exports a set of oneCCL/libfabric env
vars that are correct for ezpz mpiexec-launched training but fatal
for vLLM's standalone EngineCore subprocess:

| Env var | Set by | Why it kills vLLM |
|---|---|---|
| `CCL_PROCESS_LAUNCHER=pmix` | ezpz scripts | makes oneCCL look for PMIx; vLLM's subprocess has none |
| `CCL_ATL_TRANSPORT=mpi` | ezpz scripts | routes through MPI bootstrap; `MPIDI_GPU_init_mpl_global` segfaults at world_size=1 without mpiexec |
| `FI_PROVIDER=cxi,tcp;ofi_rxm` | `ezpz_setup_env` | `cxi` requires Slingshot NIC handle from mpiexec; without it `fi_getinfo` returns 0 providers → `atl_ofi init_transport` fails with "can't find suitable provider" |
| `FI_CXI_*` settings | `ezpz_setup_env` | tune the cxi provider that we can't even open |

Verified the diagnosis on x1921c3s0b0n0 (interactive, no `ezpz launch`):

```
2026:06:13-20:51:02 |CCL_INFO| libfabric version: 2.2.0-impi_2021.17.2
2026:06:13-20:51:02 |CCL_ERROR| atl_ofi_helper.cpp:1118
   atl_ofi_get_prov_list: fi_getinfo error: ret -61, providers 0
2026:06:13-20:51:02 |CCL_ERROR| atl_ofi_helper.cpp:1158
   can't create providers for name <default>
```

(`ret -61` = `ENODATA`, "no info available" — i.e. CXI provider
didn't open because PMIx wasn't bootstrapped.)

After `unset CCL_* FI_*` of every contaminating var, the same recipe
loaded the SFT'd 2B checkpoint, allocated 48.43 GiB KV cache (max
concurrency 1033x — bit-for-bit match with the 2026-06-10 baseline),
and ran `llm.generate()` to completion.

**Why it worked on 2026-06-10**: the 2026-06-10 verification was an
interactive shell from inside an eval allocation that had NOT sourced
`ezpz_setup_env` — the operator was poking around vLLM imports, not
running a training launcher. None of the contaminating env vars were
set. Today's smoke scripts all source `ezpz_setup_env` for its
nodefile/PBS bookkeeping, which inadvertently sets all the CCL/FI
overrides too.

## The fix

In `vllm_xpu_bare_smoke.sh` (and any other vLLM smoke):

```bash
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job

# After ezpz_setup_job (keep the PBS / nodefile bookkeeping), strip
# the oneCCL/libfabric env vars it/ezpz_setup_env exported. vLLM's
# EngineCore subprocess has no MPI bootstrap and needs the
# oneCCL defaults.
unset CCL_OP_SYNC CCL_PROCESS_LAUNCHER CCL_ATL_TRANSPORT CCL_OFI_PROVIDER
unset FI_PROVIDER FI_LOG_LEVEL FI_LOG_PROV FI_LOG_LOCATION
unset FI_CXI_DEFAULT_CQ_SIZE FI_CXI_DEFAULT_TX_SIZE FI_CXI_OFLOW_BUF_COUNT
unset FI_CXI_OFLOW_BUF_SIZE FI_CXI_RDZV_EAGER_SIZE FI_CXI_RDZV_THRESHOLD
unset FI_CXI_REQ_BUF_MAX_CACHED FI_CXI_REQ_BUF_MIN_POSTED FI_CXI_REQ_BUF_SIZE
unset FI_CXI_RX_MATCH_MODE FI_MR_CACHE_MAX_COUNT FI_MR_CACHE_MAX_SIZE

# Invoke vLLM via plain python — no `ezpz launch` / `mpiexec` wrapper.
# TP=1 doesn't need it, and the wrapper re-sets the env vars.
venvs/rl-actors/bin/python rl/scripts/vllm_xpu_bare_smoke.py
```

## Debug chain so far (jobs 12468737..12468749)

| Job | Symptom | Fix attempted | Result |
|---|---|---|---|
| 12468737 | TRL `vllm_serve`: `current_platform.device_type` empty in worker | n/a — diagnostic | revealed TRL 1.5.1 + vllm 0.22 ABI mismatch |
| 12468740 | bare vllm: `ModuleNotFoundError: xgrammar` | `uv pip install xgrammar` | next error |
| 12468741 | repeated old error (xgrammar install hadn't propagated) | retry | next error |
| 12468742 | `FileNotFoundError: '<stdin>'` from heredoc | move to real `.py` file | next error |
| 12468743 | `oneCCL atl_ofi`: `libpsm2.so.2` / `libucp.so.0` not found | `export CCL_ATL_TRANSPORT=mpi` | switched to MPI failure |
| 12468744 | `MPIR_pmi_init(197): PMIX_Init returned -25` (no MPI bootstrap) | launch via `ezpz launch --np 1` | next error |
| 12468745 | `ezpz: command not found` (path issue) | use absolute `.venv/bin/ezpz` | next error |
| 12468746 | `ZMQError: ipc path > 107 chars` | `TMPDIR=/tmp/vllm-$USER` (short path) | got past ZMQ, hit MPI segfault |
| 12468747 | `MPIDI_GPU_init_mpl_global` segfault in MPI bootstrap | n/a — root cause | failed |
| 12468748 | same segfault, even with monkey-patched `XPUPlatform.dist_backend="gloo"` | torch's XCCL fires anyway since tensors are on XPU | failed |
| 12468749 | `ezpz launch --np 2 -ppn 2` to test single-rank hypothesis | **same segfault** at both ranks — kills the "np=1 is the bug" theory | failed |
| 12468750 | Replay 2026-06-10 recipe verbatim from `venvs/vllm-test/` (py3.14) | **same failure mode** — EngineCore subprocess dies silently after `CCL_WARN| value of CCL_OP_SYNC changed to be 1` + `CCL_PROCESS_LAUNCHER changed to be pmix`. Kills "rl-actors venv (py3.13 ABI) is the bug" theory. | failed |
| 12468751 | Drop all `CCL_*` env overrides — let oneCCL pick defaults (which on Sunspot is OFI transport) | **different failure**: `RuntimeError: oneCCL: atl_ofi_comm.cpp:232 init_transport: EXCEPTION: failed to initialize ATL`. Both transports broken: MPI silently segfaults, OFI explicitly fails to init. | failed |

## Implications for downstream wiring

- **TRL `vllm_mode="server"`**: should work as soon as the server
  launcher applies the same env-scrub. The `vllm_serve_xpu.sh`
  script needs the same `unset CCL_*/FI_*` block.
- **`EzpzVLLMGenerator`**: the actor will host vLLM in its own
  spawned process (via Monarch's `spawn_procs`). Same env-scrub
  applies — the actor's `__init__` should call
  `os.environ.pop(...)` for the contaminating vars before
  constructing the `LLM(...)` object, since by then we're in a
  forked Python process and bash unsets are gone.
- **Multi-rank vLLM (TP > 1)**: untested. If you actually need
  Slingshot inter-tile RDMA for TP=8, you'd want to *keep* the
  Cassini provider and instead launch via `ezpz launch` so PMIx
  bootstraps. That's a separate test we don't need for the first
  GRPO smoke (which is TP=1 anyway).

## Second issue caught by 12468752: triton-xpu cp313 wheels don't exist

Once env-scrub got us past the oneCCL/MPI bootstrap, the rl-actors
(py3.13) venv hit a follow-up failure:

```
ERROR config.py:29 Failed to import Triton kernels.
   Error: cannot import name 'intel' from 'triton._C.libtriton'
...
TypeError: 'function' object is not subscriptable
  at _compute_slot_mapping_kernel[(num_reqs + 1,)](...)
```

Cause: `triton-xpu==3.7.1` (which vllm-xpu-kernels 0.1.9.1 needs)
only ships a `cp314-cp314-manylinux` wheel. The py3.13 fallback
chain picks up:
- vanilla `triton==3.7.0` (cp313 wheel, no Intel symbols)
- `pytorch-triton-xpu==3.5.0` from pytorch.org/whl/xpu (cp313 wheel,
  no `triton.language.target_info` module)

Neither works with vllm-xpu-kernels.

**Initial workaround** (2 venvs): use `venvs/vllm-test/` (py3.14)
for vLLM workers, `venvs/rl-actors/` (py3.13) for Monarch
controllers. Verified PBS-direct on job 12468753 (KV cache 26.04 GiB,
GEN line printed).

**Final fix** (1 venv): `venvs/rl-vllm/` (py3.12) — py3.12 is the
intersection where BOTH `torchmonarch==0.5.0` (cp310-cp313) AND
`triton-xpu==3.7.1` (cp312-cp314) ship native wheels. Single venv
covers actor framework + vLLM worker + trainer. Verified
interactively 2026-06-13 PM on x1921c3s0b0n0:
- Monarch: 2-actor mesh, both ranks see `xpu_count=12`, TorchStore
  Gloo strategy importable.
- bare vLLM: `Available KV cache 26.04 GiB`, `GEN:` line printed.

The earlier "vanilla triton from xgrammar overwrote triton-xpu" issue
is captured explicitly in `rl/scripts/build_rl_vllm_venv.sh` step 6
(uninstall vanilla triton, re-pin triton-xpu).

| Venv | Python | Role |
|---|---|---|
| `venvs/rl-vllm/` | 3.12 | **Unified RL+vLLM venv** (Monarch + vLLM worker + TRL + transformers + accelerate + datasets) |
| `.venv/` | 3.14 | Trainer (torch 2.13.dev, our main stack) |
| ~~`venvs/rl-actors/`~~ | 3.13 | superseded by `rl-vllm/` |
| ~~`venvs/vllm-test/`~~ | 3.14 | superseded by `rl-vllm/` |

`vllm_xpu_bare_smoke.sh` and `monarch_smoke.sh` both invoke
`venvs/rl-vllm/bin/python`. `rl/actors/ezpz_generator.py` docstring
updated to reference the unified venv.

## Files added this session

- `venvs/rl-actors/` — new sibling venv, py3.13 + full RL stack.
- `rl/scripts/monarch_smoke.{py,sh}` — Monarch framework smoke. PASSING.
- `rl/scripts/vllm_xpu_bare_smoke.{py,sh}` — vLLM bare smoke. FAILING
  at MPI bootstrap; see debug chain above.
- `rl/scripts/vllm_serve_xpu.sh`, `vllm_serve_smoke.sh`,
  `grpo/aurora2b_sft_arithmetic_8n_vllm.sh` — server-mode plumbing
  (depends on vllm-xpu working; currently blocked).
- `rl/actors/__init__.py`, `rl/actors/ezpz_generator.py` — skeleton
  for the Monarch+vLLM actor adaptation. Awaits a working
  vllm-xpu base.
- `docs/rl/vllm-xpu-wiring-plan.md` — architecture decision doc.
- `docs/rl/vllm-xpu-current-status.md` — this file.

## Recommended next steps

1. ~~**np=2 hypothesis**~~ — refuted (12468749).
2. ~~**rl-actors venv ABI hypothesis**~~ — refuted (12468750).
3. ~~**No-CCL-overrides hypothesis**~~ — refuted (12468751).
4. ~~**Invocation-context hypothesis**~~ — **confirmed** (interactive
   replay on x1921c3s0b0n0, 2026-06-13 PM). Root cause: env
   contamination by `ezpz_setup_env`. See "The fix" above.
5. **Re-submit `vllm_xpu_bare_smoke.sh`** with env-scrub block —
   verifies the fix lands cleanly in PBS-direct mode (not just
   interactive SSH).
6. **Update `vllm_serve_xpu.sh`** with the same env-scrub and
   re-test TRL `vllm_mode="server"` (which has been blocked on the
   same underlying issue).
7. **Start `EzpzVLLMGenerator` wiring**: actor needs to call
   `os.environ.pop(...)` for the same set of vars in its `__init__`
   before constructing `LLM(...)`. (Bash-level unsets don't carry
   into a Monarch-spawned Python process.)
