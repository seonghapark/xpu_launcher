# vLLM-XPU on torch 2.13 — investigation findings

**Date:** 2026-06-10
**Goal:** Determine whether we can replace the current all-ranks-generate
HF `.generate()` path in `ezpz/rl` with a vLLM-backed
generator/trainer split, **without** breaking our torch 2.13 XPU
build in `.venv/`.

## TL;DR

**Working end-to-end via Option B.** Pin a sibling venv
(`venvs/vllm-test/`) to `torch==2.12.0+xpu` for vLLM-XPU; keep
`.venv/` on torch 2.13 for the trainer. The generator and trainer
already run as separate processes, so the version skew is
operationally fine. Verified on Sunspot 2026-06-10 — vLLM loaded
the SFT'd `checkpoint-729-hf` in 5.6 s, allocated 48 GiB of KV
cache (2.1M tokens, 1033× concurrency at 2048 ctx), and generated
correct answers to both `"What is 3 + 7 + 2?"` (→ "11") and an
open-ended haiku prompt.

The original blocker was a libtorch C++ ABI mismatch:
`vllm-xpu-kernels` pre-built wheels link against
`torch==2.12.0+xpu` and reference symbols that don't exist in
torch 2.13's `libtorch_cpu.so`. Going with the pinned torch 2.12
sidesteps the build-from-source path entirely.

Until ezpz/rl is refactored to use vLLM as a generator, the
all-ranks-`.generate()` design stays. It's correct, just slow.

## What I tried

### 1. Install vLLM into a sibling venv (`venvs/vllm-test/`)

Used `uv venv` + `uv pip install vllm --no-deps` so vLLM couldn't
pull a different (CUDA-only) torch over our `.venv/`'s torch 2.13.

Then walked the missing-pure-python-dep tree via an auto-loop until
`from vllm import LLM, SamplingParams` succeeded:

```
distro, cloudpickle, msgspec, pyzmq, fastapi, uvicorn, openai,
pydantic, prometheus-client, tiktoken, sentencepiece, protobuf,
psutil, py-cpuinfo, cbor2, gguf, einops, blake3, partial-json-parser,
outlines, outlines-core, lm-format-enforcer, diskcache,
mistral-common, ray, starlette, anyio, sniffio, h11, httptools,
websockets, watchfiles, python-multipart, openai-harmony,
pybase64, cachetools, uvloop
```

Total ~36 pure-Python pip packages, ~30s install in aggregate. None
of them pull torch as a hard dep (we kept `--no-deps` throughout).

**Result:** vLLM imports work end-to-end on our torch 2.13 from a
sibling venv with `PYTHONPATH=.venv/lib/python3.14/site-packages`.

### 2. Platform detection on a compute node

ssh'd into one of my eval-allocation nodes (with XPU devices
visible) and ran vLLM's platform-detection probe:

```python
from vllm.platforms import current_platform
print(type(current_platform).__name__)   # → UnspecifiedPlatform
print(current_platform.is_xpu())          # → False
```

`torch.xpu.is_available()` returns `True` and `torch.xpu.device_count()`
returns 1, but vLLM still falls through to `UnspecifiedPlatform`.

With `VLLM_LOGGING_LEVEL=DEBUG`:

```
Checking if XPU platform is available.
XPU platform is not available because: No module named 'vllm_xpu_kernels'
```

### 3. Install `vllm-xpu-kernels`

Active upstream project at
[vllm-project/vllm-xpu-kernels](https://github.com/vllm-project/vllm-xpu-kernels)
(last update **today, 2026-06-10**). Releases publish a single
`vllm_xpu_kernels-<ver>-cp38-abi3-manylinux_2_28_x86_64.whl`
(stable Python ABI, ~350 MB).

Downloaded `v0.1.9.1` (2026-06-01), installed with `--no-deps`. The
`cp38-abi3` Python ABI is broad (works on cpython 3.8 through 3.14)
— but Python ABI compatibility does **not** imply libtorch ABI
compatibility.

### 4. Retry platform detection — symbol-level failure

```
XPU platform is not available because:
.../vllm_xpu_kernels/_C.abi3.so:
undefined symbol: _ZN3c104impl3cow23materialize_cow_storageERNS_11StorageImplE
```

Demangled: `c10::impl::cow::materialize_cow_storage(c10::StorageImpl&)`.

Verified the symbol does not exist in our
`.venv/lib/python3.14/site-packages/torch/lib/libtorch_cpu.so`:

```bash
$ nm -D .venv/.../libtorch_cpu.so | grep materialize_cow_storage
(no output)
```

The pre-built wheel was linked against a torch version that exposes
this symbol — most likely the upstream-pinned `torch==2.12.0+xpu`
that `vllm-xpu-kernels`'s README requires:

> **PyTorch**: 2.12.0+xpu

`cp38-abi3` is the **Python** stable ABI tag. It's about the
Python/C API surface, not about libtorch (`libtorch_cpu.so`,
`libc10.so`). The kernels wheel links against libtorch at build
time, so it's effectively pinned to whatever torch version the
maintainers used for the build.

## Why this matters

Without the kernels, vLLM falls back to a CPU-only path on this
host. Useful for sanity checks but not for the 5–10× generator
speedup we want vs the current HF `.generate()` design.

## Paths forward

**Option A — Build `vllm-xpu-kernels` from source against our torch 2.13.**
Requires the oneAPI 2025.3 + CMake ≥ 3.26 + Ninja toolchain
(already on the system) plus SYCL/DPC++ knowledge. The repo's
build instructions are minimal but appear to be standard
PyTorch-style C++ extension builds. Estimated 1–2 days of trial,
mostly debugging template instantiations against newer torch
headers.

**Option B — Pin a sibling venv to vLLM's required `torch==2.12.0+xpu`.**
Single-line `uv pip install torch==2.12.0+xpu --extra-index-url
https://download.pytorch.org/whl/xpu` into `venvs/vllm-test/`.
Doesn't touch `.venv/`. The generator process would run on torch
2.12 while the trainer runs on torch 2.13. Acceptable because the
generator only does forward passes / weight loads — no autograd
state shared across processes — but does mean two different torch
builds on the same machine, with the version skew risks that
implies.

**Option C — Defer until vLLM ships a torch 2.13 wheel.**
`vllm-xpu-kernels` is actively maintained (13 releases since
2026-01-29, latest 2026-06-01). They'll bump to torch 2.13 once
that's stable upstream. Could be weeks, could be months.

**Recommendation: Option B.** Lowest engineering cost (~10 min),
clean separation between trainer and generator processes, no
need to wait for upstream. The cost is maintaining two torch
versions in two venvs — annoying but tractable.

## Option B — working install path (2026-06-10)

```bash
# 1. sibling venv (no touch to .venv/)
uv venv venvs/vllm-test --python 3.14

# 2. torch 2.12+xpu from pytorch.org (pulls oneAPI deps; matches what
#    vllm-xpu-kernels was built against)
uv pip install --python venvs/vllm-test/bin/python3 \
  --extra-index-url https://download.pytorch.org/whl/xpu \
  torch==2.12.0+xpu

# 3. vllm itself, --no-deps so it can't pull a different torch
uv pip install --python venvs/vllm-test/bin/python3 --no-deps vllm

# 4. vllm-xpu-kernels (the SYCL/DPC++ XPU kernel pack; ABI-matches
#    torch 2.12 not 2.13)
gh release download v0.1.9.1 --repo vllm-project/vllm-xpu-kernels \
  --pattern 'vllm_xpu_kernels-*.whl' -D /tmp/
uv pip install --python venvs/vllm-test/bin/python3 --no-deps \
  /tmp/vllm_xpu_kernels-*.whl

# 5. vllm's pure-python runtime deps (this is the long tail; install
#    --no-deps each so torch is never replaced). Discovery loop:
#    keep python -c 'from vllm import LLM' and install whatever
#    ModuleNotFoundError surfaces, with PIL→pillow, attr→attrs
#    overrides.
```

Full dep set ends up being (~50 packages, all pure-python or
pre-built non-torch-touching wheels):

```
numpy pydantic-core==2.46.4 typing-inspection annotated-types
transformers huggingface-hub safetensors tokenizers>=0.22.0,<=0.23.0
multidict attrs yarl propcache aiohappyeyeballs aiosignal
frozenlist pillow annotated-doc
cloudpickle msgspec pyzmq fastapi uvicorn aiohttp openai
pydantic prometheus-client tiktoken sentencepiece protobuf
psutil py-cpuinfo cbor2 gguf einops blake3 partial-json-parser
outlines outlines-core lm-format-enforcer diskcache mistral-common
ray starlette anyio sniffio h11 httptools websockets watchfiles
python-multipart importlib-metadata zipp depyf astor filelock
packaging requests typing-extensions pyyaml regex tqdm jinja2
markupsafe charset-normalizer urllib3 idna certifi httpx httpcore
distro pybase64 cachetools uvloop openai-harmony
```

## End-to-end verification (compute node, x1921c7s3b0n0, XPU tile 0)

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="outputs/sft/aurora2b-sophiag-tulu-mix-32n-gbs6144/checkpoint-729-hf",
    dtype="bfloat16",
    enforce_eager=True,
    gpu_memory_utilization=0.85,
    max_model_len=2048,
)
sp = SamplingParams(temperature=0.7, max_tokens=32, top_p=0.95)
outputs = llm.generate([
    "What is 3 + 7 + 2?",
    "Write a haiku about Aurora supercomputer.",
], sp)
```

Logs (abridged):

```
INFO weight_utils.py:879  Prefetching checkpoint files finished in 4.86s
INFO gpu_model_runner.py:5132 Model loading took 3.74 GiB / 5.61s
INFO interface.py:496      Setting kv cache block size to 64 for FLASH_ATTN
INFO gpu_worker.py:466     Available KV cache memory: 48.43 GiB
INFO kv_cache_utils.py:1733 GPU KV cache size: 2,115,904 tokens
INFO kv_cache_utils.py:1734 Maximum concurrency for 2,048 tokens: 1033.16x
INFO kernel.py:270         IrOpPriorityConfig(rms_norm=['xpu_kernels','native'],
                                              fused_add_rms_norm=['xpu_kernels','native'])
```

Real outputs:

```
PROMPT: What is 3 + 7 + 2?
GEN:    3 + 7 + 2 = 11. So, the answer is 11. #### 11
PROMPT: Write a haiku about Aurora supercomputer.
GEN:    • Aurora supercomputer as seen from Earth.
        • Aurora supercomputer is a cloud-based supercomputer,
        • that is capable of ...
```

First inference burns ~5.8 s on JIT warmup (Triton kernels
compiling for the new shape — visible in the
`Triton kernel JIT compilation during inference` warning). After
warmup, output throughput is ~11 tok/s with `enforce_eager=True`,
which forces no torch.compile and no XPU graphs. With
`VLLM_XPU_ENABLE_XPU_GRAPH=1` (off by default) the
graph-captured throughput should be 3–5× higher.

## What's still left to do

The `venvs/vllm-test/` venv is now usable for inference. To
actually replace the all-ranks-`.generate()` path in `ezpz/rl`:

1. Refactor `train_grpo.py` / `train_sft.py` to support an
   external generator. TRL has built-in `vllm_mode="server"` and
   `vllm_mode="colocate"` paths — we'd use the **server** mode
   (separate vllm OpenAI-compatible API process, trainer hits it
   via HTTP).
2. Stand up a per-allocation vLLM server (probably 1 tile per
   server, N servers for N concurrent generator slots) and a
   trainer-side `OpenAIServingChat` client config in `train_grpo.py`.
3. Wire weight sync between trainer and generator. TRL has
   `vllm_client.update_named_param(...)` for in-process syncing;
   server mode uses an HTTP endpoint.

Estimated 2–3 days end-to-end. Worth doing once a single GRPO
trajectory consumes more than a few hours of wall time —
currently SFT is the dominant cost, but that's about to flip.

## In the meantime

The all-ranks-generate design in `ezpz/rl` stays. It's correct,
just slow. For the first production GRPO runs at 8N×1000-step
scale, the cost is fine; we burn extra compute but the
experiments still finish in walltime.
