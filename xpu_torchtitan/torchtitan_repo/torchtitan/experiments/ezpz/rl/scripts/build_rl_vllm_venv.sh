#!/bin/bash --login
# Build venvs/rl-vllm/ — unified py3.12 venv for Monarch + vLLM + RL infra.
#
# Why py3.12: it's the only Python where ALL the wheels we need ship
# native cp wheels:
#   - torchmonarch 0.5.0  → cp310/cp311/cp312/cp313 (NO cp314)
#   - triton-xpu 3.7.1    → cp312/cp313/cp314      (NO cp311 or older)
# py3.12 is the intersection.
#
# Why a unified venv (instead of rl-actors py3.13 + vllm-test py3.14):
#   - Simpler operational story (one venv, one tarball, one yeet target).
#   - Monarch actor pattern doesn't need to spawn into a different
#     interpreter via BootstrapCommand(program=...) for the worker side.
#   - Eliminates py3.13 vs py3.14 ABI debugging surface entirely.
#
# CRITICAL: per CLAUDE.md golden rule #5, NEVER pip install anything
# that depends on torch without `--no-deps --no-cache --link-mode=copy`.
# Specifically: `xgrammar` declares `triton` (vanilla) as a Linux/x86_64
# dep, which will silently overwrite triton-xpu and break vLLM-XPU. The
# sequence below uninstalls vanilla `triton` after xgrammar pulls it in
# and re-installs triton-xpu.
#
# Verified end-to-end 2026-06-13 PM (x1921c3s0b0n0):
#   - Monarch: 2-actor mesh, both ranks see xpu_count=12, TorchStore
#     transports importable.
#   - bare vLLM: Available KV cache 26.04 GiB, generation runs.
# See docs/rl/vllm-xpu-current-status.md.

set -e

REPO="${REPO:-$(pwd)}"
VENV="${REPO}/venvs/rl-vllm"

echo "=== Step 1: create py3.12 venv ==="
uv venv -p 3.12 "${VENV}" --link-mode=copy

echo ""
echo "=== Step 2: torch 2.12+xpu + triton-xpu 3.7.1 from PyTorch XPU index ==="
# Let this one resolve deps — that's how we get triton-xpu cleanly.
VIRTUAL_ENV="${VENV}" uv pip install --no-cache --link-mode=copy \
    --index-url https://download.pytorch.org/whl/xpu \
    "torch==2.12.0+xpu"

echo ""
echo "=== Step 3: vllm 0.22.1 --no-deps (avoid pulling vanilla triton) ==="
VIRTUAL_ENV="${VENV}" uv pip install --no-cache --no-deps --link-mode=copy \
    "vllm==0.22.1"

echo ""
echo "=== Step 4: vllm-xpu-kernels 0.1.9.1 from GitHub release ==="
VIRTUAL_ENV="${VENV}" uv pip install --no-cache --no-deps --link-mode=copy \
    "vllm-xpu-kernels @ https://github.com/vllm-project/vllm-xpu-kernels/releases/download/v0.1.9.1/vllm_xpu_kernels-0.1.9.1-cp38-abi3-manylinux_2_28_x86_64.whl"

echo ""
echo "=== Step 5: numpy + vllm's pure-Python dep tree ==="
# These are all safe to take their own deps — they don't touch torch/triton.
VIRTUAL_ENV="${VENV}" uv pip install --no-cache --no-deps --link-mode=copy "numpy<2.5"
VIRTUAL_ENV="${VENV}" uv pip install --no-cache --link-mode=copy \
    distro cloudpickle msgspec pyzmq fastapi 'uvicorn[standard]' openai pydantic \
    prometheus-client tiktoken sentencepiece protobuf psutil py-cpuinfo cbor2 \
    gguf einops blake3 partial-json-parser outlines outlines-core lm-format-enforcer \
    diskcache mistral-common ray starlette anyio sniffio h11 httptools websockets \
    watchfiles python-multipart openai-harmony pybase64 cachetools uvloop \
    xgrammar llguidance opentelemetry-api opentelemetry-sdk pyre-extensions tabulate \
    safetensors tokenizers regex tqdm requests filelock httpcore httpx \
    huggingface-hub hf-xet fsspec PyYAML packaging dill multiprocess pandas pyarrow \
    typing-extensions typing-inspection annotated-types pydantic-core depyf astor \
    jinja2 markupsafe sympy networkx mpmath

echo ""
echo "=== Step 6: undo damage from xgrammar's vanilla 'triton' dep ==="
# xgrammar transitively pulls `triton` (vanilla), which writes to the same
# triton/ import path as triton-xpu and breaks it. Uninstall vanilla and
# force-reinstall triton-xpu.
VIRTUAL_ENV="${VENV}" uv pip uninstall triton 2>/dev/null || true
VIRTUAL_ENV="${VENV}" uv pip install --reinstall --no-cache --no-deps --link-mode=copy \
    --index-url https://download.pytorch.org/whl/xpu "triton-xpu==3.7.1"

echo ""
echo "=== Step 6.5: deps for upstream torchtitan.experiments.rl ==="
# torchdata, tyro, spmd-types, tensorboard, wandb, renderers (Prime Intellect).
# These are pulled in transitively by upstream rl/ but not by bare vllm-xpu.
VIRTUAL_ENV="${VENV}" uv pip install --no-cache --link-mode=copy \
    torchdata tyro spmd-types tensorboard wandb
VIRTUAL_ENV="${VENV}" uv pip install --no-cache --link-mode=copy \
    "renderers @ git+https://github.com/PrimeIntellect-ai/renderers.git@main"

echo ""
echo "=== Step 7: trl, transformers, accelerate, datasets ==="
VIRTUAL_ENV="${VENV}" uv pip install --no-cache --link-mode=copy \
    "trl==1.6.0" "transformers==5.11.0" "accelerate==1.14.0" "datasets==5.0.0"

echo ""
echo "=== Step 8: torchmonarch + torchstore + stragglers ==="
VIRTUAL_ENV="${VENV}" uv pip install --no-cache --link-mode=copy \
    "torchmonarch==0.5.0"
VIRTUAL_ENV="${VENV}" uv pip install --no-cache --no-deps --link-mode=copy \
    "torchstore @ git+https://github.com/meta-pytorch/torchstore.git"
VIRTUAL_ENV="${VENV}" uv pip install --no-cache --link-mode=copy \
    pygtrie pyelftools portpicker pycountry msgpack jsonpath-ng jsonschema genson \
    interegular markdown-it-py mdurl pygments rich pillow referencing rpds-py \
    pydantic-extra-types pyzes apache-tvm-ffi tcmlib intel-cmplr-lic-rt

echo ""
echo "=== Step 8.5: uninstall impi-rt + in-venv oneCCL (CRITICAL) ==="
# torch 2.12+xpu pulls these as deps, and they install in-venv copies
# of libccl.so and libmpi.so that SHADOW the system /opt/aurora/.../oneapi
# stack at runtime. The in-venv oneCCL doesn't know about Sunspot's USM
# allocator and rejects every XCCL broadcast with
#   "ccl_check_usm_pointers: invalid usm pointer type: unknown".
# Removing them lets torch's libtorch_xpu.so resolve libccl.so → the
# system module-loaded one, which works correctly.
# Verify with: ldd venvs/rl-vllm/lib/python3.12/site-packages/torch/lib/libtorch_xpu.so | grep ccl
# should point to /opt/aurora/.../oneapi/ccl/latest/lib, not venvs/rl-vllm/lib.
VIRTUAL_ENV="${VENV}" uv pip uninstall impi-rt oneccl oneccl-devel 2>/dev/null || true

echo ""
echo "=== Step 9: verify full stack ==="
"${VENV}/bin/python" -c "
import torch; print('torch', torch.__version__)
import triton; print('triton', triton.__version__)
from triton._C.libtriton import intel; print('intel symbols OK')
from triton.language import target_info; print('target_info OK')
import vllm; print('vllm', vllm.__version__)
import vllm_xpu_kernels; print('vllm_xpu_kernels OK')
import monarch
from monarch.actor import Actor, endpoint, this_host
print('monarch.actor OK')
import torchstore
from torchstore.transport import TransportType
print('torchstore transports:', list(TransportType.__members__.keys()))
import trl; print('trl', trl.__version__)
import transformers; print('transformers', transformers.__version__)
import accelerate; print('accelerate', accelerate.__version__)
import datasets; print('datasets', datasets.__version__)
print()
print('venvs/rl-vllm/ build complete ✓')
"
