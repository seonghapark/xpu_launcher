#!/bin/bash --login
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
module load oneapi/release/2025.3.1 hdf5 pti-gpu frameworks/2025.3.1 2>/dev/null
export HF_HUB_ENABLE_HF_TRANSFER=0

cd /lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz
export PYTHONPATH="$(pwd)"

echo "=== Checking existing converted checkpoints ==="
python3 -c "
from safetensors import safe_open
for step in [1000, 4000, 10000]:
    try:
        f = safe_open(f'outputs/evals/agpt-2b/step-{step}/hf/model-00001-of-00001.safetensors', framework='pt')
        w = f.get_tensor('model.embed_tokens.weight')
        print(f'step-{step}: embed[0,:3] = {w[0,:3].tolist()}, norm = {w.norm():.2f}')
    except Exception as e:
        print(f'step-{step}: ERROR {e}')
"

echo ""
echo "=== Testing DCP load directly ==="
python3 -c "
import torch
import torch.distributed.checkpoint as dcp
from torchtitan.experiments.ezpz.agpt import model_registry
from torchtitan.components.checkpoint import ModelWrapper

spec = model_registry('2b')
config = spec.model

with torch.device('cpu'):
    model = config.build()
model = ModelWrapper(model)

sd = model._get_state_dict()

# Snapshot before
keys = list(sd.keys())[:3]
before = {k: sd[k].flatten()[:3].clone() for k in keys}
print('Before DCP load:')
for k in keys:
    print(f'  {k}: norm={sd[k].norm():.4f}, vals={before[k].tolist()}')

print()
print('Loading DCP from step-1000...')
try:
    dcp.load(sd, checkpoint_id='outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs3072/step-1000')
    print('DCP load succeeded')
except Exception as e:
    print(f'DCP load FAILED: {e}')

print()
print('After DCP load:')
for k in keys:
    changed = not torch.equal(before[k], sd[k].flatten()[:3])
    print(f'  {k}: norm={sd[k].norm():.4f}, vals={sd[k].flatten()[:3].tolist()}, changed={changed}')
"
