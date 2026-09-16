#!/bin/bash --login
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

module load oneapi/release/2025.3.1 hdf5 pti-gpu frameworks/2025.3.1 2>/dev/null

cd /lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz

VENV="venvs/aurora/tt-lm-eval"

# Remove bad transformers install, use system 4.57.6
pip uninstall -y transformers 2>/dev/null
source "${VENV}/bin/activate"

echo "Python: $(which python3)"
python3 -c "import transformers; print(f'transformers: {transformers.__version__}')"
python3 -c "import lm_eval; print(f'lm_eval: {lm_eval.__version__}')"

echo ""
echo "=== Quick test with CUDA warmup patched ==="
python3 -c "
# Monkey-patch the CUDA warmup to be a no-op
import transformers.modeling_utils as mu
mu.caching_allocator_warmup = lambda *args, **kwargs: None
print('Patched caching_allocator_warmup')

from lm_eval import evaluator
from lm_eval.api.registry import get_model

results = evaluator.simple_evaluate(
    model='hf',
    model_args='pretrained=outputs/evals/agpt-2b/step-1000/hf',
    tasks=['hellaswag'],
    batch_size=4,
    num_fewshot=0,
    limit=100,
)

for task, metrics in results['results'].items():
    acc = metrics.get('acc_norm,none', '?')
    print(f'{task}: acc_norm={acc}')
"
