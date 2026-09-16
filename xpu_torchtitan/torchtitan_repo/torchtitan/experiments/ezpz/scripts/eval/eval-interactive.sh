#!/bin/bash --login
# Interactive eval script — run on compute node via SSH
# Usage: ssh <node> "PBS_JOBID=... PBS_NODEFILE=... bash /path/to/.eval-interactive.sh"

export PBS_JOBID="${PBS_JOBID}"
export PBS_NODEFILE="${PBS_NODEFILE}"
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env

cd /lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz
export PYTHONPATH="$(pwd):${PYTHONPATH}"

echo "=== Environment ==="
echo "Python: $(which python3)"
python3 -c "import torch; print(f'Torch: {torch.__version__}')"
python3 -c "import lm_eval; print(f'lm_eval: {lm_eval.__version__}')" 2>&1
python3 -c "import vllm; print(f'vllm: {vllm.__version__}')" 2>&1
python3 -c "import torchtitan; print(f'torchtitan: {torchtitan.__file__}')"

echo ""
echo "=== Converting 2B step-5000 DCP → HF ==="
time python3 torchtitan/experiments/ezpz/eval/convert_to_hf.py \
    outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs3072/step-5000 \
    outputs/evals/agpt-2b/step-5000/hf \
    --model_name experiments.ezpz.agpt \
    --model_flavor 2b \
    --hf_assets_path assets/hf/gemma-7b \
    --export_dtype bfloat16 \
    2>&1

RC=$?
echo ""
echo "=== Conversion exit code: ${RC} ==="
if [[ $RC -eq 0 ]]; then
    echo "=== Output files ==="
    ls -lh outputs/evals/agpt-2b/step-5000/hf/ 2>/dev/null | head -15

    echo ""
    echo "=== Copying tokenizer + config ==="
    cp torchtitan/experiments/ezpz/eval/configs/agpt_2b_config.json outputs/evals/agpt-2b/step-5000/hf/config.json
    cp assets/hf/gemma-7b/tokenizer.json outputs/evals/agpt-2b/step-5000/hf/
    cp assets/hf/gemma-7b/tokenizer.model outputs/evals/agpt-2b/step-5000/hf/
    cp assets/hf/gemma-7b/tokenizer_config.json outputs/evals/agpt-2b/step-5000/hf/
    cp assets/hf/gemma-7b/special_tokens_map.json outputs/evals/agpt-2b/step-5000/hf/
    echo "Done."
    ls -lh outputs/evals/agpt-2b/step-5000/hf/

    echo ""
    echo "=== Quick sanity check: load with transformers ==="
    python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
print('Loading model...')
model = AutoModelForCausalLM.from_pretrained('outputs/evals/agpt-2b/step-5000/hf', torch_dtype=torch.bfloat16, device_map='cpu')
print(f'Model loaded: {model.config.hidden_size}d, {model.config.num_hidden_layers}L, {model.config.vocab_size} vocab')
print(f'Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B')
tok = AutoTokenizer.from_pretrained('outputs/evals/agpt-2b/step-5000/hf')
print(f'Tokenizer vocab: {tok.vocab_size}')
print('Sanity check PASSED')
" 2>&1
fi
