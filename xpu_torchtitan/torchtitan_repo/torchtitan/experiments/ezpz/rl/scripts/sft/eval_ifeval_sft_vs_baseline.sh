#!/bin/bash --login
#
# IFEval (instruction-following eval): the actual metric SFT is supposed
# to improve. 541 generative prompts × 2 models. Single XPU tile each.
#
# IFEval scores how well a response follows verifiable structural
# instructions ("include the word X", "do not use commas", "respond
# in exactly 5 sentences"). A base LM scores low because it has no
# concept of "follow this prompt"; an SFT'd LM should score
# substantially higher.
#
# Runs both models on the SAME node, SAME tile, back-to-back —
# same fairness rule as the multi-task eval.

set -o pipefail

SUBMIT_DIR="${PBS_O_WORKDIR:-/lus/flare/projects/datascience/foremans/projects/saforem2/torchtitan}"
cd "${SUBMIT_DIR}"

source /etc/profile.d/lmod.sh 2>/dev/null || source /etc/profile 2>/dev/null
module load oneapi/release/2025.3.1 hdf5 pti-gpu frameworks/2025.3.1 2>&1 | tail -1

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ONEAPI_DEVICE_SELECTOR="level_zero:0"
export HF_HUB_ENABLE_HF_TRANSFER=0
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

source venvs/sunspot/torchtitan-aurora_frameworks-2025.3.1/bin/activate
python3 -c "import torch; print('xpu count:', torch.xpu.device_count())"

OUT_BASE="outputs/evals/aurora2b-ifeval-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${OUT_BASE}"
echo "${OUT_BASE}" > /tmp/eval-ifeval-out.txt

run_one() {
    local model_path="$1"
    local label="$2"
    local out_dir="${OUT_BASE}/${label}"
    mkdir -p "${out_dir}"
    echo ""
    echo "=== IFEval: ${label} at $(date) ==="
    time python3 - <<EOF 2>&1 | tee "${out_dir}/eval.log"
import transformers
_real = transformers.AutoModelForCausalLM.from_pretrained
def _shim(*args, **kwargs):
    if 'dtype' in kwargs and 'torch_dtype' not in kwargs:
        kwargs['torch_dtype'] = kwargs.pop('dtype')
    return _real(*args, **kwargs)
transformers.AutoModelForCausalLM.from_pretrained = _shim

import sys
from lm_eval.__main__ import cli_evaluate
sys.argv = [
    'lm_eval',
    '--model', 'hf',
    '--model_args', 'pretrained=${model_path}',
    '--tasks', 'ifeval',
    '--batch_size', '4',
    '--num_fewshot', '0',
    '--device', 'xpu:0',
    '--output_path', '${out_dir}/',
]
cli_evaluate()
EOF
    echo "=== ${label} done at $(date) ==="
}

run_one AuroraGPT-2B-sophiag-gs138650 baseline
run_one outputs/sft/aurora2b-sophiag-tulu-mix-32n-gbs6144/checkpoint-729-hf sft-step729

echo ""
echo "=== IFEval comparison ==="
python3 <<PYEOF
import glob, json, os
OUT = "${OUT_BASE}"
print(f'{"metric":<48} {"baseline":>10} {"sft-step729":>14} {"delta":>10}')
print('-' * 84)
b_paths = glob.glob(os.path.join(OUT, 'baseline', '**', 'results*.json'), recursive=True)
s_paths = glob.glob(os.path.join(OUT, 'sft-step729', '**', 'results*.json'), recursive=True)
if not b_paths or not s_paths:
    print('[warn] no results JSON found')
    raise SystemExit
b = json.load(open(b_paths[0]))['results']['ifeval']
s = json.load(open(s_paths[0]))['results']['ifeval']
for key in sorted(set(b) & set(s)):
    bv, sv = b[key], s[key]
    if not (isinstance(bv, (int, float)) and isinstance(sv, (int, float))):
        continue
    delta = sv - bv
    sign = '+' if delta >= 0 else ''
    print(f'{key:<48} {bv:>10.4f} {sv:>14.4f} {sign}{delta:>9.4f}')
PYEOF
