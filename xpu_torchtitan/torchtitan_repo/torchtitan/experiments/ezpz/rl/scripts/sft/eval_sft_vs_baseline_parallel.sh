#!/bin/bash --login
#
# Parallel lm-eval: 1 task per XPU tile, both models simultaneously
# on different compute nodes. Estimated wall time: ~30 min total
# (the slowest of the 7 multiple-choice tasks runs on its own tile).
#
# Skips gsm8k because:
#   (a) it's *generative* (lm-eval `generate_until`), one decode per
#       question, 1319 questions, ~7s/iter on a 2B model = 2.5h alone
#   (b) we'd be running it at 0-shot (the script's CLI flag is
#       --num_fewshot 0), and a 2B base model scores near-zero at
#       0-shot gsm8k because it has no examples to learn the
#       <num>####<answer> format from. The signal is dominated by
#       format adherence, not reasoning.
# If we care about math, re-run gsm8k separately at 5-shot.
#
# 7 tasks × 1 tile each = 7 of the 12 XPU tiles on a node.
# Spawn each as a background process, wait for all to finish,
# aggregate results.
#
# Usage (from login node, with multi-node allocation up):
#   bash torchtitan/experiments/ezpz/rl/scripts/sft/eval_sft_vs_baseline_parallel.sh

set -o pipefail

SUBMIT_DIR="${PBS_O_WORKDIR:-/lus/flare/projects/datascience/foremans/projects/saforem2/torchtitan}"
cd "${SUBMIT_DIR}"

# Pick two hosts from an active PBS allocation; fall back to localhost.
HOSTFILE="${PBS_NODEFILE:-/var/spool/pbs/aux/${PBS_JOBID:-12468401.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov}}"
if [[ ! -f "${HOSTFILE}" ]]; then
    echo "FATAL: no hostfile at ${HOSTFILE}"
    exit 1
fi
HOST_BASELINE=$(sed -n '1p' "${HOSTFILE}")
HOST_SFT=$(sed -n '2p' "${HOSTFILE}")
if [[ -z "${HOST_SFT}" ]]; then
    # Only one node — run sequentially on the same node, different tiles
    HOST_SFT="${HOST_BASELINE}"
    SAME_NODE=1
else
    SAME_NODE=0
fi
echo "baseline host: ${HOST_BASELINE}"
echo "sft host:      ${HOST_SFT} (same_node=${SAME_NODE})"

OUT_BASE="outputs/evals/aurora2b-sft-vs-baseline-parallel-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${OUT_BASE}"
echo "${OUT_BASE}" > /tmp/eval-parallel-out.txt

# 7 tasks, one per tile. Order is arbitrary but kept stable for log
# reproducibility. hellaswag is the largest (10042 q) so it gets tile 0
# (most likely to finish last; this gives it a head start).
TASKS=(hellaswag arc_easy arc_challenge winogrande piqa openbookqa boolq)
N_TASKS=${#TASKS[@]}

# Per-tile launch helper. Pinned to one XPU tile via
# ONEAPI_DEVICE_SELECTOR; lm-eval still sees device='xpu:0' inside the
# pinned env. Runs in the background so we can fan out across tiles.
launch_one_task() {
    local host="$1"
    local model_path="$2"
    local label="$3"
    local task="$4"
    local tile="$5"
    local out_dir="${OUT_BASE}/${label}/${task}"
    mkdir -p "${out_dir}"

    # Send the whole command over ssh in one heredoc; the eval venv +
    # transformers shim + lm_eval CLI all run on the compute node.
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${host}" bash -s <<EOF >"${out_dir}/eval.log" 2>&1 &
set -o pipefail
cd ${SUBMIT_DIR}
source /etc/profile.d/lmod.sh 2>/dev/null || source /etc/profile 2>/dev/null
module load oneapi/release/2025.3.1 hdf5 pti-gpu frameworks/2025.3.1 2>&1 | tail -1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ONEAPI_DEVICE_SELECTOR="level_zero:${tile}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
source venvs/sunspot/torchtitan-aurora_frameworks-2025.3.1/bin/activate

echo "=== ${label}:${task} tile=${tile} on \$(hostname) ==="
python3 -c "import torch; print('xpu count:', torch.xpu.device_count())"

python3 - <<'PYEOF'
# Translate lm_eval's dtype= kwarg to torch_dtype= for transformers 4.50.1
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
    '--tasks', '${task}',
    '--batch_size', '4',
    '--num_fewshot', '0',
    '--device', 'xpu:0',
    '--output_path', '${out_dir}/',
]
cli_evaluate()
PYEOF
echo "=== ${label}:${task} done ==="
EOF
}

echo ""
echo "=== Fanning out: ${N_TASKS} tasks per model across XPU tiles ==="
START=$(date +%s)

# Always run both models on the SAME node, SAME tile sequence,
# back-to-back. Cross-node variance (NUMA, NIC, neighbor noise) is
# too large to do a fair per-tile comparison if we split models
# across nodes — caught this empirically when sft ran ~4x slower
# than baseline simply because it landed on a different physical box.
# Trade-off: ~2x wall time (no model overlap), but the only thing
# different between the two runs is the model weights, which is what
# we actually want to compare.
HOST="${HOST_BASELINE}"
echo "running both models on ${HOST} (tiles 0..$((N_TASKS - 1))), back-to-back"

echo ""
echo "=== Model 1/2: baseline at $(date) ==="
for i in "${!TASKS[@]}"; do
    launch_one_task "${HOST}" "AuroraGPT-2B-sophiag-gs138650" "baseline" "${TASKS[$i]}" "$i"
done
wait
echo "=== baseline done at $(date) ==="

echo ""
echo "=== Model 2/2: sft-step729 at $(date) ==="
for i in "${!TASKS[@]}"; do
    launch_one_task "${HOST}" "outputs/sft/aurora2b-sophiag-tulu-mix-32n-gbs6144/checkpoint-729-hf" "sft-step729" "${TASKS[$i]}" "$i"
done
wait
echo "=== sft-step729 done at $(date) ==="

END=$(date +%s)
ELAPSED=$((END - START))
echo "elapsed: ${ELAPSED}s ($((ELAPSED / 60))m)"

echo ""
echo "=== Aggregating results ==="
python3 <<PYEOF
import glob, json, os
OUT = "${OUT_BASE}"
results = {}
for label in ('baseline', 'sft-step729'):
    results[label] = {}
    for task_dir in sorted(glob.glob(os.path.join(OUT, label, '*'))):
        task = os.path.basename(task_dir)
        # lm_eval writes results into pretrained-args subdir like:
        # <task_dir>/AuroraGPT-2B-sophiag-gs138650/results_<timestamp>.json
        candidates = glob.glob(os.path.join(task_dir, '**', 'results*.json'), recursive=True)
        if not candidates:
            print(f'  [warn] no results for {label}:{task}')
            continue
        with open(candidates[0]) as f:
            d = json.load(f)
        if 'results' in d and task in d['results']:
            results[label][task] = d['results'][task]
        else:
            print(f'  [warn] {label}:{task} json has no results.{task} key')

if results['baseline'] and results['sft-step729']:
    tasks = sorted(set(results['baseline']) & set(results['sft-step729']))
    print(f'{"task":<24} {"baseline":>12} {"sft-step729":>14} {"delta":>10}')
    print('-' * 64)
    for t in tasks:
        b, s = results['baseline'][t], results['sft-step729'][t]
        metric_keys = ['acc_norm,none', 'acc,none', 'exact_match,strict-match', 'exact_match,flexible-extract']
        bv, sv = None, None
        for k in metric_keys:
            if k in b and k in s and isinstance(b[k], (int, float)) and isinstance(s[k], (int, float)):
                bv, sv = b[k], s[k]
                break
        if bv is None:
            continue
        delta = sv - bv
        sign = '+' if delta >= 0 else ''
        print(f'{t:<24} {bv:>12.4f} {sv:>14.4f} {sign}{delta:>9.4f}')
else:
    print('  [error] missing results for one or both models')
PYEOF

echo ""
echo "=== Results in ${OUT_BASE} ==="
