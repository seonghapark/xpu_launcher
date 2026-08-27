Updated todo list

아래 그대로 쓰시면 됩니다. 현재 검증된 경로 기준입니다.

**Single-rank XPU 1-step Smoke**
```bash
ssh x4516c6s6b0n0
source /lus/flare/projects/datascience/seonghapark/llm_evaluation/venv/bin/activate
cd /lus/flare/projects/datascience/seonghapark/xpu_launcher

./run_gemma_non_torchtitan_xpu.sh single
```

**Dry Run**
실제로 학습하지 않고 `xpu launch`가 어떤 명령을 만들지 보려면:

```bash
source /lus/flare/projects/datascience/seonghapark/llm_evaluation/venv/bin/activate
cd /lus/flare/projects/datascience/seonghapark/xpu_launcher

./run_gemma_non_torchtitan_xpu.sh single --dry-run
```

**조금 바꿔서 실행**
예를 들어 sequence length와 step 수를 바꾸려면:

```bash
source /lus/flare/projects/datascience/seonghapark/llm_evaluation/venv/bin/activate
cd /lus/flare/projects/datascience/seonghapark/xpu_launcher

TRAIN_STEPS=3 \
SEQ_LEN=256 \
BATCH_SIZE=1 \
./run_gemma_non_torchtitan_xpu.sh single
```

**추가 train arg 넘기기**
스크립트 뒤 `--` 이후는 Python train script로 전달됩니다.

```bash
source /lus/flare/projects/datascience/seonghapark/llm_evaluation/venv/bin/activate
cd /lus/flare/projects/datascience/seonghapark/xpu_launcher

./run_gemma_non_torchtitan_xpu.sh single -- --steps 2 --seq-len 512 --lr 5e-6
```

**모델/로그 경로 지정**
```bash
source /lus/flare/projects/datascience/seonghapark/llm_evaluation/venv/bin/activate
cd /lus/flare/projects/datascience/seonghapark/xpu_launcher

MODEL_PATH=/lus/flare/projects/datascience/seonghapark/torchtitan/assets/hf/gemma-7b \
LOG_DIR=/lus/flare/projects/datascience/seonghapark/xpu_launcher/outputs/my_gemma_run \
./run_gemma_non_torchtitan_xpu.sh single
```

**결과 확인**
```bash
cat /lus/flare/projects/datascience/seonghapark/xpu_launcher/outputs/my_gemma_run/non_torchtitan_gemma_train_rank0.json
```

기본값은 이미 안전한 smoke run입니다:

```text
NPROC_PER_NODE=1
TRAIN_STEPS=1
SEQ_LEN=128
BATCH_SIZE=1
TRAIN_MODE=lm_head
DTYPE=bfloat16
DEVICE=xpu
```

multi-rank는 현재 `oneccl_bindings_for_pytorch`가 없어서 아직 권장하지 않습니다. 그래도 command 형태만 보면:

```bash
./run_gemma_non_torchtitan_xpu.sh multi /path/to/hostfile --dry-run
```
