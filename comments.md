
## 대표적인 Long-Context Dataset
| Dataset                  | 용도                           | 특징                                 |
| ------------------------ | ---------------------------- | ---------------------------------- |
| **LongBench**            | Long-context 이해/QA 평가        | 다양한 task와 길이의 문서 기반 평가             |
| **InfiniteBench**        | 매우 긴 context 평가              | 100K+ token 수준의 long-context 능력 평가 |
| **RULER**                | Context retrieval 평가         | 긴 context에서 특정 정보를 얼마나 잘 찾는지 측정    |
| **Needle-in-a-Haystack** | Context retrieval            | 긴 문서 속 특정 정보를 찾는 능력 테스트            |
| **PG19**                 | Language modeling            | 장문의 책(book) 데이터                    |
| **Qasper**               | Long-document QA             | 학술 논문 + 질문/답변                      |
| **NarrativeQA**          | Long-document QA             | 책/영화 스크립트 기반 이해                    |
| **GovReport**            | Summarization                | 긴 정부 보고서 요약                        |
| **MultiNews**            | Multi-document summarization | 여러 뉴스 문서의 종합 요약                    |
| **SCROLLS**              | Long-document NLP            | 여러 long-context task를 묶은 benchmark |




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
