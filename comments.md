
## Representative Long-Context Datasets
| Dataset                  | Purpose                       | Notes                                        |
| ------------------------ | ----------------------------- | -------------------------------------------- |
| **LongBench**            | Long-context comprehension/QA | Document-based tasks of varying types/lengths |
| **InfiniteBench**        | Very long context             | Evaluates 100K+ token long-context ability   |
| **RULER**                | Context retrieval             | Measures how well specific info is found in long context |
| **Needle-in-a-Haystack** | Context retrieval             | Tests finding specific info inside long documents |
| **PG19**                 | Language modeling             | Long-form book data                          |
| **Qasper**               | Long-document QA              | Academic papers + questions/answers          |
| **NarrativeQA**          | Long-document QA              | Comprehension of books/movie scripts         |
| **GovReport**            | Summarization                 | Summarizing long government reports          |
| **MultiNews**            | Multi-document summarization  | Aggregate summaries across news articles     |
| **SCROLLS**              | Long-document NLP             | Benchmark bundling several long-context tasks |




**Single-rank XPU 1-step Smoke**
```bash
ssh x4516c6s6b0n0
source /lus/flare/projects/datascience/seonghapark/llm_evaluation/venv/bin/activate
cd /lus/flare/projects/datascience/seonghapark/xpu_launcher

./run_gemma_non_torchtitan_xpu.sh single
```

**Dry Run**
To see what command `xpu launch` would build without actually training:

```bash
source /lus/flare/projects/datascience/seonghapark/llm_evaluation/venv/bin/activate
cd /lus/flare/projects/datascience/seonghapark/xpu_launcher

./run_gemma_non_torchtitan_xpu.sh single --dry-run
```

**Run with small tweaks**
For example, to change the sequence length and step count:

```bash
source /lus/flare/projects/datascience/seonghapark/llm_evaluation/venv/bin/activate
cd /lus/flare/projects/datascience/seonghapark/xpu_launcher

TRAIN_STEPS=3 \
SEQ_LEN=256 \
BATCH_SIZE=1 \
./run_gemma_non_torchtitan_xpu.sh single
```

**Passing extra train args**
Everything after `--` is forwarded to the Python train script.

```bash
source /lus/flare/projects/datascience/seonghapark/llm_evaluation/venv/bin/activate
cd /lus/flare/projects/datascience/seonghapark/xpu_launcher

./run_gemma_non_torchtitan_xpu.sh single -- --steps 2 --seq-len 512 --lr 5e-6
```

**Setting model/log paths**
```bash
source /lus/flare/projects/datascience/seonghapark/llm_evaluation/venv/bin/activate
cd /lus/flare/projects/datascience/seonghapark/xpu_launcher

MODEL_PATH=/lus/flare/projects/datascience/seonghapark/torchtitan/assets/hf/gemma-7b \
LOG_DIR=/lus/flare/projects/datascience/seonghapark/xpu_launcher/outputs/my_gemma_run \
./run_gemma_non_torchtitan_xpu.sh single
```

**Checking results**
```bash
cat /lus/flare/projects/datascience/seonghapark/xpu_launcher/outputs/my_gemma_run/non_torchtitan_gemma_train_rank0.json
```

The defaults already make for a safe smoke run:

```text
NPROC_PER_NODE=1
TRAIN_STEPS=1
SEQ_LEN=128
BATCH_SIZE=1
TRAIN_MODE=lm_head
DTYPE=bfloat16
DEVICE=xpu
```

Multi-rank is not recommended yet because `oneccl_bindings_for_pytorch` is missing. To just inspect the command shape anyway:

```bash
./run_gemma_non_torchtitan_xpu.sh multi /path/to/hostfile --dry-run
```
