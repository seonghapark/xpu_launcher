
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
./run_torchtitan_xpu.sh single
```

**Dry Run**
To see what command `xpu launch` would build without actually training:

```bash
./run_torchtitan_xpu.sh single --dry-run
```

**Run with small tweaks**
For example, to change the sequence length and step count:

```bash
TRAIN_STEPS=3 \
SEQ_LEN=256 \
BATCH_SIZE=1 \
./run_torchtitan_xpu.sh single
```

**Passing extra train args**
Everything after `--` is forwarded to the Python train script.

```bash
./run_torchtitan_xpu.sh single -- --steps 2 --seq-len 512 --lr 5e-6
```

**Setting model/log paths**
```bash
MODEL_PATH=./assets/hf/gemma-7b \
LOG_DIR=./outputs/my_gemma_run \
./run_torchtitan_xpu.sh single
```

**Checking results**
```bash
cat xpu_launcher/outputs/my_run/torchtitan_train_rank0.json
```

The defaults already make for a safe smoke run:

```text
NPROC_PER_NODE=12 \
NNODES=10 \
SPARE_NODES=2 \
MODEL_PATH=../../agpt-2b-v2-256n-step-92859-safetensors \
CONFIG=agpt_2b \
SEQ_LEN=128
BATCH_SIZE=1
TRAIN_MODE=lm_head
DTYPE=bfloat16
DEVICE=xpu
```
