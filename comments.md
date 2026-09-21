
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




## TorchTitan quick commands

All TorchTitan runs use `xpu_torchtitan/run_train_torchtitan.sh`.

**Single-node offline smoke dry-run**

```bash
cd /lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan

MODEL_PATH=./torchtitan_repo/tests/assets/tokenizer \
DATASET_NAME=c4_test \
TRAINING_STEPS=1 \
SEQ_LEN=128 \
./run_train_torchtitan.sh single --dry-run
```

Remove `--dry-run` to execute the training command. For multi-node execution
inside a PBS allocation, use `multi`; the wrapper reads `PBS_NODEFILE`.

```bash
MODEL_PATH=/path/to/model/assets \
MODULE=llama3 \
CONFIG=llama3_8b \
TRAINING_STEPS=3 \
SEQ_LEN=256 \
./run_train_torchtitan.sh multi
```

**Load the AGPT 2B DCP model weights**

```bash
CKPT=/lus/flare/projects/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144/step-92859

MODEL_PATH=/lus/flare/projects/datascience/seonghapark/agpt-2b-v2-256n-step-92859-safetensors \
MODULE=agpt \
CONFIG=agpt_2b \
./run_train_torchtitan.sh multi -- \
	--checkpoint.initial_load_path "$CKPT"
```

`MODEL_PATH` contains model/tokenizer assets. `MODULE` and `CONFIG` select the
TorchTitan architecture. The DCP path supplies model weights. This is
model-only initialization because the old checkpoint's SophiaG optimizer state
does not match the current AdamW configuration.

Everything after `--` is forwarded to TorchTitan. For example:

```bash
MODEL_PATH=/path/to/model/assets \
./run_train_torchtitan.sh single --dry-run -- \
	--training.steps 2 --training.seq_len 512
```

Useful defaults:

```text
MODULE=llama3
CONFIG=llama3_debugmodel
DATASET_NAME=pg19_multinews
TRAINING_STEPS=100
SEQ_LEN=16384
NPROC_PER_NODE=4
```

The launcher detects XPU/CUDA/ROCm automatically. `NPROC_PER_NODE` remains an
explicit topology setting and is not inferred from accelerator device count.
