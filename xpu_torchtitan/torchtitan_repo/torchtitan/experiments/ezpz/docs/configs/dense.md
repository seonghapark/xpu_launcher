# Dense Models: AuroraGPT-{2,20}B

Multi-stage Pre-Training for AuroraGPT-{2, 20}B dense models using saforem2/torchtitan@ezpz.

- W\&B Report:
  [**Pre-Training AuroraGPT-20B with TorchTitan**](https://api.wandb.ai/links/aurora_gpt/ibv5q8u0)
- Training Scripts:
  - AuroraGPT-2B: [submit_agpt_2B.sh](../../submit/aurora/submit_agpt_2b.sh)
  - AuroraGPT-20B: [submit_agpt_20B.sh](../../submit/aurora/submit_agpt_20b.sh)

## Training Configs

- Model implementations:
  [torchtitan/experiments/ezpz/agpt/__init__.py](../../agpt/__init__.py)
- Training Configs:
  [torchtitan/experiments/ezpz/agpt/config_registry.py](../../agpt/config_registry.py)

|                       | AGPT-2B           | AGPT-20B          |
|-----------------------|-------------------|-------------------|
| **Architecture**      |                   |                   |
| Model dim             | 2048              | 5120              |
| Layers                | 12                | 64                |
| Heads                 | 16                | 40                |
| KV heads              | 4                 | 8                 |
| Hidden dim (FFN)      | 11008             | ~13312            |
| Vocab size            | 256128            | 256128            |
| RoPE theta            | 50000             | 500000            |
| Attention backend     | sdpa              | sdpa              |
| **Training**          |                   |                   |
| Nodes                 | 256               | 256               |
| GPUs                  | 3072 (256×12)     | 3072 (256×12)     |
| Seq length            | 8192              | 8192              |
| Local batch size      | 2 (default)       | 1                 |
| Global batch size     | 6144              | 6144              |
| Training steps        | 92,859            | 92,859            |
| Total tokens          | ~4.67T            | ~4.67T            |
| Activation checkpoint | selective (op)    | full              |
| **Optimizer**         |                   |                   |
| Type                  | SophiaG           | SophiaG           |
| Learning rate         | 2.28e-5           | 2.28e-5           |
| **Data**              |                   |                   |
| Dataset               | blendcorpus       | blendcorpus       |
| Data list             | olmo-mix-1124.txt | olmo-mix-1124.txt |
| **Checkpointing**     |                   |                   |
| Interval              | 100 steps         | 100 steps         |
| Keep latest k         | 0 (keep all)      | 0 (keep all)      |
