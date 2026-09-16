# Production Training — MoE 10B_2B_sdpa EP=12

**Status:** Planned

## Configuration (Draft)

| Field | Value |
|-------|-------|
| Model | moe_10b_2b_sdpa (9.41B total / 1.98B active) |
| Experts | 36, top_k=3 |
| EP | 12 |
| Vocab size | 256,128 (Gemma tokenizer) |
| Seq len | 8,192 |
| Optimizer | TBD (AdamW suggested LR=3.99e-4 from 7B LR finder) |
| Compile | TBD |

## Prerequisites

- LR finder at production scale (GBS >> 24)
- Verify EP=12 works at multi-node scale
- Determine optimal TP/EP/FSDP decomposition
