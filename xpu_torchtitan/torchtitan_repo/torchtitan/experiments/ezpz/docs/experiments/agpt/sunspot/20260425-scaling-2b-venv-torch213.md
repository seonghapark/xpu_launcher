# AuroraGPT-2B Scaling Study (torch 2.13, .venv)

> Sam Foreman
> 2026-04-25 (Sunspot)

## Configuration

| Field | Value |
|-------|-------|
| Date | 2026-04-25 |
| Machine | Sunspot |
| Model | AuroraGPT 2B (dense) |
| Torch | 2.13.0.dev20260418+xpu (.venv) |
| Compile | ON (model + loss) |
| AC | none |
| Optimizer | SophiaG (LR=2.28e-5) / AdamW (LR=3e-4) for n=4 |
| Seq Length | 8192 |
| Local Batch Size | 2 |
| Dataset | books (Sunspot) |
| FSDP | shard-only (dp_shard=-1) |
| TP | 1 |

## Results

| Nodes | GPUs | GBS | TPS/GPU | TFLOPS/GPU | MFU | Memory | Steps | Optimizer |
|------:|-----:|----:|--------:|-----------:|----:|-------:|------:|-----------|
| 2 | 24 | 48 | 7,142 | 82 | 27.6% | 44.57 GiB | 5,000 | SophiaG |
| 4 | 48 | 96 | 7,397 | 83 | 27.8% | 44.18 GiB | 300 | AdamW |
| 8 | 96 | 192 | 7,251 | 81 | 27.2% | 44.18 GiB | 5,000 | SophiaG |
| 16 | 192 | 384 | 6,866 | 77 | 25.8% | 44.00 GiB | 1,371 | SophiaG |
| 32 | 384 | 768 | 7,066 | 79 | 26.5% | 43.96 GiB | 100 | SophiaG |
| 64 | 768 | 1536 | 6,851 | 77 | 25.7% | 43.87 GiB | 25 | SophiaG |

## Scaling Efficiency

Weak scaling efficiency relative to n=2 (82 tflops baseline):

| Nodes | Efficiency |
|------:|-----------:|
| 2 | 100% |
| 4 | 101% |
| 8 | 99% |
| 16 | 94% |
| 32 | 96% |
| 64 | 94% |

- **2→8 nodes**: near-perfect scaling (~100%)
- **2→64 nodes**: 94% efficiency (6% overhead from cross-node FSDP all-reduces)
- Per-GPU throughput is stable at 77-83 tflops across all node counts

## Notes

- n=4 with SophiaG (LR=2.28e-5) diverged (NaN loss) repeatedly at GBS=96.
  Re-ran with AdamW (LR=3e-4) which converges cleanly. Throughput is identical
  regardless of optimizer, confirming the divergence is a hyperparameter issue.
- The n=16/32/64 runs show ~6% lower per-GPU tflops vs n=2/4/8, consistent
  with increased inter-node communication overhead for FSDP all-reduces.
- Memory usage is nearly constant (~44 GiB) across all node counts, as expected
  for weak scaling with fixed LBS.

## Environment

```bash
module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
source .venv/bin/activate
# torch 2.13.0.dev20260418+xpu
```

## Scripts

```bash
# Submit scaling study:
for N in 2 4 8 16 32 64; do
    qsub -l select=${N} -N "scale-2b-n${N}" \
         -v "DFL_NAME=books,TRAIN_TOKENS=100000000000" \
         torchtitan/experiments/ezpz/scripts/train_agpt_2b_venv.sh
done
```
