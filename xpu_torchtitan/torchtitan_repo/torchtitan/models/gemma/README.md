# Gemma-7B Instruction Fine-Tuning

Standalone instruction fine-tuning of `google/gemma-7b` using HuggingFace
Transformers (`GemmaForCausalLM`) directly, wrapped with FSDP2 for
distributed training. This folder does **not** use TorchTitan's core
`TrainSpec`/`Trainer` pipeline -- it's a self-contained SFT script.

## Files

| File | Purpose |
| ---- | ------- |
| [config.py](config.py) | `GemmaSFTConfig` dataclass (model, data, training args) |
| [model.py](model.py) | `GemmaModel` -- thin wrapper around `GemmaForCausalLM` |
| [tokenizer.py](tokenizer.py) | `build_tokenizer` -- installs the Gemma chat template |
| [sftdataset.py](sftdataset.py) | Alpaca-style `SFTDataset` + collator with response-only loss |
| [parallelize.py](parallelize.py) | `parallelize_gemma` -- FSDP2 wrapping with mixed precision |
| [train.py](train.py) | Main training entry point (`torchrun`-compatible) |
| [run_sft.sh](run_sft.sh) | Convenience `torchrun` launcher |

## Data format

Records must be Alpaca-style dicts:

```json
{"instruction": "...", "input": "", "output": "..."}
```

The `input` field is optional. You can either point to an HF Hub dataset
(default: `tatsu-lab/alpaca`) via `--dataset_name`, or to a local
JSON / JSONL file via `--dataset_local_path`.

When `--mask_instruction True` (default), loss is computed only on the
assistant response tokens.

## Quick start

Local weights (recommended, no HF auth needed):

```bash
NGPU=8 ./torchtitan/models/gemma/run_sft.sh \
    --dataset_name tatsu-lab/alpaca \
    --output_dir outputs/gemma-7b-alpaca \
    --per_device_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_epochs 3 \
    --lr 2e-5
```

Local JSONL:

```bash
NGPU=8 ./torchtitan/models/gemma/run_sft.sh \
    --dataset_local_path ./my_sft_data.jsonl \
    --output_dir outputs/gemma-7b-custom
```

Any additional flags are forwarded to `tyro`, so every field of
`GemmaSFTConfig` is exposed on the CLI.

## Output

- Intermediate: `outputs/.../step-<N>/` (when `--save_interval > 0`)
- Final: `outputs/.../final/`

Each checkpoint is a standard HuggingFace directory (`config.json`,
`model.safetensors*`, tokenizer files), directly loadable with
`GemmaForCausalLM.from_pretrained(save_dir)`.
