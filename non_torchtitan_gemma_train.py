#!/usr/bin/env python3
"""Minimal non-TorchTitan Gemma training smoke test for XPU.

This intentionally avoids TorchTitan. It loads a Hugging Face causal LM, builds a
small token batch from Hugging Face datasets (or synthetic tokens), runs one or
more forward/backward/optimizer steps, and prints a JSON summary. By default only
lm_head is trainable to keep the smoke run small; pass --train-mode full only
when you really want full-model training.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path
from typing import Any


DATASET_ALIASES = {
    "multinews": "Awesome075/multi_news_parquet",
    "multi-news": "Awesome075/multi_news_parquet",
    "multi_news": "Awesome075/multi_news_parquet",
    "pg19": "emozilla/pg19",
    "pg-19": "emozilla/pg19",
}

TEXT_FIELDS_BY_DATASET = {
    "Awesome075/multi_news_parquet": ("document", "summary"),
    "emozilla/pg19": ("text",),
    "multi_news": ("document", "summary"),
    "pg19": ("text",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="HF model directory.")
    parser.add_argument(
        "--dataset-path",
        default="synthetic",
        help="Comma-separated HF dataset names/aliases (pg19,multi_news) or synthetic.",
    )
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument(
        "--validation-dataset-path",
        default=None,
        help="Comma-separated HF validation dataset names/aliases. Default: --dataset-path.",
    )
    parser.add_argument("--validation-dataset-split", default="validation")
    parser.add_argument("--validation-every", type=int, default=1)
    parser.add_argument("--disable-validation", action="store_true")
    parser.add_argument("--dataset-cache-dir", default=None)
    parser.add_argument("--dataset-max-samples", type=int, default=64)
    parser.add_argument(
        "--dataset-streaming",
        action="store_true",
        help="Stream datasets instead of materializing full downloads.",
    )
    parser.add_argument("--log-dir", required=True, help="Directory for run summary.")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument(
        "--seq-len",
        default="128",
        help="Input sequence length in tokens, or 'max' to use the model maximum.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
        help="Micro-batch size used for lm_head training to reduce activation memory.",
    )
    parser.add_argument(
        "--logit-chunk-size",
        type=int,
        default=512,
        help="Sequence chunk size used for lm_head logits/loss computation.",
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", default="xpu", choices=("xpu", "cpu"))
    parser.add_argument("--dtype", default="bfloat16", choices=("auto", "bfloat16", "float16", "float32"))
    parser.add_argument("--train-mode", default="lm_head", choices=("lm_head", "full"))
    parser.add_argument("--require-ccl", action="store_true", help="Fail distributed runs when oneCCL bindings are unavailable.")
    parser.add_argument(
        "--save-after-minutes",
        type=float,
        default=45.0,
        help="Save the current model after this many minutes from training start.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory for time-based model checkpoints. Default: <log-dir>/checkpoints.",
    )
    parser.add_argument(
        "--disable-time-checkpoint",
        action="store_true",
        help="Disable the 45-minute time-based model checkpoint.",
    )
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "xpu-non-torchtitan-gemma"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-run-name", default=os.environ.get("WANDB_RUN_NAME"))
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def normalize_dataset_names(dataset_path: str) -> list[str]:
    names = []
    for raw_name in dataset_path.split(","):
        key = raw_name.strip().lower()
        if not key:
            continue
        names.append(DATASET_ALIASES.get(key, raw_name.strip()))
    return names


def dataset_name_pairs(dataset_path: str) -> list[tuple[str, str]]:
    pairs = []
    for raw_name in dataset_path.split(","):
        requested = raw_name.strip()
        if not requested:
            continue
        resolved = DATASET_ALIASES.get(requested.lower(), requested)
        pairs.append((requested, resolved))
    return pairs


def text_from_sample(dataset_name: str, sample: dict[str, Any]) -> str:
    fields = TEXT_FIELDS_BY_DATASET.get(dataset_name, ("text", "document", "summary"))
    chunks = []
    for field in fields:
        value = sample.get(field)
        if isinstance(value, str) and value.strip():
            chunks.append(value.strip())
    if chunks:
        return "\n\n".join(chunks)
    for value in sample.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def build_input_ids_from_datasets(args: argparse.Namespace, tokenizer: Any, torch: Any, device: Any) -> tuple[Any, dict[str, Any]]:
    if args.dataset_path == "synthetic":
        vocab_size = int(getattr(tokenizer, "vocab_size", 256000))
        input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(args.batch_size, args.seq_len),
            device=device,
            dtype=torch.long,
        )
        return input_ids, {
            "dataset_mode": "synthetic",
            "dataset_names": [],
            "dataset_samples_used": 0,
            "dataset_cache_dir": args.dataset_cache_dir,
            "dataset_streaming": False,
        }

    from datasets import load_dataset

    dataset_pairs = dataset_name_pairs(args.dataset_path)
    if not dataset_pairs:
        raise ValueError("--dataset-path did not contain any dataset names")
    dataset_names = [resolved for _requested, resolved in dataset_pairs]

    token_buffer: list[int] = []
    samples_used = 0
    target_tokens = args.batch_size * args.seq_len
    max_samples_per_dataset = max(1, args.dataset_max_samples)
    min_tokens_per_dataset = max(1, target_tokens // len(dataset_pairs))
    samples_used_by_dataset: dict[str, int] = {}

    for _requested_name, dataset_name in dataset_pairs:
        dataset_token_count = 0
        dataset_samples_used = 0
        dataset = load_dataset(
            dataset_name,
            split=args.dataset_split,
            cache_dir=args.dataset_cache_dir,
            streaming=args.dataset_streaming,
        )
        iterator = iter(dataset)
        for sample in itertools.islice(iterator, max_samples_per_dataset):
            text = text_from_sample(dataset_name, sample)
            if not text:
                continue
            token_ids = tokenizer.encode(text, add_special_tokens=True)
            if token_ids:
                token_buffer.extend(token_ids)
                samples_used += 1
                dataset_samples_used += 1
                dataset_token_count += len(token_ids)
            if dataset_token_count >= min_tokens_per_dataset and len(token_buffer) >= target_tokens:
                break
        samples_used_by_dataset[dataset_name] = dataset_samples_used

    if len(token_buffer) < target_tokens:
        raise RuntimeError(
            f"not enough tokens from datasets {dataset_names}: "
            f"needed {target_tokens}, got {len(token_buffer)} from {samples_used} samples"
        )

    input_ids = torch.tensor(
        token_buffer[:target_tokens],
        device=device,
        dtype=torch.long,
    ).reshape(args.batch_size, args.seq_len)
    return input_ids, {
        "dataset_mode": "huggingface",
        "dataset_requested_names": [requested for requested, _resolved in dataset_pairs],
        "dataset_names": dataset_names,
        "dataset_split": args.dataset_split,
        "dataset_samples_used": samples_used,
        "dataset_samples_used_by_dataset": samples_used_by_dataset,
        "dataset_cache_dir": args.dataset_cache_dir,
        "dataset_streaming": args.dataset_streaming,
    }


def dtype_from_name(torch: Any, name: str) -> Any:
    if name == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def resolve_seq_len(seq_len: str, model_config: Any) -> int:
    if seq_len.lower() != "max":
        resolved = int(seq_len)
        if resolved < 2:
            raise ValueError("--seq-len must be >= 2")
        return resolved
    for field_name in ("max_position_embeddings", "n_positions", "seq_length"):
        value = getattr(model_config, field_name, None)
        if isinstance(value, int) and value >= 2:
            return value
    raise ValueError("--seq-len max requested, but the model config does not expose a usable max sequence length")


def import_acceleration_modules() -> dict[str, Any]:
    info: dict[str, Any] = {
        "ipex_available": False,
        "ipex_version": "",
        "ipex_error": "",
        "oneccl_available": False,
        "oneccl_version": "",
        "oneccl_error": "",
    }
    try:
        import intel_extension_for_pytorch as ipex

        info["ipex_available"] = True
        info["ipex_version"] = getattr(ipex, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - depends on cluster env
        info["ipex_error"] = f"{exc.__class__.__name__}: {exc}"

    try:
        import oneccl_bindings_for_pytorch as ccl

        info["oneccl_available"] = True
        info["oneccl_version"] = getattr(ccl, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - depends on cluster env
        info["oneccl_error"] = f"{exc.__class__.__name__}: {exc}"

    return info


def maybe_init_distributed(require_ccl: bool) -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE") or os.environ.get("PMI_SIZE") or "1")
    rank = int(os.environ.get("RANK") or os.environ.get("PMI_RANK") or "0")
    local_rank = int(
        os.environ.get("LOCAL_RANK")
        or os.environ.get("PMI_LOCAL_RANK")
        or os.environ.get("PALS_LOCAL_RANKID")
        or "0"
    )
    ccl_available = False
    ccl_error = ""

    try:
        import oneccl_bindings_for_pytorch  # noqa: F401

        ccl_available = True
    except Exception as exc:  # pragma: no cover - depends on cluster env
        ccl_error = f"{exc.__class__.__name__}: {exc}"

    if world_size > 1:
        if not ccl_available and require_ccl:
            raise RuntimeError(
                "WORLD_SIZE > 1 requires oneccl_bindings_for_pytorch for this XPU smoke test; "
                f"import failed with {ccl_error}"
            )
        backend = "ccl" if ccl_available else "gloo"
        dist.init_process_group(backend=backend)
    else:
        backend = "none"

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device_count = torch.xpu.device_count()
        if device_count > 0:
            torch.xpu.set_device(local_rank % device_count)
    else:
        device_count = 0

    return {
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "distributed_backend": backend,
        "ccl_available": ccl_available,
        "ccl_error": ccl_error,
        "xpu_device_count": device_count,
    }


def set_trainable_params(model: Any, mode: str) -> int:
    if mode == "full":
        for param in model.parameters():
            param.requires_grad_(True)
    else:
        for param in model.parameters():
            param.requires_grad_(False)
        if hasattr(model, "lm_head"):
            for param in model.lm_head.parameters():
                param.requires_grad_(True)
        else:
            raise RuntimeError("model has no lm_head; use --train-mode full")
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def save_time_checkpoint(
    model: Any,
    tokenizer: Any,
    checkpoint_root: Path,
    *,
    step: int,
    elapsed_seconds: float,
) -> Path:
    checkpoint_path = checkpoint_root / f"time_checkpoint_step{step}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_path, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint_path)
    metadata = {
        "step": step,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_minutes": elapsed_seconds / 60.0,
        "reason": "time_checkpoint",
    }
    (checkpoint_path / "checkpoint_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return checkpoint_path


def causal_lm_accuracy(torch: Any, logits: Any, labels: Any) -> float:
    if labels.shape[-1] < 2:
        return 0.0
    predictions = logits[:, :-1, :].argmax(dim=-1)
    targets = labels[:, 1:]
    return float((predictions == targets).to(torch.float32).mean().detach().cpu())


def evaluate_batch(model: Any, input_ids: Any, torch: Any) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = float(outputs.loss.detach().cpu())
        accuracy = causal_lm_accuracy(torch, outputs.logits, input_ids)
    if was_training:
        model.train()
    return loss, accuracy


def lm_head_chunk_loss_and_accuracy(
    model: Any,
    hidden_states: Any,
    labels: Any,
    torch: Any,
    *,
    logit_chunk_size: int,
    total_loss_tokens: int | None = None,
    backward: bool = False,
) -> tuple[float, float]:
    import torch.nn.functional as F

    if labels.shape[-1] < 2:
        return 0.0, 0.0
    chunk_size = max(1, logit_chunk_size)
    total_targets = labels[:, 1:].numel()
    loss_denominator = total_loss_tokens or total_targets
    loss_sum = 0.0
    correct = 0
    seen = 0
    for start in range(0, labels.shape[-1] - 1, chunk_size):
        end = min(labels.shape[-1] - 1, start + chunk_size)
        logits = model.lm_head(hidden_states[:, start:end, :])
        targets = labels[:, start + 1 : end + 1]
        chunk_loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="sum",
        )
        if backward:
            (chunk_loss_sum / loss_denominator).backward()
        loss_sum += float(chunk_loss_sum.detach().cpu())
        predictions = logits.argmax(dim=-1)
        correct += int((predictions == targets).sum().detach().cpu())
        seen += targets.numel()
        del logits, targets, chunk_loss_sum, predictions
    return loss_sum / total_targets, (correct / seen if seen else 0.0)


def forward_backbone_no_grad(model: Any, input_ids: Any, torch: Any) -> Any:
    with torch.no_grad():
        outputs = model.model(input_ids=input_ids, use_cache=False)
    return outputs.last_hidden_state


def train_lm_head_step(model: Any, input_ids: Any, optimizer: Any, torch: Any, args: argparse.Namespace) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    total_tokens = input_ids[:, 1:].numel()
    loss_weighted_sum = 0.0
    accuracy_weighted_sum = 0.0
    seen_tokens = 0
    micro_batch_size = max(1, min(args.micro_batch_size, input_ids.shape[0]))
    for start in range(0, input_ids.shape[0], micro_batch_size):
        micro_input_ids = input_ids[start : start + micro_batch_size]
        hidden_states = forward_backbone_no_grad(model, micro_input_ids, torch)
        micro_loss, micro_accuracy = lm_head_chunk_loss_and_accuracy(
            model,
            hidden_states,
            micro_input_ids,
            torch,
            logit_chunk_size=args.logit_chunk_size,
            total_loss_tokens=total_tokens,
            backward=True,
        )
        micro_tokens = micro_input_ids[:, 1:].numel()
        loss_weighted_sum += micro_loss * micro_tokens
        accuracy_weighted_sum += micro_accuracy * micro_tokens
        seen_tokens += micro_tokens
        del micro_input_ids, hidden_states
    optimizer.step()
    return loss_weighted_sum / seen_tokens, accuracy_weighted_sum / seen_tokens


def evaluate_lm_head_batch(model: Any, input_ids: Any, torch: Any, args: argparse.Namespace) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    loss_weighted_sum = 0.0
    accuracy_weighted_sum = 0.0
    seen_tokens = 0
    micro_batch_size = max(1, min(args.micro_batch_size, input_ids.shape[0]))
    with torch.no_grad():
        for start in range(0, input_ids.shape[0], micro_batch_size):
            micro_input_ids = input_ids[start : start + micro_batch_size]
            outputs = model.model(input_ids=micro_input_ids, use_cache=False)
            micro_loss, micro_accuracy = lm_head_chunk_loss_and_accuracy(
                model,
                outputs.last_hidden_state,
                micro_input_ids,
                torch,
                logit_chunk_size=args.logit_chunk_size,
            )
            micro_tokens = micro_input_ids[:, 1:].numel()
            loss_weighted_sum += micro_loss * micro_tokens
            accuracy_weighted_sum += micro_accuracy * micro_tokens
            seen_tokens += micro_tokens
            del micro_input_ids, outputs
    if was_training:
        model.train()
    return loss_weighted_sum / seen_tokens, accuracy_weighted_sum / seen_tokens


def prefix_keys(values: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in values.items()}


def maybe_init_wandb(args: argparse.Namespace, log_dir: Path, rank: int, config: dict[str, Any]) -> Any | None:
    if args.disable_wandb or rank != 0:
        return None
    try:
        import wandb
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("wandb logging is enabled, but importing wandb failed; install wandb or pass --disable-wandb") from exc
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        dir=str(log_dir),
        config=config,
    )


def main() -> int:
    args = parse_args()
    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    accel_info = import_acceleration_modules()
    dist_info = maybe_init_distributed(args.require_ccl)
    rank = dist_info["rank"]

    model_path = Path(args.model_path).resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype_from_name(torch, args.dtype),
        low_cpu_mem_usage=True,
    )
    args.seq_len = resolve_seq_len(args.seq_len, model.config)
    if args.micro_batch_size < 1:
        raise ValueError("--micro-batch-size must be >= 1")
    if args.logit_chunk_size < 1:
        raise ValueError("--logit-chunk-size must be >= 1")

    if args.device == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("requested --device xpu, but torch.xpu is not available")
        device = torch.device("xpu", dist_info["local_rank"] % max(1, dist_info["xpu_device_count"]))
    else:
        device = torch.device("cpu")

    model.to(device)
    model.train()
    trainable_params = set_trainable_params(model, args.train_mode)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    input_ids, dataset_info = build_input_ids_from_datasets(args, tokenizer, torch, device)
    validation_input_ids = None
    validation_info: dict[str, Any] = {"validation_enabled": not args.disable_validation}
    if not args.disable_validation:
        validation_args = argparse.Namespace(**vars(args))
        validation_args.dataset_path = args.validation_dataset_path or args.dataset_path
        validation_args.dataset_split = args.validation_dataset_split
        validation_input_ids, raw_validation_info = build_input_ids_from_datasets(validation_args, tokenizer, torch, device)
        validation_info.update(prefix_keys(raw_validation_info, "validation_"))

    wandb_run = maybe_init_wandb(
        args,
        log_dir,
        rank,
        {
            "model_path": str(model_path),
            "device": str(device),
            "dtype": args.dtype,
            "steps": args.steps,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "micro_batch_size": args.micro_batch_size,
            "logit_chunk_size": args.logit_chunk_size,
            "lr": args.lr,
            "dataset_path": args.dataset_path,
            "dataset_split": args.dataset_split,
            "validation_dataset_path": args.validation_dataset_path or args.dataset_path,
            "validation_dataset_split": args.validation_dataset_split,
            "train_mode": args.train_mode,
            "trainable_params": trainable_params,
            **dataset_info,
            **validation_info,
            **accel_info,
            **dist_info,
        },
    )

    losses: list[float] = []
    accuracies: list[float] = []
    validation_losses: list[float] = []
    validation_accuracies: list[float] = []
    training_start_time = time.monotonic()
    checkpoint_root = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else log_dir / "checkpoints"
    time_checkpoint_saved = False
    time_checkpoint_path = ""
    time_checkpoint_step = None
    save_after_seconds = max(0.0, args.save_after_minutes * 60.0)
    for step in range(1, args.steps + 1):
        if args.train_mode == "lm_head":
            train_loss, train_accuracy = train_lm_head_step(model, input_ids, optimizer, torch, args)
        else:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            train_accuracy = causal_lm_accuracy(torch, outputs.logits, input_ids)
            loss.backward()
            optimizer.step()
            train_loss = float(loss.detach().cpu())
        losses.append(train_loss)
        accuracies.append(train_accuracy)
        elapsed_seconds = time.monotonic() - training_start_time
        metrics = {
            "train/loss": train_loss,
            "train/accuracy": train_accuracy,
            "train/elapsed_seconds": elapsed_seconds,
        }
        if validation_input_ids is not None and step % max(1, args.validation_every) == 0:
            if args.train_mode == "lm_head":
                validation_loss, validation_accuracy = evaluate_lm_head_batch(model, validation_input_ids, torch, args)
            else:
                validation_loss, validation_accuracy = evaluate_batch(model, validation_input_ids, torch)
            validation_losses.append(validation_loss)
            validation_accuracies.append(validation_accuracy)
            metrics.update(
                {
                    "validation/loss": validation_loss,
                    "validation/accuracy": validation_accuracy,
                }
            )
        if wandb_run is not None:
            wandb_run.log(metrics, step=step)
        if (
            not args.disable_time_checkpoint
            and not time_checkpoint_saved
            and elapsed_seconds >= save_after_seconds
        ):
            if rank == 0:
                saved_path = save_time_checkpoint(
                    model,
                    tokenizer,
                    checkpoint_root,
                    step=step,
                    elapsed_seconds=elapsed_seconds,
                )
                time_checkpoint_path = str(saved_path)
            time_checkpoint_saved = True
            time_checkpoint_step = step
            if dist_info["world_size"] > 1:
                import torch.distributed as dist

                dist.barrier()

    training_elapsed_seconds = time.monotonic() - training_start_time

    summary = {
        "ok": True,
        "model_path": str(model_path),
        "device": str(device),
        "dtype": args.dtype,
        "steps": args.steps,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "micro_batch_size": args.micro_batch_size,
        "logit_chunk_size": args.logit_chunk_size,
        "dataset_path": args.dataset_path,
        "train_mode": args.train_mode,
        "trainable_params": trainable_params,
        "losses": losses,
        "accuracies": accuracies,
        "validation_losses": validation_losses,
        "validation_accuracies": validation_accuracies,
        "training_elapsed_seconds": training_elapsed_seconds,
        "time_checkpoint_enabled": not args.disable_time_checkpoint,
        "time_checkpoint_after_minutes": args.save_after_minutes,
        "time_checkpoint_saved": time_checkpoint_saved,
        "time_checkpoint_step": time_checkpoint_step,
        "time_checkpoint_path": time_checkpoint_path,
        "max_position_embeddings": getattr(model.config, "max_position_embeddings", None),
        "rope_scaling": getattr(model.config, "rope_scaling", None),
        **dataset_info,
        **validation_info,
        **accel_info,
        **dist_info,
    }
    summary_path = log_dir / f"non_torchtitan_gemma_train_rank{rank}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    if rank == 0:
        print(json.dumps(summary, indent=2, sort_keys=True))

    if wandb_run is not None:
        wandb_run.finish()

    if dist_info["world_size"] > 1:
        import torch.distributed as dist

        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
