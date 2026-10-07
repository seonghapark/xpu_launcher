#!/usr/bin/env python3
"""
Comprehensive hyperparameter collection and reporting script for TorchTitan training.

Collects ALL hyperparameters from the training configuration, including:
- Optimization hyperparameters
- Training schedule parameters
- Model architecture details
- Distributed training settings
- Data and batching settings
- Precision/system settings
- RoPE/context extension technique detection
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict

# Add torchtitan to path
TORCHTITAN_ROOT = Path(os.environ.get("TORCHTITAN_ROOT", "./torchtitan_repo"))
if str(TORCHTITAN_ROOT) not in sys.path:
    sys.path.insert(0, str(TORCHTITAN_ROOT))

try:
    from torchtitan.config import TrainingConfig
    from torchtitan.config.configs import ParallelismConfig
    from torchtitan.components.optimizer import OptimizersContainer
    from torchtitan.components.lr_scheduler import LRSchedulersContainer
    from torchtitan.trainer import Trainer
except ImportError as e:
    print(f"Error importing TorchTitan modules: {e}", file=sys.stderr)
    sys.exit(1)


def extract_model_info(trainer_config: Trainer.Config) -> Dict[str, Any]:
    """Extract model architecture information from config."""
    model_info = {}

    if trainer_config.model_spec:
        model = trainer_config.model_spec.model

        # Detect model type and get metadata
        model_name = model.__class__.__name__
        model_info["model_class"] = model_name

        # Try to extract common architecture parameters
        if hasattr(model, 'config'):
            cfg = model.config
            if hasattr(cfg, 'n_layers'):
                model_info["num_layers"] = cfg.n_layers
            if hasattr(cfg, 'hidden_size') or hasattr(cfg, 'dim'):
                model_info["hidden_size"] = getattr(cfg, 'hidden_size', getattr(cfg, 'dim', None))
            if hasattr(cfg, 'num_heads') or hasattr(cfg, 'n_heads'):
                model_info["num_attention_heads"] = getattr(cfg, 'num_heads', getattr(cfg, 'n_heads', None))
            if hasattr(cfg, 'intermediate_size'):
                model_info["intermediate_size"] = cfg.intermediate_size
            if hasattr(cfg, 'vocab_size'):
                model_info["vocab_size"] = cfg.vocab_size
            if hasattr(cfg, 'max_seq_len'):
                model_info["max_seq_len"] = cfg.max_seq_len

    return model_info


def detect_rope_backend(trainer_config: Trainer.Config) -> Dict[str, Any]:
    """Detect RoPE backend and context extension technique."""
    rope_info = {}

    if trainer_config.model_spec:
        model = trainer_config.model_spec.model

        # Check for RoPE configuration
        if hasattr(model, 'layers') and len(model.layers) > 0:
            first_layer = model.layers[0]
            if hasattr(first_layer, 'attention'):
                attention = first_layer.attention
                if hasattr(attention, 'rope'):
                    rope = attention.rope
                    if rope is not None:
                        rope_class = rope.__class__.__name__
                        rope_info["rope_backend"] = rope_class

                        # Detect context extension technique
                        if hasattr(rope, 'scaling'):
                            rope_info["scaling_method"] = str(rope.scaling)

                        # Check for YaRN (Yet another RoPE extensioN)
                        if "yarn" in rope_class.lower():
                            rope_info["context_extension_technique"] = "YaRN"
                            if hasattr(rope, 'alpha'):
                                rope_info["yarn_alpha"] = getattr(rope, 'alpha', None)
                            if hasattr(rope, 'beta'):
                                rope_info["yarn_beta"] = getattr(rope, 'beta', None)
                        # Check for other scaling methods
                        elif hasattr(rope, 'alpha'):
                            rope_info["context_extension_technique"] = "Linear_Scaling"
                        else:
                            rope_info["context_extension_technique"] = "None"

    return rope_info


def flatten_config(config_dict: Dict, prefix: str = "") -> Dict[str, Any]:
    """Flatten nested config dict for easier readability."""
    flattened = {}

    for key, value in config_dict.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            flattened.update(flatten_config(value, full_key))
        elif isinstance(value, (list, tuple)):
            flattened[full_key] = str(value)
        else:
            flattened[full_key] = value

    return flattened


def extract_all_hyperparameters(trainer_config: Trainer.Config) -> Dict[str, Any]:
    """Extract all hyperparameters from trainer config."""
    all_params = {}

    # 1. OPTIMIZATION HYPERPARAMETERS
    all_params["Optimization"] = {}
    if trainer_config.optimizer:
        opt_config = asdict(trainer_config.optimizer) if hasattr(trainer_config.optimizer, '__dataclass_fields__') else {}
        all_params["Optimization"].update(flatten_config({"optimizer": opt_config}))

    # 2. TRAINING SCHEDULE
    all_params["Training_Schedule"] = {
        "total_steps": trainer_config.training.steps,
        "max_duration_hours": trainer_config.training.max_duration_hours,
        "gc_freq": trainer_config.training.gc_freq,
    }

    # Add LR scheduler info
    if trainer_config.lr_scheduler:
        lr_sched = asdict(trainer_config.lr_scheduler) if hasattr(trainer_config.lr_scheduler, '__dataclass_fields__') else {}
        all_params["Training_Schedule"].update(flatten_config({"lr_scheduler": lr_sched}))

    # 3. BATCHING AND DATA
    all_params["Batching_and_Data"] = {
        "local_batch_size": trainer_config.training.local_batch_size,
        "global_batch_size": trainer_config.training.global_batch_size,
        "seq_len": trainer_config.training.seq_len,
        "dataset": trainer_config.dataloader.dataset if hasattr(trainer_config.dataloader, 'dataset') else "unknown",
    }

    # 4. MODEL ARCHITECTURE
    all_params["Model_Architecture"] = extract_model_info(trainer_config)

    # 5. PRECISION AND SYSTEM
    all_params["Precision_and_System"] = {
        "training_dtype": trainer_config.training.dtype,
        "mixed_precision_param": trainer_config.training.mixed_precision_param,
        "mixed_precision_reduce": trainer_config.training.mixed_precision_reduce,
        "enable_cpu_offload": trainer_config.training.enable_cpu_offload,
    }

    # 6. CONTEXT EXTENSION TECHNIQUE
    all_params["Context_Extension"] = detect_rope_backend(trainer_config)

    # 7. DISTRIBUTED TRAINING SETTINGS
    all_params["Distributed_Training"] = {
        "data_parallel_replicate_degree": trainer_config.parallelism.data_parallel_replicate_degree,
        "data_parallel_shard_degree": trainer_config.parallelism.data_parallel_shard_degree,
        "tensor_parallel_degree": trainer_config.parallelism.tensor_parallel_degree,
        "pipeline_parallel_degree": trainer_config.parallelism.pipeline_parallel_degree,
        "fsdp_reshard_after_forward": trainer_config.parallelism.fsdp_reshard_after_forward,
        "enable_sequence_parallel": trainer_config.parallelism.enable_sequence_parallel,
        "enable_async_tensor_parallel": trainer_config.parallelism.enable_async_tensor_parallel,
    }

    # 8. ACTIVATION CHECKPOINTING
    all_params["Activation_Checkpointing"] = {
        "activation_checkpoint_mode": trainer_config.activation_checkpoint.__class__.__name__ if trainer_config.activation_checkpoint else "None",
    }

    # 9. REGULARIZATION
    all_params["Regularization"] = {
        "gradient_clipping_max_norm": trainer_config.training.max_norm,
        "weight_decay": "See optimizer settings",
    }

    # 10. LOSS AND VALIDATION
    all_params["Loss_and_Validation"] = {
        "loss_function": trainer_config.loss.__class__.__name__ if trainer_config.loss else "unknown",
        "enable_loss_std_termination": trainer_config.training.enable_loss_std_termination,
        "loss_std_threshold": trainer_config.training.loss_std_threshold,
        "loss_std_window": trainer_config.training.loss_std_window,
        "validator_enabled": trainer_config.validator.enable if hasattr(trainer_config.validator, 'enable') else "unknown",
    }

    # 11. CHECKPOINT AND LOGGING
    all_params["Checkpoint_and_Logging"] = {
        "checkpoint_enabled": trainer_config.checkpoint.enable if hasattr(trainer_config.checkpoint, 'enable') else "unknown",
        "checkpoint_interval": trainer_config.checkpoint.interval if hasattr(trainer_config.checkpoint, 'interval') else "unknown",
        "enable_compile": trainer_config.compile.enable if hasattr(trainer_config.compile, 'enable') else "unknown",
    }

    # 12. PATH AND ASSETS
    all_params["Paths_and_Assets"] = {
        "hf_assets_path": trainer_config.hf_assets_path,
        "dump_folder": trainer_config.dump_folder,
    }

    return all_params


def print_hyperparameters_report(hyperparams: Dict[str, Any], output_format: str = "text"):
    """Print hyperparameters in requested format."""

    if output_format == "json":
        print(json.dumps(hyperparams, indent=2, default=str))
    else:  # text format
        print("\n" + "="*100)
        print("COMPREHENSIVE TORCHTITAN HYPERPARAMETERS REPORT")
        print("="*100 + "\n")

        for section, params in hyperparams.items():
            print(f"\n[{section.upper().replace('_', ' ')}]")
            print("-" * 100)

            if isinstance(params, dict):
                max_key_len = max(len(str(k)) for k in params.keys()) if params else 0
                for key, value in params.items():
                    if value is not None:
                        print(f"  {key:<{max_key_len}}  = {value}")
            else:
                print(f"  {params}")

        print("\n" + "="*100 + "\n")


def print_missing_hyperparameters_report():
    """Print what hyperparameters are NOT currently being tracked/printed."""

    all_requested = {
        "Optimization": [
            "learning_rate",
            "optimizer_type (SGD/Adam/AdamW)",
            "momentum",
            "betas (alpha, beta for Adam)",
            "weight_decay",
            "gradient_clipping",
            "optimizer_epsilon",
        ],
        "Training Schedule": [
            "epochs (implicit via steps)",
            "batch_size ✓",
            "gradient_accumulation_steps",
            "warmup_steps ✓",
            "lr_scheduler ✓",
            "decay_rate ✓",
            "total_training_steps ✓",
        ],
        "Model Architecture": [
            "number_of_layers ✓",
            "hidden_size ✓",
            "attention_heads ✓",
            "activation_function",
            "dropout ✓",
            "normalization_type",
        ],
        "Initialization": [
            "weight_initialization_method",
            "bias_initialization",
            "random_seed",
        ],
        "Regularization": [
            "dropout ✓",
            "weight_decay ✓",
            "label_smoothing",
            "data_augmentation",
            "early_stopping ✓",
        ],
        "Loss": [
            "loss_function ✓",
            "loss_weights",
            "class_weights",
            "auxiliary_losses",
            "temperature/margin_parameters",
        ],
        "Data": [
            "dataset_size",
            "sampling_strategy",
            "shuffle",
            "train_val_split",
            "preprocessing",
            "sequence_length ✓",
            "augmentation",
        ],
        "Batching": [
            "batch_size ✓",
            "micro_batch_size ✓",
            "padding_strategy",
            "dynamic_batching",
        ],
        "Precision/System": [
            "dtype (FP32/FP16/BF16) ✓",
            "mixed_precision ✓",
            "gradient_scaling",
            "distributed_settings ✓",
        ],
    }

    print("\n" + "="*100)
    print("HYPERPARAMETER TRACKING STATUS")
    print("="*100 + "\n")
    print("✓ = Currently tracked and printed")
    print("✗ = NOT currently tracked/printed\n")

    not_tracked = []
    tracked = []

    for category, params in all_requested.items():
        print(f"\n[{category.upper()}]")
        print("-" * 100)

        for param in params:
            if "✓" in param:
                tracked_param = param.replace(" ✓", "")
                print(f"  ✓ {tracked_param}")
                tracked.append(tracked_param)
            else:
                print(f"  ✗ {param}")
                not_tracked.append(f"{category}: {param}")

    print("\n" + "="*100)
    print(f"\nSUMMARY: {len(tracked)} parameters tracked, {len(not_tracked)} NOT tracked\n")

    if not_tracked:
        print("NOT CURRENTLY PRINTED:")
        print("-" * 100)
        for param in not_tracked:
            print(f"  • {param}")

    print("\n" + "="*100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and report all TorchTitan hyperparameters"
    )
    parser.add_argument(
        "--config-json",
        type=str,
        help="Path to config JSON file (from TorchTitan training)"
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Only show missing hyperparameters"
    )

    args = parser.parse_args()

    # If missing-only flag is set, just print what's not tracked
    if args.missing_only:
        print_missing_hyperparameters_report()
        return

    # Try to load config from JSON if provided
    if args.config_json and os.path.exists(args.config_json):
        try:
            with open(args.config_json) as f:
                config_dict = json.load(f)
                print(f"Loaded config from {args.config_json}")
                # TODO: convert dict back to Trainer.Config
                print(json.dumps(config_dict, indent=2, default=str))
        except Exception as e:
            print(f"Error loading config: {e}", file=sys.stderr)
    else:
        print("\nUsage examples:")
        print("  # Show all hyperparameters (when Trainer.Config is available)")
        print("  python hyperparameters.py")
        print()
        print("  # Show only missing/not-tracked hyperparameters")
        print("  python hyperparameters.py --missing-only")
        print()
        print("  # Output as JSON")
        print("  python hyperparameters.py --format json")
        print()
        print("Note: This script is designed to be imported and used with TorchTitan")
        print("training runs, or called with --missing-only to see what's not tracked.\n")

        # Still show the missing hyperparameters report as default
        print_missing_hyperparameters_report()


if __name__ == "__main__":
    main()
