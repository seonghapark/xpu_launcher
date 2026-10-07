#!/usr/bin/env python3
"""
Standalone hyperparameter tracking status report.
Does NOT require TorchTitan environment - works standalone.

Shows:
- All hyperparameters that SHOULD be tracked (per ML best practices)
- Which ones ARE currently printed in the wrapper scripts
- Which ones are NOT currently printed
- Recommendations for adding missing ones
"""

def print_missing_hyperparameters_report():
    """Print what hyperparameters are NOT currently being tracked/printed."""

    all_requested = {
        "Optimization": [
            ("learning_rate", "In config_registry but not printed pre-training", "❌"),
            ("optimizer_type", "Hardcoded as AdamW, not shown", "❌"),
            ("momentum", "Not exposed in TorchTitan interface", "❌"),
            ("betas", "In default_adamw() but not in launch config", "❌"),
            ("weight_decay", "In optimizer config, not in launch config", "❌"),
            ("gradient_clipping_max_norm", "In training.max_norm, NOT printed", "❌"),
            ("optimizer_epsilon", "In default_adamw(), not printed", "❌"),
        ],
        "Training Schedule": [
            ("epochs", "Implicit via training.steps, no explicit epochs", "❌"),
            ("total_training_steps", "Printed as training.steps", "✓"),
            ("batch_size", "Printed as local_batch_size", "✓"),
            ("gradient_accumulation_steps", "Derived at runtime, not pre-computed", "⚠️"),
            ("warmup_steps", "In lr_scheduler config, printed during training", "✓"),
            ("lr_scheduler_type", "Fixed as linear decay, not printed", "⚠️"),
            ("decay_rate", "Printed as decay_ratio in lr_scheduler", "✓"),
        ],
        "Model Architecture": [
            ("num_layers", "Available in model.config, printed at training start", "✓"),
            ("hidden_size", "Available in model.config, printed at training start", "✓"),
            ("attention_heads", "Available in model.config, printed at training start", "✓"),
            ("intermediate_size", "Available but not pre-printed", "⚠️"),
            ("vocab_size", "Available in model.config, printed at training start", "✓"),
            ("max_seq_len", "Available in model.config, printed at training start", "✓"),
            ("activation_function", "Hardcoded in model, not exposed", "❌"),
            ("dropout_rate", "Hardcoded in model, not exposed", "❌"),
            ("normalization_type", "RMSNorm in Llama/AGPT, not printed", "❌"),
        ],
        "Initialization": [
            ("weight_initialization_method", "TorchTitan default, not configurable", "❌"),
            ("bias_initialization", "TorchTitan default, not configurable", "❌"),
            ("random_seed", "Not exposed in config, not printed", "❌"),
        ],
        "Regularization": [
            ("dropout", "Hardcoded in model, not printed", "❌"),
            ("weight_decay", "In optimizer, not pre-printed", "⚠️"),
            ("label_smoothing", "Not implemented in CrossEntropyLoss", "❌"),
            ("data_augmentation", "Dataset-specific, not in config", "❌"),
            ("early_stopping", "Implemented as loss_std_termination, partially printed", "✓"),
        ],
        "Loss": [
            ("loss_function", "Printed as loss type at training start", "✓"),
            ("loss_weights", "Single loss, not used", "❌"),
            ("class_weights", "Not implemented", "❌"),
            ("auxiliary_losses", "Not in base config", "❌"),
            ("loss_std_termination", "Printed in launch config", "✓"),
            ("loss_std_threshold", "Printed in launch config", "✓"),
            ("loss_std_window", "Printed in launch config", "✓"),
        ],
        "Data": [
            ("dataset_name", "Printed in MODEL & DATASET section", "✓"),
            ("dataset_path", "Printed in MODEL & DATASET section", "✓"),
            ("dataset_size", "Not tracked in config", "❌"),
            ("sampling_strategy", "Dataset-specific, not exposed", "❌"),
            ("shuffle", "Dataset-specific, not exposed", "❌"),
            ("train_val_split", "Hardcoded in dataset loaders (5% validation)", "❌"),
            ("preprocessing", "Dataset-specific, not exposed", "❌"),
            ("sequence_length", "Printed as seq_len in TRAINING HYPERPARAMETERS", "✓"),
            ("augmentation", "Dataset-specific, not exposed", "❌"),
        ],
        "Batching": [
            ("local_batch_size", "Printed in TRAINING HYPERPARAMETERS section", "✓"),
            ("global_batch_size", "Computed at runtime, not pre-printed", "⚠️"),
            ("micro_batch_size", "Same as local_batch_size", "✓"),
            ("padding_strategy", "Dataset-specific, not exposed", "❌"),
            ("dynamic_batching", "Not supported in TorchTitan", "❌"),
        ],
        "Precision and System": [
            ("training_dtype", "Printed as dtype in training config", "✓"),
            ("mixed_precision_param", "Printed during training", "✓"),
            ("mixed_precision_reduce", "Printed during training", "✓"),
            ("gradient_scaling", "Handled by torch.autocast, not explicit", "⚠️"),
            ("distributed_parallelism", "Printed in launch config (NNODES, NPROC)", "✓"),
        ],
        "Context Extension (RoPE)": [
            ("rope_backend_type", "Available in model but NOT printed", "❌"),
            ("context_extension_technique", "YaRN/Linear/None - NOT printed", "❌"),
            ("yarn_alpha", "Available if YaRN used, NOT printed", "❌"),
            ("yarn_beta", "Available if YaRN used, NOT printed", "❌"),
            ("rope_scaling_method", "Available but NOT printed", "❌"),
        ],
        "Model Provenance": [
            ("current_model_name", "Set via MODULE/CONFIG, not printed", "❌"),
            ("base_model_name", "Implicit (Llama3), not printed", "❌"),
            ("original_model_name", "Not tracked", "❌"),
            ("initialization_checkpoint", "Set via CKPT env var, not printed", "⚠️"),
        ],
    }

    print("\n" + "="*120)
    print("HYPERPARAMETER TRACKING STATUS REPORT")
    print("="*120 + "\n")
    print("Legend:")
    print("  ✓  = Currently printed in pre-training launch config")
    print("  ⚠️  = Available but not pre-printed (shown during training or derived at runtime)")
    print("  ❌ = NOT currently tracked/printed anywhere\n")

    total_params = 0
    tracked_count = 0
    available_count = 0
    missing_count = 0

    for category, params in all_requested.items():
        print(f"\n[{category.upper()}]")
        print("-" * 120)

        for param_name, reason, status in params:
            total_params += 1
            if status == "✓":
                tracked_count += 1
            elif status == "⚠️":
                available_count += 1
            else:
                missing_count += 1

            # Format output with consistent spacing
            print(f"  {status:^3}  {param_name:<35}  {reason:<75}")

    print("\n" + "="*120)
    print(f"\nSUMMARY:")
    print(f"  Total hyperparameters evaluated:        {total_params}")
    print(f"  Fully tracked & printed (✓):           {tracked_count:>3}  ({100*tracked_count//total_params}%)")
    print(f"  Available but not pre-printed (⚠️):    {available_count:>3}  ({100*available_count//total_params}%)")
    print(f"  Missing/Not tracked (❌):              {missing_count:>3}  ({100*missing_count//total_params}%)")
    print("\n" + "="*120)


def print_what_is_printed():
    """Show what IS currently being printed."""

    print("\n" + "="*120)
    print("WHAT IS CURRENTLY PRINTED (Pre-Training Launch Config)")
    print("="*120 + "\n")

    sections = {
        "[LAUNCH MODE]": [
            "NNODES (number of nodes)",
            "NPROC_PER_NODE (processes/devices per node)",
            "NPROC (total processes)",
            "Auto Retry (enabled/disabled)",
            "Spare Nodes (count for failover)",
        ],
        "[MODEL & DATASET]": [
            "Model Path",
            "Module/Config (e.g., agpt_yarn / agpt_2b_yarn)",
            "Dataset Name",
            "Dataset Path (if set)",
            "HF Assets Path",
        ],
        "[TRAINING HYPERPARAMETERS]": [
            "Training Steps",
            "Sequence Length (seq_len)",
            "Loss Std Termination (enabled/disabled)",
            "Loss Std Threshold",
            "Loss Std Window",
        ],
        "[SYSTEM & PATHS]": [
            "TorchTitan Root",
            "Log Directory",
            "Checkpoint Folder",
            "Python Binary",
        ],
        "[MONITORING]": [
            "Resource Monitor (enabled/disabled)",
            "Resource Interval (seconds)",
            "Resource Output Directory",
        ],
        "[EXTRA ARGUMENTS]": [
            "Any command-line arguments after --",
        ],
    }

    for section, items in sections.items():
        print(f"\n{section}")
        print("-" * 120)
        for item in items:
            print(f"  • {item}")

    print("\n" + "="*120)
    print("DURING TRAINING (Printed by TorchTitan at startup)")
    print("="*120 + "\n")

    during_training = {
        "Optimizer Settings": [
            "learning_rate (lr)",
            "optimizer type (AdamW)",
            "betas (momentum parameters)",
            "epsilon (eps)",
            "weight_decay",
        ],
        "Training Settings": [
            "local_batch_size",
            "global_batch_size (computed)",
            "gradient_accumulation_steps (computed)",
            "max_norm (gradient clipping)",
        ],
        "Model Architecture": [
            "num_layers",
            "hidden_size (dim)",
            "num_attention_heads",
            "vocab_size",
            "intermediate_size",
        ],
        "Parallelism": [
            "data_parallel_replicate_degree",
            "data_parallel_shard_degree",
            "tensor_parallel_degree",
            "pipeline_parallel_degree",
            "sequence_parallel (enabled)",
        ],
        "Precision": [
            "training.dtype (float32, bfloat16)",
            "mixed_precision_param",
            "mixed_precision_reduce",
        ],
        "Checkpoint & Compile": [
            "checkpoint.enable",
            "checkpoint.interval",
            "compile.enable",
        ],
    }

    for section, items in during_training.items():
        print(f"\n{section}")
        print("-" * 120)
        for item in items:
            print(f"  • {item}")

    print("\n" + "="*120 + "\n")


def print_recommendations():
    """Print recommendations for improvement."""

    print("\n" + "="*120)
    print("RECOMMENDATIONS FOR IMPROVEMENT")
    print("="*120 + "\n")

    recommendations = [
        ("Add RoPE info to launch config",
         "Print rope backend and YaRN parameters if detected\n"
         "        Add to run_train_torchtitan.sh:\n"
         "        [CONTEXT EXTENSION]\n"
         "          RoPE Backend       = <detected or default>\n"
         "          Context Technique  = YaRN|Linear|None\n"
         "          YaRN Parameters    = alpha=..., beta=...",
         "Medium priority"),

        ("Add model provenance tracking",
         "Track original vs current model name\n"
         "        Add to run_train_torchtitan.sh:\n"
         "        [MODEL PROVENANCE]\n"
         "          Current Model       = {MODULE}_{CONFIG}\n"
         "          Base Model          = Llama3 (for agpt)\n"
         "          Original Model      = AGPT-2B (from CKPT)\n"
         "          Init Checkpoint     = {CKPT or 'random'}",
         "Medium priority"),

        ("Add missing optimizer params to launch config",
         "Show lr, optimizer type, betas, weight_decay before training starts\n"
         "        Add env var handling in run_train_torchtitan.sh:\n"
         "        OPTIMIZER_LR, OPTIMIZER_WEIGHT_DECAY, etc.",
         "Low priority (shown during training anyway)"),

        ("Use hyperparameters.py script",
         "Created script can extract and report all available hyperparameters\n"
         "        Usage: python hyperparameters.py --missing-only\n"
         "        Can be integrated into training pipeline for logging",
         "High priority"),

        ("Add random seed tracking",
         "Include seed in config and log it\n"
         "        Add to Trainer.Config or print from environment",
         "High priority"),

        ("Track data augmentation",
         "Document any preprocessing/augmentation done by dataloader\n"
         "        Add to dataset config or launch output",
         "Medium priority"),
    ]

    for i, (title, detail, priority) in enumerate(recommendations, 1):
        print(f"{i}. {title} [{priority}]")
        print(f"   {detail}\n")

    print("="*120 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Hyperparameter tracking status report"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Show full report (default shows only missing)"
    )
    parser.add_argument(
        "--recommendations", action="store_true",
        help="Show recommendations only"
    )

    args = parser.parse_args()

    if args.recommendations:
        print_recommendations()
    elif args.full:
        print_what_is_printed()
        print_missing_hyperparameters_report()
        print_recommendations()
    else:
        # Default: show missing hyperparameters
        print_missing_hyperparameters_report()


if __name__ == "__main__":
    main()
