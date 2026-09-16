# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from dataclasses import is_dataclass
from typing import Any, Literal

import ezpz
import ezpz.distributed

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, OptimizersContainer
from torchtitan.components.quantization.float8 import (
    Float8GroupedExpertsConverter,
    Float8LinearConverter,
)
from torchtitan.config import (
    CommConfig,
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments.ezpz.blendcorpus.blendcorpus_builder import (
    BlendCorpusDataLoader,
)
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.experiments.ezpz.moe.activation_checkpoint import MoeSelectiveAC
from torchtitan.experiments.ezpz.blendcorpus.build_tokenizer import EZPZTokenizer
from torchtitan.experiments.ezpz.trainer import FaultTolerantTrainer
from torchtitan.experiments.torchft.config.job_config import FaultTolerance

from . import model_registry

TT_CONFIG_JSON_ENV = "TT_CONFIG_JSON"


def _load_json_overrides() -> dict[str, Any]:
    path = os.environ.get(TT_CONFIG_JSON_ENV, "").strip()
    if not path:
        raise ValueError(
            f"{TT_CONFIG_JSON_ENV} must point to a JSON file when using *_from_json configs."
        )

    with open(path, encoding="utf-8") as f:
        overrides = json.load(f)

    if not isinstance(overrides, dict):
        raise ValueError(
            f"Expected top-level JSON object in {path!r}, got {type(overrides).__name__}."
        )

    return overrides


def _apply_config_overrides(
    target: Any,
    overrides: dict[str, Any],
    path: str = "",
) -> None:
    for key, value in overrides.items():
        if not hasattr(target, key):
            raise KeyError(f"Unknown config field {key!r} at path {path or '<root>'}.")

        current_value = getattr(target, key)
        field_path = f"{path}.{key}" if path else key

        if isinstance(value, dict):
            if not is_dataclass(current_value):
                raise TypeError(
                    f"Expected dataclass at {field_path!r} for nested override, "
                    f"got {type(current_value).__name__}."
                )
            _apply_config_overrides(current_value, value, field_path)
            continue

        setattr(target, key, value)


def _config_from_json(base_fn) -> FaultTolerantTrainer.Config:
    cfg = base_fn()
    _apply_config_overrides(cfg, _load_json_overrides())
    return cfg


def _base_config(flavor: str) -> FaultTolerantTrainer.Config:
    return FaultTolerantTrainer.Config(
        hf_assets_path="./assets/hf/gemma-7b",
        model_spec=model_registry(flavor),
        tokenizer=EZPZTokenizer.Config(backend="hf"),
        loss=CrossEntropyLoss.Config(),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=200,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=8192,
            steps=10000,
        ),
        dataloader=BlendCorpusDataLoader.Config(dataset="c4_test"),
        metrics=MetricsProcessor.Config(log_freq=10),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        # MoeSelectiveAC drops _c10d_functional.all_to_all_single from the
        # save list -- needed for PR14's padded a2a fast-path. See
        # ezpz/moe/activation_checkpoint.py for rationale.
        activation_checkpoint=MoeSelectiveAC.Config(),
        comm=CommConfig(train_timeout_seconds=100),
        fault_tolerance=FaultTolerance(enable=False),
    )


def moe(
    flavor: str,
    local_batch_size: int = 1,
    activation_checkpoint_mode: Literal["none", "full", "selective"] = "full",
    seq_len: int = 8192,
    # See agpt/config_registry.py for the full reasoning. bf16 master
    # weights silently freeze RMSNorm.weight at init=1.0 because the
    # per-step update (~lr * exp_avg / sqrt(hessian)) is sub-ulp at
    # bf16 scale 1.0.
    dtype: Literal["bfloat16", "float32"] = "float32",
    compile: bool = True,
    checkpoint_interval: int = 50,
    hf_assets_path: str = "./assets/hf/gemma-7b",
    dataset_path: str | None = None,
) -> FaultTolerantTrainer.Config:
    cfg = _base_config(flavor)
    cfg.hf_assets_path = hf_assets_path
    cfg.debug.print_config = True
    cfg.training.local_batch_size = local_batch_size
    # 57th sync: PR #3674 replaced the `mode` string with a policy class
    # hierarchy. Map the knob: none -> None (AC off); full -> FullAC;
    # selective -> MoeSelectiveAC (SelectiveAC minus the all_to_all_single
    # save, needed for PR14's padded a2a fast-path).
    if activation_checkpoint_mode == "none":
        cfg.activation_checkpoint = None
    elif activation_checkpoint_mode == "full":
        cfg.activation_checkpoint = FullAC.Config()
    else:
        cfg.activation_checkpoint = MoeSelectiveAC.Config()
    cfg.training.seq_len = seq_len
    cfg.training.dtype = dtype
    cfg.dataloader.dataset = "blendcorpus"
    if dataset_path is None:
        dataset_path = f"torchtitan/experiments/ezpz/data-lists/{ezpz.distributed.get_machine().lower()}/books.txt"
    cfg.dataloader.dataset_path = dataset_path
    cfg.metrics.log_freq = 1
    cfg.metrics.enable_wandb = True
    if compile:
        cfg.compile = CompileConfig(enable=True)
    cfg.checkpoint.enable = True
    cfg.checkpoint.interval = checkpoint_interval
    return cfg


def moe_500m() -> FaultTolerantTrainer.Config:
    return moe("500M", local_batch_size=4)


def moe_2b() -> FaultTolerantTrainer.Config:
    return moe("2B", local_batch_size=16)


def moe_4b() -> FaultTolerantTrainer.Config:
    return moe("4B", local_batch_size=16)


def moe_7b() -> FaultTolerantTrainer.Config:
    return moe("7B", local_batch_size=2, activation_checkpoint_mode="none")


def moe_debugmodel() -> FaultTolerantTrainer.Config:
    return moe("debugmodel", local_batch_size=8)


def moe_debugmodel_hf() -> FaultTolerantTrainer.Config:
    cfg = moe("debugmodel", local_batch_size=8)
    cfg.dataloader.dataset_path = None
    return cfg


def moe_debugmodel_ep() -> FaultTolerantTrainer.Config:
    """EP=2 variant of moe_debugmodel.

    LBS pinned to 2 (the EP=1 base uses LBS=8): at LBS=8 with EP=2 on
    24 XPU ranks (2N Sunspot), the bf16 vocab-projection logits
    ``(LBS * seq_len, vocab_size) = (8 * 8192, 256128) * 2 B`` request
    ~33 GiB on a single tile and OOM at init. Validated 2026-05-20
    (see docs/experiments/moe/sunspot/20260520-smoke-n2-pr3386-ep-followup.md).
    """
    cfg = moe("debugmodel", local_batch_size=2)
    cfg.model_spec = model_registry("debugmodel", moe_comm_backend="standard")
    cfg.parallelism.expert_parallel_degree = 2
    return cfg


def moe_debugmodel_flex_attn() -> FaultTolerantTrainer.Config:
    cfg = moe_debugmodel()
    cfg.model_spec = model_registry("debugmodel_flex_attn")
    return cfg


def moe_debugmodel_flex_attn_hf() -> FaultTolerantTrainer.Config:
    cfg = moe_debugmodel_hf()
    cfg.model_spec = model_registry("debugmodel_flex_attn_hf")
    return cfg


def moe_small() -> FaultTolerantTrainer.Config:
    return moe("small", local_batch_size=8)


def moe_small_hf() -> FaultTolerantTrainer.Config:
    cfg = moe("small", local_batch_size=8)
    cfg.dataloader.dataset_path = None
    return cfg


def moe_16b() -> FaultTolerantTrainer.Config:
    cfg = moe(
        "16B",
        local_batch_size=4,
        hf_assets_path="./assets/hf/deepseek-moe-16b-base",
    )
    cfg.optimizer.param_groups[0].optimizer_kwargs["lr"] = 2.2e-4
    cfg.lr_scheduler.decay_type = "cosine"
    cfg.lr_scheduler.min_lr_factor = 0.1
    cfg.lr_scheduler.warmup_steps = 200
    cfg.training.steps = 1000
    cfg.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    cfg.parallelism.expert_parallel_degree = 8
    cfg.compile = CompileConfig(enable=True, components=["loss"])
    return cfg


def moe_671b() -> FaultTolerantTrainer.Config:
    cfg = moe(
        "671B",
        local_batch_size=4,
        hf_assets_path="./assets/hf/DeepSeek-V3.1-Base",
    )
    cfg.optimizer.param_groups[0].optimizer_kwargs["lr"] = 2.2e-4
    cfg.lr_scheduler.warmup_steps = 2000
    cfg.lr_scheduler.decay_type = "cosine"
    cfg.lr_scheduler.min_lr_factor = 0.1
    cfg.training.steps = 10000
    cfg.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    cfg.checkpoint.interval = 500
    cfg.compile = CompileConfig(enable=True, components=["loss"])
    # Quantization is now applied to the config at model_registry time
    # rather than to the runtime model (#3127). Re-register with Float8.
    cfg.model_spec = model_registry(
        "671B",
        quantization=[
            Float8LinearConverter.Config(filter_fqns=["output", "router.gate"]),
            Float8GroupedExpertsConverter.Config(),
        ],
    )
    return cfg


def moe_7b_ep() -> FaultTolerantTrainer.Config:
    cfg = moe_7b()
    cfg.model_spec = model_registry("7B", moe_comm_backend="standard")
    cfg.parallelism.expert_parallel_degree = 2
    return cfg


def moe_10b_2b_sdpa_ep() -> FaultTolerantTrainer.Config:
    cfg = moe_10b_2b_sdpa()
    cfg.model_spec = model_registry("10B_2B_sdpa", moe_comm_backend="standard")
    cfg.parallelism.expert_parallel_degree = 2
    return cfg


def moe_2b_ep() -> FaultTolerantTrainer.Config:
    """EP=2 variant of moe_2b.

    LBS pinned to 2 (the EP=1 ``moe_2b`` base uses LBS=16). With the
    full Gemma vocab (256128) at seq_len=8192, the bf16 lm_head
    logits ``(LBS * seq_len, vocab_size) * 2 B`` request
    ``LBS * ~3.9 GiB`` on a single tile *before* any expert/dispatch
    memory. At LBS=16 that's ~62.5 GiB — overflows a Max 1550 tile's
    64 GiB budget at first forward. Measured peaks on 2N Sunspot:
    LBS=1 → 14.95 GiB (24%), LBS=2 → 27.08 GiB (42%), LBS=16 → OOM.
    LBS=2 leaves headroom for activations + dispatch buffers while
    keeping the global batch reasonable.
    """
    cfg = moe("2B", local_batch_size=2)
    cfg.model_spec = model_registry("2B", moe_comm_backend="standard")
    cfg.parallelism.expert_parallel_degree = 2
    return cfg


def moe_10b_2b() -> FaultTolerantTrainer.Config:
    cfg = moe("10B_2B", local_batch_size=1)
    cfg.optimizer.param_groups[0].optimizer_kwargs["lr"] = 2.2e-4
    cfg.lr_scheduler.decay_type = "cosine"
    cfg.lr_scheduler.min_lr_factor = 0.1
    cfg.training.steps = 1000
    cfg.checkpoint.interval = 100
    return cfg


def moe_10b_2b_sdpa() -> FaultTolerantTrainer.Config:
    # LBS=1 + AC=selective is the empirically-working combo on 2N
    # Sunspot. The prior (LBS=2, AC="none") default OOMs at first
    # forward; flipping to AC="full" hits PyTorch's
    # ``CheckpointError: Recomputed values have different metadata``
    # because MoE token-routing isn't bit-exact under recompute (the
    # router selects one fewer/more token in a few experts → saved
    # shape (N, hidden) vs recomputed (N±1, hidden)). AC="selective"
    # only checkpoints the SAC save-list — which excludes the router
    # — so the non-deterministic op never gets recomputed and the
    # shape stays stable. PR #3146/#3450 made the routing's ``histc``
    # → ``bincount`` swap deterministic in the *forward* path, but
    # AC=full still recomputes a different routing each pass; the
    # shape divergence is fundamental until AC saves the routing
    # result instead of recomputing it.
    cfg = moe("10B_2B_sdpa", local_batch_size=1,
              activation_checkpoint_mode="selective")
    cfg.optimizer.param_groups[0].optimizer_kwargs["lr"] = 2.2e-4
    cfg.lr_scheduler.decay_type = "cosine"
    cfg.lr_scheduler.min_lr_factor = 0.1
    cfg.training.steps = 1000
    cfg.checkpoint.interval = 100
    return cfg


def smoke_moe_500m_50steps() -> FaultTolerantTrainer.Config:
    """50-step moe smoke test for the post-#2963/#2937 replay.

    Smallest non-debug moe flavor (500M) on 2 nodes, AdamW, no checkpoint.
    Verifies imports, model build, sharding-config population on MLA
    attention + dense FFN + MoE submodules, Module.parallelize,
    per-block compile, FSDP wrap, optimizer step, loss decreasing.
    Uses fineweb-edu HF stream so no local data is required.
    """
    cfg = moe(
        "500M",
        local_batch_size=2,
        activation_checkpoint_mode="none",
        seq_len=8192,
        compile=True,
        checkpoint_interval=10_000,
    )
    cfg.dataloader.dataset = "HuggingFaceFW/fineweb-edu"
    cfg.dataloader.dataset_path = None
    cfg.training.steps = 50
    cfg.checkpoint.enable = False
    cfg.optimizer.param_groups[0].optimizer_kwargs["lr"] = 8e-4
    cfg.lr_scheduler.warmup_steps = 5
    cfg.lr_scheduler.decay_ratio = 0.0
    cfg.metrics.log_freq = 1
    return cfg


def moe_debugmodel_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json(moe_debugmodel)


def moe_small_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json(moe_small)


def moe_16b_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json(moe_16b)


def moe_10b_2b_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json(moe_10b_2b)


def moe_671b_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json(moe_671b)
