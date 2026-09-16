# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Speedrun configs for the agpt_2b loss competition.
#
# Goal: lowest loss in 1000 steps on 2 nodes (24 XPU tiles).
# Fixed: dataset=HuggingFaceFW/fineweb-edu, LBS=2, seq_len=8192.
#
# Usage:
#   ezpz launch python3 -m torchtitan.experiments.ezpz.train \
#       --module ezpz.agpt --config speedrun_2b_muon

import torchtitan.experiments.ezpz.datasets  # noqa: F401 — register HF datasets

from torchtitan.experiments.ezpz.agpt.config_registry import agpt
from torchtitan.experiments.ezpz.optimizer import (
    default_mano,
    default_muon,
    default_schedule_free,
    default_sophiag,
    default_spam,
    default_torch_muon,
    ManoOptimizersContainer,
    MuonOptimizersContainer,
    ScheduleFreeOptimizersContainer,
    SPAMOptimizersContainer,
    SophiaGOptimizersContainer,
    TorchMuonOptimizersContainer,
)

# Fixed competition parameters — same for all configs
DATASET = "HuggingFaceFW/fineweb-edu"
LOCAL_BATCH_SIZE = 2
SEQ_LEN = 8192
STEPS = 1000


def _speedrun_base():
    """Base speedrun config: 1000 steps, WSD schedule, fineweb-edu streaming.

    Fixed parameters (not tunable for fairness):
    - Dataset: HuggingFaceFW/fineweb-edu (streaming)
    - LBS: 2 (GBS=48 on 24 tiles)
    - seq_len: 8192
    - Steps: 1000

    Tunable: optimizer, LR, LR schedule, gradient clipping, etc.
    """
    cfg = agpt(
        "2b",
        local_batch_size=LOCAL_BATCH_SIZE,
        activation_checkpoint_mode="none",
        seq_len=SEQ_LEN,
        compile=True,
        checkpoint_interval=STEPS,
    )

    # Dataset: stream from HF (no local data needed)
    cfg.dataloader.dataset = DATASET
    cfg.dataloader.dataset_path = None

    # Training: 1000 steps
    cfg.training.steps = STEPS

    # No checkpointing for speedruns
    cfg.checkpoint.enable = False

    # LR schedule: WSD (warmup 20, stable to 800, decay last 200)
    cfg.lr_scheduler.warmup_steps = 20
    cfg.lr_scheduler.decay_ratio = 0.2
    cfg.lr_scheduler.decay_type = "linear"
    cfg.lr_scheduler.min_lr_factor = 0.0

    return cfg


# ---- Competition configs ----


def speedrun_2b_adamw():
    """AdamW baseline — LR from LR finder (1.3e-3)."""
    cfg = _speedrun_base()
    cfg.optimizer.param_groups[0].optimizer_kwargs["lr"] = 1.3e-3
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_adamw"
    return cfg


def speedrun_2b_muon():
    """Muon — best NanoGPT speedrun optimizer. LR from LR finder (2.4e-3)."""
    cfg = _speedrun_base()
    cfg.optimizer = default_muon(lr=2.4e-3)
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_muon"
    return cfg


def speedrun_2b_sophiag():
    """SophiaG — second-order Hessian approx. LR from LR finder (3.1e-4)."""
    cfg = _speedrun_base()
    cfg.optimizer = default_sophiag(lr=3.1e-4)
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_sophiag"
    return cfg


def speedrun_2b_muon_aggressive():
    """Muon with 2x LR — pushing convergence speed."""
    cfg = _speedrun_base()
    cfg.optimizer = default_muon(lr=4.8e-3)
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_muon_aggressive"
    return cfg


def speedrun_2b_adamw_high_lr():
    """AdamW with 2x LR — testing upper bound."""
    cfg = _speedrun_base()
    cfg.optimizer.param_groups[0].optimizer_kwargs["lr"] = 2.6e-3
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_adamw_high_lr"
    return cfg


# ---- Hparam tweak configs ----


def speedrun_2b_adamw_short_decay():
    """AdamW with 10% decay (vs 20%) — more time at peak LR."""
    cfg = _speedrun_base()
    cfg.optimizer.param_groups[0].optimizer_kwargs["lr"] = 1.3e-3
    cfg.lr_scheduler.decay_ratio = 0.1
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_adamw_short_decay"
    return cfg


def speedrun_2b_adamw_cosine():
    """AdamW with cosine decay instead of linear."""
    cfg = _speedrun_base()
    cfg.optimizer.param_groups[0].optimizer_kwargs["lr"] = 1.3e-3
    cfg.lr_scheduler.decay_type = "cosine"
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_adamw_cosine"
    return cfg


def speedrun_2b_adamw_fast_warmup():
    """AdamW with 5-step warmup + 10% decay — max time at peak LR."""
    cfg = _speedrun_base()
    cfg.optimizer.param_groups[0].optimizer_kwargs["lr"] = 1.3e-3
    cfg.lr_scheduler.warmup_steps = 5
    cfg.lr_scheduler.decay_ratio = 0.1
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_adamw_fast_warmup"
    return cfg


def speedrun_2b_muon_short_decay():
    """Muon with 10% decay — more time at peak LR."""
    cfg = _speedrun_base()
    cfg.optimizer = default_muon(lr=2.4e-3)
    cfg.lr_scheduler.decay_ratio = 0.1
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_muon_short_decay"
    return cfg


def speedrun_2b_muon_fast_warmup():
    """Muon with 5-step warmup + 10% decay."""
    cfg = _speedrun_base()
    cfg.optimizer = default_muon(lr=2.4e-3)
    cfg.lr_scheduler.warmup_steps = 5
    cfg.lr_scheduler.decay_ratio = 0.1
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_muon_fast_warmup"
    return cfg


# ---- New optimizer configs ----


def speedrun_2b_mano():
    """Mano — manifold-normalized optimizer, 1.75x faster than Muon."""
    cfg = _speedrun_base()
    cfg.optimizer = default_mano(lr=3.0e-4)
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_mano"
    return cfg


def speedrun_2b_spam():
    """SPAM — spike-aware Adam with momentum reset."""
    cfg = _speedrun_base()
    cfg.optimizer = default_spam(lr=1.3e-3)
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_spam"
    return cfg


# ---- Architecture tweak configs ----


def _speedrun_qknorm_base():
    """Base config with QK-Norm enabled."""
    cfg = agpt(
        "2b_qknorm",
        local_batch_size=LOCAL_BATCH_SIZE,
        activation_checkpoint_mode="none",
        seq_len=SEQ_LEN,
        compile=True,
        checkpoint_interval=STEPS,
    )
    cfg.dataloader.dataset = DATASET
    cfg.dataloader.dataset_path = None
    cfg.training.steps = STEPS
    cfg.lr_scheduler.warmup_steps = 20
    cfg.lr_scheduler.decay_ratio = 0.2
    cfg.lr_scheduler.decay_type = "linear"
    cfg.lr_scheduler.min_lr_factor = 0.0
    return cfg


def speedrun_2b_adamw_qknorm():
    """AdamW + QK-Norm — stabilizes early attention training."""
    cfg = _speedrun_qknorm_base()
    cfg.optimizer.param_groups[0].optimizer_kwargs["lr"] = 1.3e-3
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_adamw_qknorm"
    return cfg


def speedrun_2b_muon_qknorm():
    """Muon + QK-Norm — best optimizer + attention stabilization."""
    cfg = _speedrun_qknorm_base()
    cfg.optimizer = default_muon(lr=2.4e-3)
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_muon_qknorm"
    return cfg


# ---- Round 2: combo configs based on round 1 findings ----


def speedrun_2b_mano_high_lr():
    """Mano with higher LR (6e-4) — try to close gap to Muon."""
    cfg = _speedrun_base()
    cfg.optimizer = default_mano(lr=6.0e-4)
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_mano_high_lr"
    return cfg


def speedrun_2b_mano_1e3():
    """Mano with LR=1e-3 — aggressive push."""
    cfg = _speedrun_base()
    cfg.optimizer = default_mano(lr=1.0e-3)
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_mano_1e3"
    return cfg


def speedrun_2b_muon_cosine():
    """Muon + cosine decay — combine best optimizer with best schedule."""
    cfg = _speedrun_base()
    cfg.optimizer = default_muon(lr=2.4e-3)
    cfg.lr_scheduler.decay_type = "cosine"
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_muon_cosine"
    return cfg


def speedrun_2b_mano_cosine():
    """Mano + cosine decay — fast optimizer with best schedule."""
    cfg = _speedrun_base()
    cfg.optimizer = default_mano(lr=3.0e-4)
    cfg.lr_scheduler.decay_type = "cosine"
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_mano_cosine"
    return cfg


def speedrun_2b_mano_qknorm():
    """Mano + QK-Norm — fast manifold optimizer with attention stabilization."""
    cfg = _speedrun_qknorm_base()
    cfg.optimizer = default_mano(lr=3.0e-4)
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_mano_qknorm"
    return cfg


# ---- torch.optim.Muon (built-in, optimized) ----


def speedrun_2b_torchmuon():
    """torch.optim.Muon — official PyTorch implementation, much faster per-step."""
    cfg = _speedrun_base()
    cfg.optimizer = default_torch_muon(lr=2.4e-3)
    return cfg


def speedrun_2b_torchmuon_cosine():
    """torch.optim.Muon + cosine decay."""
    cfg = _speedrun_base()
    cfg.optimizer = default_torch_muon(lr=2.4e-3)
    cfg.lr_scheduler.decay_type = "cosine"
    return cfg


# ---- Architecture tweak speedruns (round 3) ----


def _speedrun_variant_base(variant: str):
    """Base speedrun using a model variant (softcap, relu2, kitchen_sink)."""
    cfg = agpt(
        variant,
        local_batch_size=LOCAL_BATCH_SIZE,
        activation_checkpoint_mode="none",
        seq_len=SEQ_LEN,
        compile=True,
        checkpoint_interval=STEPS,
    )
    cfg.dataloader.dataset = DATASET_LOCAL
    cfg.dataloader.dataset_path = None
    cfg.training.steps = STEPS
    cfg.checkpoint.enable = False
    cfg.lr_scheduler.warmup_steps = 20
    cfg.lr_scheduler.decay_ratio = 0.2
    cfg.lr_scheduler.decay_type = "cosine"
    cfg.lr_scheduler.min_lr_factor = 0.0
    return cfg


def speedrun_2b_softcap():
    """AdamW + logit softcapping at 30.0 (Gemma 2 style)."""
    cfg = _speedrun_variant_base("2b_softcap")
    cfg.optimizer.param_groups[0].optimizer_kwargs["lr"] = 1.3e-3
    return cfg


def speedrun_2b_relu2():
    """AdamW + ReLU-squared activation in FFN (NanoGPT speedrun)."""
    cfg = _speedrun_variant_base("2b_relu2")
    cfg.optimizer.param_groups[0].optimizer_kwargs["lr"] = 1.3e-3
    return cfg


def speedrun_2b_kitchen_sink():
    """QK-Norm + logit softcap + ReLU² — everything combined."""
    cfg = _speedrun_variant_base("2b_kitchen_sink")
    cfg.optimizer.param_groups[0].optimizer_kwargs["lr"] = 1.3e-3
    return cfg


def speedrun_2b_mano_kitchen_sink():
    """Mano + QK-Norm + logit softcap + ReLU² — best optimizer + all tweaks."""
    cfg = _speedrun_variant_base("2b_kitchen_sink")
    cfg.optimizer = default_mano(lr=3.0e-4)
    return cfg


# ---- Full training configs (10B tokens, 8 nodes, local dataset) ----
#
# 8 nodes = 96 tiles, LBS=2, GAS=2 → GBS=384
# 10B tokens / (384 * 8192) = ~3,180 steps
# ~4 hours at AdamW/Mano speed (~7,200 TPS/GPU)

DATASET_LOCAL = "fineweb_edu_local"
TOKENS_10B = 10_000_000_000
TILES_8N = 96
GAS = 2


def _full_train_base():
    """Base config for 10B token training on 8 nodes.

    Uses locally cached FineWeb-Edu for reproducibility.
    GBS = 96 tiles × LBS=2 × GAS=2 = 384.
    ~3,180 steps, ~4 hours at AdamW speed.
    """
    gbs = TILES_8N * LOCAL_BATCH_SIZE * GAS
    tokens_per_step = gbs * SEQ_LEN
    steps = TOKENS_10B // tokens_per_step

    cfg = agpt(
        "2b",
        local_batch_size=LOCAL_BATCH_SIZE,
        activation_checkpoint_mode="none",
        seq_len=SEQ_LEN,
        compile=True,
        checkpoint_interval=500,
    )

    cfg.dataloader.dataset = DATASET_LOCAL
    cfg.dataloader.dataset_path = None
    cfg.training.steps = steps
    cfg.training.global_batch_size = gbs

    # WSD: warmup 2%, stable, cosine decay last 20%
    warmup = max(steps // 50, 10)
    cfg.lr_scheduler.warmup_steps = warmup
    cfg.lr_scheduler.decay_ratio = 0.2
    cfg.lr_scheduler.decay_type = "cosine"
    cfg.lr_scheduler.min_lr_factor = 0.0

    cfg.checkpoint.enable = True

    return cfg


def full_2b_adamw():
    """AdamW baseline, 10B tokens, 8 nodes."""
    cfg = _full_train_base()
    cfg.optimizer.param_groups[0].optimizer_kwargs["lr"] = 1.3e-3
    cfg.checkpoint.folder = "checkpoints/full_2b_adamw"
    return cfg


def full_2b_adamw_qknorm():
    """AdamW + QK-Norm — wall-clock champion, 10B tokens, 8 nodes."""
    cfg = _full_train_base()
    cfg.model_spec = agpt("2b_qknorm").model_spec
    cfg.optimizer.param_groups[0].optimizer_kwargs["lr"] = 1.3e-3
    cfg.checkpoint.folder = "checkpoints/full_2b_adamw_qknorm"
    return cfg


def full_2b_muon():
    """Muon — best loss optimizer, 10B tokens, 8 nodes."""
    cfg = _full_train_base()
    cfg.optimizer = default_muon(lr=2.4e-3)
    cfg.checkpoint.folder = "checkpoints/full_2b_muon"
    return cfg


def full_2b_mano():
    """Mano — fast manifold optimizer, 10B tokens, 8 nodes."""
    cfg = _full_train_base()
    cfg.optimizer = default_mano(lr=3.0e-4)
    cfg.checkpoint.folder = "checkpoints/full_2b_mano"
    return cfg


def full_2b_mano_qknorm():
    """Mano + QK-Norm — best combo, 10B tokens, 8 nodes."""
    cfg = _full_train_base()
    cfg.model_spec = agpt("2b_qknorm").model_spec
    cfg.optimizer = default_mano(lr=3.0e-4)
    cfg.checkpoint.folder = "checkpoints/full_2b_mano_qknorm"
    return cfg


# ---- Round 4: 2-node speedrun with GAS=8, local dataset ----
#
# 2 nodes = 24 tiles, LBS=2, GAS=8 → GBS=384
# 1000 steps × 384 × 8192 = 3.15B tokens
# ~5 hours at AdamW speed (~7,200 TPS/GPU)
# Uses local FineWeb-Edu (reproducible, no HF rate limits)

TILES_2N = 24
GAS_R4 = 8


def _r4_base(variant: str = "2b"):
    """Round 4 base: 2 nodes, GAS=8, local dataset, cosine WSD, no checkpoints."""
    gbs = TILES_2N * LOCAL_BATCH_SIZE * GAS_R4

    cfg = agpt(
        variant,
        local_batch_size=LOCAL_BATCH_SIZE,
        activation_checkpoint_mode="none",
        seq_len=SEQ_LEN,
        compile=True,
        checkpoint_interval=STEPS,
    )

    cfg.dataloader.dataset = DATASET_LOCAL
    cfg.dataloader.dataset_path = None
    cfg.training.steps = STEPS
    cfg.training.global_batch_size = gbs
    cfg.checkpoint.enable = False

    # Cosine WSD (best schedule from earlier rounds)
    cfg.lr_scheduler.warmup_steps = 20
    cfg.lr_scheduler.decay_ratio = 0.2
    cfg.lr_scheduler.decay_type = "cosine"
    cfg.lr_scheduler.min_lr_factor = 0.0

    return cfg


def r4_adamw():
    """AdamW baseline — GAS=8, local dataset."""
    cfg = _r4_base()
    cfg.optimizer.param_groups[0].optimizer_kwargs["lr"] = 1.3e-3
    return cfg


def r4_adamw_qknorm():
    """AdamW + QK-Norm — best wall-clock config from speedruns."""
    cfg = _r4_base("2b_qknorm")
    cfg.optimizer.param_groups[0].optimizer_kwargs["lr"] = 1.3e-3
    return cfg


def r4_mano():
    """Mano — fast manifold optimizer."""
    cfg = _r4_base()
    cfg.optimizer = default_mano(lr=3.0e-4)
    return cfg


def r4_mano_qknorm():
    """Mano + QK-Norm."""
    cfg = _r4_base("2b_qknorm")
    cfg.optimizer = default_mano(lr=3.0e-4)
    return cfg


def r4_adamw_softcap():
    """AdamW + logit softcapping (FlexAttention). Slow but tests convergence."""
    cfg = _r4_base("2b_softcap")
    cfg.optimizer.param_groups[0].optimizer_kwargs["lr"] = 1.3e-3
    return cfg


def r4_adamw_qknorm_softcap():
    """AdamW + QK-Norm + softcap — best speedrun tweak + softcap."""
    cfg = _r4_base("2b_kitchen_sink")
    # kitchen_sink has QK-Norm + softcap + ReLU² — but ReLU² hurt,
    # so let's use a new variant without it
    # For now just use kitchen_sink since we don't have a qknorm+softcap-only variant
    cfg.optimizer.param_groups[0].optimizer_kwargs["lr"] = 1.3e-3
    return cfg


def r4_adamw_higher_lr():
    """AdamW with sqrt-scaled LR for GBS=384 (vs GBS=48 baseline).

    Linear scaling rule: LR_new = LR_base * sqrt(GBS_new / GBS_base)
    = 1.3e-3 * sqrt(384/48) = 1.3e-3 * 2.83 = 3.7e-3
    """
    cfg = _r4_base()
    cfg.optimizer.param_groups[0].optimizer_kwargs["lr"] = 3.7e-3
    return cfg


def r4_mano_higher_lr():
    """Mano with sqrt-scaled LR for GBS=384.

    LR_new = 3.0e-4 * sqrt(384/48) = 8.5e-4
    """
    cfg = _r4_base()
    cfg.optimizer = default_mano(lr=8.5e-4)
    return cfg


# ---- Round 5: 8-node experiments (Mano LR sweep, WSM, Schedule-Free) ----
#
# 8 nodes = 96 tiles, LBS=2, GAS=2 → GBS=384
# 10B tokens / (384 * 8192) = ~3,180 steps
# ~4 hours at AdamW/Mano speed


def _r5_base():
    """Round 5 base: 8 nodes, GAS=2, local dataset, cosine WSD."""
    return _full_train_base()


# ---- Mano LR sweep ----


def r5_mano_lr3e4():
    """Mano at 3e-4 (baseline from LR finder)."""
    cfg = _r5_base()
    cfg.optimizer = default_mano(lr=3.0e-4)
    cfg.checkpoint.folder = "checkpoints/r5_mano_lr3e4"
    return cfg


def r5_mano_lr6e4():
    """Mano at 6e-4 (2x baseline)."""
    cfg = _r5_base()
    cfg.optimizer = default_mano(lr=6.0e-4)
    cfg.checkpoint.folder = "checkpoints/r5_mano_lr6e4"
    return cfg


def r5_mano_lr1e3():
    """Mano at 1e-3 (3.3x baseline)."""
    cfg = _r5_base()
    cfg.optimizer = default_mano(lr=1.0e-3)
    cfg.checkpoint.folder = "checkpoints/r5_mano_lr1e3"
    return cfg


def r5_mano_lr2e3():
    """Mano at 2e-3 (6.7x baseline — aggressive)."""
    cfg = _r5_base()
    cfg.optimizer = default_mano(lr=2.0e-3)
    cfg.checkpoint.folder = "checkpoints/r5_mano_lr2e3"
    return cfg


# ---- WSM: constant LR, checkpoint for later merging ----


def r5_adamw_constant_lr():
    """AdamW at constant LR (no decay) — for WSM checkpoint merging."""
    cfg = _r5_base()
    cfg.optimizer.param_groups[0].optimizer_kwargs["lr"] = 1.3e-3
    cfg.lr_scheduler.decay_ratio = 0.0  # no decay — stable phase only
    cfg.checkpoint.enable = True
    cfg.checkpoint.interval = 200
    cfg.checkpoint.folder = "checkpoints/r5_adamw_constant_lr"
    return cfg


def r5_mano_constant_lr():
    """Mano at constant LR (no decay) — for WSM checkpoint merging."""
    cfg = _r5_base()
    cfg.optimizer = default_mano(lr=3.0e-4)
    cfg.lr_scheduler.decay_ratio = 0.0
    cfg.checkpoint.enable = True
    cfg.checkpoint.interval = 200
    cfg.checkpoint.folder = "checkpoints/r5_mano_constant_lr"
    return cfg


# ---- Schedule-Free AdamW ----


def r5_schedulefree():
    """Schedule-Free AdamW — no LR schedule at all.

    Uses the schedule-free framework (arxiv 2405.15682) which eliminates
    the need for a learning rate schedule. LR is set 1-10x higher than
    with cosine/WSD (paper recommendation).
    """
    cfg = _r5_base()
    cfg.optimizer = default_schedule_free(
        lr=2.5e-3,  # ~2x higher than scheduled AdamW
        warmup_steps=200,
    )
    # No LR schedule needed — schedule-free handles it internally
    cfg.lr_scheduler.warmup_steps = 0
    cfg.lr_scheduler.decay_ratio = 0.0
    cfg.checkpoint.folder = "checkpoints/r5_schedulefree"
    return cfg


# ---- Smoke tests (post-21st-sync replay verification) ----


def smoke_2b_50steps():
    """50-step AdamW smoke test — verifies the post-#2963/#2937 replay.

    Mirrors speedrun configs (compile on, ac=none, LBS=2, seq_len=8192)
    but only 50 steps and warmup=5. Compile is required because plain
    cross_entropy on [B*T, vocab=256k] OOMs on a single XPU tile without
    the compiled chunked path. ~10 min wall time including compile.
    """
    cfg = agpt(
        "2b",
        local_batch_size=LOCAL_BATCH_SIZE,
        activation_checkpoint_mode="none",
        seq_len=SEQ_LEN,
        compile=True,
        checkpoint_interval=10_000,
    )
    cfg.dataloader.dataset = DATASET
    cfg.dataloader.dataset_path = None
    cfg.training.steps = 50
    cfg.checkpoint.enable = False
    cfg.optimizer.param_groups[0].optimizer_kwargs["lr"] = 1.3e-3
    cfg.lr_scheduler.warmup_steps = 5
    cfg.lr_scheduler.decay_ratio = 0.0
    cfg.metrics.log_freq = 1
    return cfg


def smoke_2b_async_ckpt():
    """50-step AdamW smoke that exercises async checkpointing end-to-end.

    Same model + optimizer as smoke_2b_50steps, but with:
    - checkpoint.enable = True
    - checkpoint.async_mode = "async" (background dcp.async_save)
    - checkpoint.enable_first_step_checkpoint = True (catches setup
      bugs at step 1 before training proceeds)
    - checkpoint.interval = 10 (so we get ~5 saves across the run and
      can confirm TPS doesn't dip during/after a save)
    - checkpoint.folder unique per run (no cross-run interference)

    Verifies the on-disk format is wire-compatible with sync DCP saves
    (it should be — async only changes when the bytes hit disk, not
    what gets written) and that resume from an async-saved checkpoint
    works.
    """
    import time

    cfg = agpt(
        "2b",
        local_batch_size=LOCAL_BATCH_SIZE,
        activation_checkpoint_mode="none",
        seq_len=SEQ_LEN,
        compile=True,
        checkpoint_interval=10,
    )
    cfg.dataloader.dataset = DATASET
    cfg.dataloader.dataset_path = None
    cfg.training.steps = 50
    cfg.optimizer.param_groups[0].optimizer_kwargs["lr"] = 1.3e-3
    cfg.lr_scheduler.warmup_steps = 5
    cfg.lr_scheduler.decay_ratio = 0.0
    cfg.metrics.log_freq = 1

    cfg.checkpoint.enable = True
    cfg.checkpoint.async_mode = "async"
    cfg.checkpoint.enable_first_step_checkpoint = True
    # Unique dir each run so concurrent submissions don't collide and
    # nobody resumes from a stale dir.
    cfg.checkpoint.folder = f"checkpoints/smoke_async_ckpt_{int(time.time())}"
    return cfg


def smoke_2b_async_ckpt_pinned():
    """Same as smoke_2b_async_ckpt but uses async_with_pinned_mem.

    Spawns a separate process for GPU->CPU transfer with pinned memory.
    Higher CPU memory pressure but near-zero in-band cost. More likely
    than plain `async` to expose XPU-side issues with pinned-memory
    allocation paths — run after the plain async test passes.
    """
    cfg = smoke_2b_async_ckpt()
    cfg.checkpoint.async_mode = "async_with_pinned_mem"
    cfg.checkpoint.folder = cfg.checkpoint.folder.replace(
        "smoke_async_ckpt_", "smoke_async_ckpt_pinned_"
    )
    return cfg
